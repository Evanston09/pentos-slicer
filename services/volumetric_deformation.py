from dataclasses import dataclass

import numpy as np
from rtree import index
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._isolve import lsqr
import tetgen
import trimesh
from trimesh import transformations as tf

from models import GuideSurfaceSnapshot
from services.auto_planes import quaternion_from_z_to

# Full-XYZ local-frame deformation follows S³ DeformFDM (BSD-3-Clause).
# Tetrahedral inverse mapping follows Joshua Bird's GPL-3.0 S4 Slicer.


@dataclass(frozen=True)
class BarycentricLocations:
    tetrahedron_indices: np.ndarray
    weights: np.ndarray


@dataclass
class TetrahedralVolume:
    original_vertices: np.ndarray
    deformed_vertices: np.ndarray
    tetrahedra: np.ndarray
    boundary_faces: np.ndarray

    def locate_original(self, points: np.ndarray) -> BarycentricLocations:
        """Locate points in the original volume and return barycentric coordinates."""
        return self._locate(points, self.original_vertices)

    def locate_deformed(self, points: np.ndarray) -> BarycentricLocations:
        """Locate points in the deformed volume and return barycentric coordinates."""
        return self._locate(points, self.deformed_vertices)

    def map_to_original(self, points: np.ndarray) -> np.ndarray:
        """Map points from the deformed volume back to the original volume."""
        locations = self.locate_deformed(points)
        return self._interpolate(locations, self.original_vertices)

    def map_to_deformed(self, points: np.ndarray) -> np.ndarray:
        """Map points from the original volume into the deformed volume."""
        locations = self.locate_original(points)
        return self._interpolate(locations, self.deformed_vertices)

    def _interpolate(
        self,
        locations: BarycentricLocations,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """Interpolate point positions from tetrahedron vertices and barycentric weights."""
        tetrahedra = self.tetrahedra[locations.tetrahedron_indices]
        return np.einsum("ni,nij->nj", locations.weights, vertices[tetrahedra])

    def _locate(
        self,
        points: np.ndarray,
        vertices: np.ndarray,
    ) -> BarycentricLocations:
        """Find containing tetrahedra using spatial and barycentric containment tests."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape (n, 3)")

        tetrahedron_points = vertices[self.tetrahedra]
        edges = np.stack(
            (
                tetrahedron_points[:, 1] - tetrahedron_points[:, 0],
                tetrahedron_points[:, 2] - tetrahedron_points[:, 0],
                tetrahedron_points[:, 3] - tetrahedron_points[:, 0],
            ),
            axis=2,
        )
        determinants = np.linalg.det(edges)
        usable = ~np.isclose(determinants, 0.0)
        inverse_edges = np.zeros_like(edges)
        inverse_edges[usable] = np.linalg.inv(edges[usable])

        properties = index.Property()
        properties.dimension = 3
        spatial_index = index.Index(properties=properties)
        lower = tetrahedron_points.min(axis=1)
        upper = tetrahedron_points.max(axis=1)
        for tetrahedron_index, (minimum, maximum) in enumerate(zip(lower, upper)):
            if usable[tetrahedron_index]:
                spatial_index.insert(
                    tetrahedron_index,
                    (*minimum, *maximum),
                )

        tolerance = 1e-9
        point_tolerance = tolerance * max(float(np.ptp(vertices)), 1.0)
        tetrahedron_indices = np.empty(len(points), dtype=np.int32)
        weights = np.empty((len(points), 4), dtype=np.float64)

        for point_index, point in enumerate(points):
            candidates = np.fromiter(
                spatial_index.intersection(
                    (*point - point_tolerance, *point + point_tolerance)
                ),
                dtype=np.int32,
            )
            if not len(candidates):
                raise ValueError(
                    f"Point {point_index} is outside the tetrahedral volume"
                )

            coordinates = np.einsum(
                "nij,nj->ni",
                inverse_edges[candidates],
                point - tetrahedron_points[candidates, 0],
            )
            candidate_weights = np.column_stack(
                (1.0 - coordinates.sum(axis=1), coordinates)
            )
            contained = np.all(candidate_weights >= -tolerance, axis=1) & np.all(
                candidate_weights <= 1.0 + tolerance,
                axis=1,
            )
            if not np.any(contained):
                raise ValueError(
                    f"Point {point_index} is outside the tetrahedral volume"
                )

            # ponytail: S4-compatible prototype behavior: if folded tetrahedra
            # overlap, the first containing cell wins. Replace with injective
            # deformation if overlapping mappings become a practical problem.
            match = int(np.flatnonzero(contained)[0])
            tetrahedron_indices[point_index] = candidates[match]
            point_weights = np.clip(candidate_weights[match], 0.0, 1.0)
            weights[point_index] = point_weights / point_weights.sum()

        return BarycentricLocations(tetrahedron_indices, weights)


def _guide_field(
    points: np.ndarray,
    guides: list[GuideSurfaceSnapshot],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate layer normals and flattened heights from adjacent guide surfaces."""
    ordered = sorted(guides, key=lambda item: item.guide_id)
    if not ordered:
        return np.repeat([[0.0, 0.0, 1.0]], len(points), axis=0), np.zeros(len(points))

    residuals = []
    normals = []
    for guide in ordered:
        rotation = tf.quaternion_matrix(guide.wxyz)[:3, :3]
        local = (points - guide.position) @ rotation
        residuals.append(
            local[:, 2]
            - guide.bend_x * local[:, 0] ** 2
            - guide.bend_y * local[:, 1] ** 2
        )
        local_normals = np.column_stack(
            (
                -2.0 * guide.bend_x * local[:, 0],
                -2.0 * guide.bend_y * local[:, 1],
                np.ones(len(points)),
            )
        )
        world_normals = local_normals @ rotation.T
        normals.append(
            world_normals / np.linalg.norm(world_normals, axis=1, keepdims=True)
        )

    residuals = np.asarray(residuals)
    normals = np.asarray(normals)
    if len(ordered) == 1:
        return normals[0], np.zeros(len(points))

    positions = np.asarray([guide.position for guide in ordered])
    segments = positions[1:] - positions[:-1]
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(np.isclose(segment_lengths, 0.0)):
        raise ValueError("Adjacent guides must have different positions")
    relative = points[:, None, :] - positions[:-1][None, :, :]
    centerline_amounts = np.clip(
        np.einsum("psi,si->ps", relative, segments) / segment_lengths**2,
        0.0,
        1.0,
    )
    closest = (
        positions[:-1][None, :, :]
        + centerline_amounts[:, :, None] * segments[None, :, :]
    )
    segment_indices = np.argmin(
        np.sum((points[:, None, :] - closest) ** 2, axis=2),
        axis=1,
    )
    point_indices = np.arange(len(points))
    first_residuals = residuals[segment_indices, point_indices]
    second_residuals = residuals[segment_indices + 1, point_indices]
    denominator = first_residuals - second_residuals
    fallback = centerline_amounts[point_indices, segment_indices]
    amounts = np.divide(
        first_residuals,
        denominator,
        out=fallback.copy(),
        where=~np.isclose(denominator, 0.0),
    )
    amounts = np.clip(amounts, 0.0, 1.0)

    field_normals = (1.0 - amounts[:, None]) * normals[
        segment_indices, point_indices
    ] + amounts[:, None] * normals[segment_indices + 1, point_indices]
    field_normals /= np.linalg.norm(field_normals, axis=1, keepdims=True)
    guide_heights = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    heights = (
        guide_heights[segment_indices] + amounts * segment_lengths[segment_indices]
    )
    return field_normals, heights


def guide_target_rotations(
    volume: TetrahedralVolume,
    guides: list[GuideSurfaceSnapshot],
) -> np.ndarray:
    """Compute a smoothed target rotation for every tetrahedron from the guide field."""
    centroids = volume.original_vertices[volume.tetrahedra].mean(axis=1)
    normals, _ = _guide_field(centroids, guides)
    neighbors = _tetrahedron_neighbors(volume.tetrahedra)
    for _ in range(2):
        accumulated = normals.copy()
        counts = np.ones((len(normals), 1))
        if len(neighbors):
            np.add.at(accumulated, neighbors[:, 0], normals[neighbors[:, 1]])
            np.add.at(accumulated, neighbors[:, 1], normals[neighbors[:, 0]])
            np.add.at(counts[:, 0], neighbors[:, 0], 1.0)
            np.add.at(counts[:, 0], neighbors[:, 1], 1.0)
        normals = accumulated / counts
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return np.asarray(
        [
            tf.quaternion_matrix(quaternion_from_z_to(normal))[:3, :3].T
            for normal in normals
        ]
    )


def solve_guide_deformation(
    volume: TetrahedralVolume,
    guides: list[GuideSurfaceSnapshot],
) -> np.ndarray:
    """Deform a tetrahedral volume so its guide field becomes flat in Z."""
    if len(guides) < 2:
        raise ValueError("Add at least two guides to define the flattened layer range")
    solve_deformation(volume, guide_target_rotations(volume, guides))
    _, heights = _guide_field(volume.original_vertices, guides)
    volume.deformed_vertices[:, 2] = heights
    return volume.deformed_vertices


def solve_deformation(
    volume: TetrahedralVolume,
    target_rotations: np.ndarray,
) -> np.ndarray:
    """Solve shared vertex positions that best match each tetrahedron's target rotation."""
    tetrahedron_count = len(volume.tetrahedra)
    target_rotations = np.asarray(target_rotations, dtype=np.float64)
    if target_rotations.shape != (tetrahedron_count, 3, 3):
        raise ValueError("Target rotations must have shape (tetrahedra, 3, 3)")
    if not np.allclose(
        target_rotations @ np.swapaxes(target_rotations, 1, 2),
        np.eye(3),
        atol=1e-6,
    ) or not np.allclose(np.linalg.det(target_rotations), 1.0, atol=1e-6):
        raise ValueError("Target matrices must be proper 3D rotations")

    centering = np.eye(4) - np.full((4, 4), 0.25)
    original_tetrahedra = volume.original_vertices[volume.tetrahedra]
    targets = (centering @ original_tetrahedra) @ np.swapaxes(target_rotations, 1, 2)
    rows = np.repeat(np.arange(tetrahedron_count * 4), 4)
    columns = np.repeat(volume.tetrahedra, 4, axis=0).ravel()
    system = coo_matrix(
        (np.tile(centering.ravel(), tetrahedron_count), (rows, columns)),
        shape=(tetrahedron_count * 4, len(volume.original_vertices)),
    ).tocsr()
    deformed = np.column_stack(
        [
            lsqr(
                system,
                targets[:, :, dimension].ravel(),
                atol=1e-12,
                btol=1e-12,
            )[0]
            for dimension in range(3)
        ]
    )
    deformed += volume.original_vertices[0] - deformed[0]
    volume.deformed_vertices = deformed
    return deformed


def tetrahedralize(mesh: trimesh.Trimesh) -> TetrahedralVolume:
    """Fill a watertight surface mesh with valid tetrahedra and recover its boundary."""
    if not mesh.is_volume:
        raise ValueError(
            "Nonplanar slicing requires a watertight, consistently wound mesh"
        )

    vertices, tetrahedra, _, _ = tetgen.TetGen(
        mesh.vertices,
        mesh.faces,
    ).tetrahedralize(
        quality=True,
        # TetGen can crash while splitting dense CAD boundaries.
        nobisect=True,
    )

    volumes = _tetrahedron_determinants(vertices, tetrahedra)
    if len(tetrahedra) == 0 or np.any(volumes <= 0.0):
        raise ValueError("TetGen produced invalid tetrahedra")

    faces = tetrahedra[:, ([1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1])].reshape(-1, 3)
    _, first_indices, counts = np.unique(
        np.sort(faces, axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    boundary_faces = faces[first_indices[counts == 1]]
    boundary = trimesh.Trimesh(vertices=vertices, faces=boundary_faces, process=False)
    if (
        not boundary.is_volume
        or not np.allclose(boundary.bounds, mesh.bounds)
        or not np.isclose(boundary.volume, mesh.volume)
    ):
        raise ValueError("TetGen boundary does not match the input mesh")

    return TetrahedralVolume(
        original_vertices=vertices,
        deformed_vertices=vertices.copy(),
        tetrahedra=tetrahedra,
        boundary_faces=boundary_faces,
    )


def _tetrahedron_neighbors(tetrahedra: np.ndarray) -> np.ndarray:
    """Return pairs of tetrahedra that share a triangular face."""
    face_owners: dict[tuple[int, int, int], int] = {}
    neighbors = []
    for tetrahedron_index, tetrahedron in enumerate(tetrahedra):
        for face in (
            tetrahedron[[1, 2, 3]],
            tetrahedron[[0, 2, 3]],
            tetrahedron[[0, 1, 3]],
            tetrahedron[[0, 1, 2]],
        ):
            first, second, third = sorted(int(vertex) for vertex in face)
            key = (first, second, third)
            owner = face_owners.pop(key, -1)
            if owner == -1:
                face_owners[key] = tetrahedron_index
            else:
                neighbors.append((owner, tetrahedron_index))
    return np.asarray(neighbors, dtype=np.int32).reshape(-1, 2)


def _tetrahedron_determinants(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
) -> np.ndarray:
    """Calculate signed determinant values proportional to tetrahedron volumes."""
    points = vertices[tetrahedra]
    return np.linalg.det(
        np.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ),
            axis=2,
        )
    )
