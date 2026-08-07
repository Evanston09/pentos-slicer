from dataclasses import dataclass, field

import numpy as np
from rtree import index
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg._isolve import lsqr
import tetgen
import trimesh
from trimesh import transformations as tf

from models import GuideSurfaceSnapshot
from services.auto_planes import quaternion_from_z_to

# Full-XYZ local-frame deformation follows S³ DeformFDM (BSD-3-Clause).
# Tetrahedral inverse mapping follows Joshua Bird's GPL-3.0 S4 Slicer.

BARYCENTRIC_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BarycentricLocations:
    tetrahedron_indices: np.ndarray
    weights: np.ndarray


@dataclass
class _TetrahedronLocator:
    points: np.ndarray
    inverse_edges: np.ndarray
    spatial_index: index.Index
    point_tolerance: float


@dataclass
class TetrahedralVolume:
    original_vertices: np.ndarray
    deformed_vertices: np.ndarray
    tetrahedra: np.ndarray
    boundary_faces: np.ndarray
    scalar_values: np.ndarray | None = None
    _original_locator: _TetrahedronLocator | None = field(
        default=None, init=False, repr=False
    )
    _deformed_locator: _TetrahedronLocator | None = field(
        default=None, init=False, repr=False
    )

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

    def layer_normals(self, points: np.ndarray) -> np.ndarray:
        """Return scalar-field normals for points in the original volume."""
        if self.scalar_values is None:
            raise ValueError("The volume has no solved layer field")
        locations = self.locate_original(points)
        gradients = self.scalar_gradients()[locations.tetrahedron_indices]
        return gradients / np.linalg.norm(gradients, axis=1, keepdims=True)

    def extrusion_multipliers(self, points: np.ndarray) -> np.ndarray:
        """Return original-to-deformed tetrahedron volume ratios at points."""
        locations = self.locate_deformed(points)
        tetrahedra = self.tetrahedra[locations.tetrahedron_indices]
        original = np.abs(_tetrahedron_determinants(self.original_vertices, tetrahedra))
        deformed = np.abs(_tetrahedron_determinants(self.deformed_vertices, tetrahedra))
        return original / deformed

    def scalar_gradients(self) -> np.ndarray:
        """Return the constant scalar gradient inside every tetrahedron."""
        if self.scalar_values is None:
            raise ValueError("The volume has no solved layer field")
        differences = (
            self.scalar_values[self.tetrahedra[:, 1:]]
            - self.scalar_values[self.tetrahedra[:, :1]]
        )
        return np.einsum(
            "nji,nj->ni",
            self._locator(self.original_vertices).inverse_edges,
            differences,
        )

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

        locator = self._locator(vertices)
        tetrahedron_points = locator.points
        inverse_edges = locator.inverse_edges
        spatial_index = locator.spatial_index
        tolerance = BARYCENTRIC_TOLERANCE
        tetrahedron_indices = np.empty(len(points), dtype=np.int32)
        weights = np.empty((len(points), 4), dtype=np.float64)

        for point_index, point in enumerate(points):
            candidates = np.fromiter(
                spatial_index.intersection(
                    (
                        *point - locator.point_tolerance,
                        *point + locator.point_tolerance,
                    )
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

            # ponytail: preview-only fallback; machine output must reject folds.
            match = int(np.flatnonzero(contained)[0])
            tetrahedron_indices[point_index] = candidates[match]
            point_weights = np.clip(candidate_weights[match], 0.0, 1.0)
            weights[point_index] = point_weights / point_weights.sum()

        return BarycentricLocations(tetrahedron_indices, weights)

    def _locator(self, vertices: np.ndarray) -> _TetrahedronLocator:
        """Build and cache the spatial data used to locate points in a volume."""
        attribute = (
            "_original_locator"
            if vertices is self.original_vertices
            else "_deformed_locator"
        )
        cached = getattr(self, attribute)
        if cached is not None:
            return cached

        points = vertices[self.tetrahedra]
        edges = np.stack(
            (
                points[:, 1] - points[:, 0],
                points[:, 2] - points[:, 0],
                points[:, 3] - points[:, 0],
            ),
            axis=2,
        )
        usable = ~np.isclose(np.linalg.det(edges), 0.0)
        inverse_edges = np.zeros_like(edges)
        inverse_edges[usable] = np.linalg.inv(edges[usable])
        properties = index.Property()
        properties.dimension = 3
        spatial_index = index.Index(properties=properties)
        for tetrahedron_index in np.flatnonzero(usable):
            minimum = points[tetrahedron_index].min(axis=0)
            maximum = points[tetrahedron_index].max(axis=0)
            spatial_index.insert(int(tetrahedron_index), (*minimum, *maximum))

        locator = _TetrahedronLocator(
            points,
            inverse_edges,
            spatial_index,
            BARYCENTRIC_TOLERANCE * max(float(np.ptp(vertices)), 1.0),
        )
        setattr(self, attribute, locator)
        return locator


def solve_guide_scalar_field(
    volume: TetrahedralVolume,
    guides: list[GuideSurfaceSnapshot],
) -> np.ndarray:
    """Solve the smooth scalar field whose level sets include the guide surfaces."""
    ordered = sorted(guides, key=lambda item: item.guide_id)
    if len(ordered) < 2:
        raise ValueError("Add at least two guides to define the flattened layer range")

    positions = np.asarray([guide.position for guide in ordered])
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.any(np.isclose(segment_lengths, 0.0)):
        raise ValueError("Adjacent guides must have different positions")
    guide_heights = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    edges = np.unique(
        np.sort(
            volume.tetrahedra[:, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])].reshape(
                -1, 2
            ),
            axis=1,
        ),
        axis=0,
    )
    constraints, targets = _guide_constraints(
        volume.original_vertices,
        edges,
        ordered,
        guide_heights,
    )

    points = volume.original_vertices[volume.tetrahedra]
    edge_matrices = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )
    inverse_edges = np.linalg.inv(edge_matrices)
    basis_gradients = np.concatenate(
        (-inverse_edges.sum(axis=1, keepdims=True), inverse_edges),
        axis=1,
    )
    weights = np.sqrt(np.abs(np.linalg.det(edge_matrices)) / 6.0)
    rows = np.repeat(np.arange(len(volume.tetrahedra) * 3), 4)
    columns = np.tile(volume.tetrahedra, (1, 3)).ravel()
    smoothness = coo_matrix(
        (
            (basis_gradients * weights[:, None, None]).transpose(0, 2, 1).ravel(),
            (rows, columns),
        ),
        shape=(len(volume.tetrahedra) * 3, len(volume.original_vertices)),
    ).tocsr()

    guide_normals = np.asarray(
        [tf.quaternion_matrix(guide.wxyz)[:3, 2] for guide in ordered]
    )
    preferred_gradient = guide_normals.mean(axis=0)
    preferred_gradient /= np.linalg.norm(preferred_gradient)
    smoothness_targets = (weights[:, None] * preferred_gradient).ravel()

    constraint_weight = 100.0
    system = vstack((smoothness, constraints * constraint_weight), format="csr")
    right_hand_side = np.concatenate((smoothness_targets, targets * constraint_weight))
    scalar_values = lsqr(
        system,
        right_hand_side,
        atol=1e-12,
        btol=1e-12,
    )[0]
    volume.scalar_values = scalar_values

    gradient_lengths = np.linalg.norm(volume.scalar_gradients(), axis=1)
    if np.any(gradient_lengths < 1e-8):
        raise ValueError("Guide field contains a zero-gradient region")
    return scalar_values


def solve_guide_deformation(
    volume: TetrahedralVolume,
    guides: list[GuideSurfaceSnapshot],
) -> np.ndarray:
    """Flatten a guide-constrained scalar field through a tetrahedral volume."""
    heights = solve_guide_scalar_field(volume, guides)
    gradients = volume.scalar_gradients()
    volumes = np.abs(
        _tetrahedron_determinants(volume.original_vertices, volume.tetrahedra)
    )
    average_normal = np.average(gradients, axis=0, weights=volumes)
    average_normal /= np.linalg.norm(average_normal)
    rotation = tf.quaternion_matrix(quaternion_from_z_to(average_normal))[:3, :3]
    deformed_xy = volume.original_vertices @ rotation
    deformed_xy += volume.original_vertices[0] - deformed_xy[0]
    volume.deformed_vertices = np.column_stack((deformed_xy[:, :2], heights))
    volume._deformed_locator = None
    if np.any(
        _tetrahedron_determinants(volume.deformed_vertices, volume.tetrahedra) <= 0.0
    ):
        raise ValueError("Guide field reverses direction inside the model")
    return volume.deformed_vertices


def _guide_constraints(
    vertices: np.ndarray,
    edges: np.ndarray,
    guides: list[GuideSurfaceSnapshot],
    guide_heights: np.ndarray,
) -> tuple[coo_matrix, np.ndarray]:
    """Build scalar-value constraints where guide surfaces cross mesh edges."""
    row_indices = []
    column_indices = []
    values = []
    targets = []
    endpoints = vertices[edges]
    tolerance = 1e-10

    for guide, height in zip(guides, guide_heights):
        rotation = tf.quaternion_matrix(guide.wxyz)[:3, :3]
        local = (endpoints - guide.position) @ rotation
        start = local[:, 0]
        delta = local[:, 1] - start
        quadratic = -guide.bend_x * delta[:, 0] ** 2 - guide.bend_y * delta[:, 1] ** 2
        linear = (
            delta[:, 2]
            - 2.0 * guide.bend_x * start[:, 0] * delta[:, 0]
            - 2.0 * guide.bend_y * start[:, 1] * delta[:, 1]
        )
        constant = (
            start[:, 2]
            - guide.bend_x * start[:, 0] ** 2
            - guide.bend_y * start[:, 1] ** 2
        )
        roots: list[tuple[int, float]] = []
        for edge_index in np.flatnonzero(
            np.isclose(quadratic, 0.0, atol=tolerance)
            & ~np.isclose(linear, 0.0, atol=tolerance)
        ):
            roots.append(
                (int(edge_index), float(-constant[edge_index] / linear[edge_index]))
            )
        discriminants = linear**2 - 4.0 * quadratic * constant
        for edge_index in np.flatnonzero(
            ~np.isclose(quadratic, 0.0, atol=tolerance) & (discriminants >= 0.0)
        ):
            root = np.sqrt(discriminants[edge_index])
            roots.extend(
                (
                    (
                        int(edge_index),
                        float(
                            (-linear[edge_index] - root) / (2.0 * quadratic[edge_index])
                        ),
                    ),
                    (
                        int(edge_index),
                        float(
                            (-linear[edge_index] + root) / (2.0 * quadratic[edge_index])
                        ),
                    ),
                )
            )
        for edge_index in np.flatnonzero(
            np.isclose(quadratic, 0.0, atol=tolerance)
            & np.isclose(linear, 0.0, atol=tolerance)
            & np.isclose(constant, 0.0, atol=tolerance)
        ):
            roots.extend(((int(edge_index), 0.0), (int(edge_index), 1.0)))

        accepted = [
            (edge_index, np.clip(amount, 0.0, 1.0))
            for edge_index, amount in roots
            if -tolerance <= amount <= 1.0 + tolerance
        ]
        if not accepted:
            raise ValueError(f"Guide {guide.guide_id} does not intersect the model")
        for edge_index, amount in accepted:
            row = len(targets)
            first, second = edges[edge_index]
            row_indices.extend((row, row))
            column_indices.extend((first, second))
            values.extend((1.0 - amount, amount))
            targets.append(height)

    return (
        coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(targets), len(vertices)),
        ).tocsr(),
        np.asarray(targets),
    )


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
