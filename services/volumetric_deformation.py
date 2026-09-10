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
# Neighboring-gradient continuity adapts S4 Slicer's neighboring-cell smoothing idea.

BARYCENTRIC_TOLERANCE = 1e-6
MAX_EXTRAPOLATION_DISTANCE = 1.0


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
    _layer_vertex_gradients: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    def locate_original(self, points: np.ndarray) -> BarycentricLocations:
        """Locate points in the original volume and return barycentric coordinates."""
        return self._locate(points, self.original_vertices)

    def locate_deformed(
        self,
        points: np.ndarray,
        *,
        allow_extrapolation: bool = False,
    ) -> BarycentricLocations:
        """Locate points in the deformed volume and return barycentric coordinates."""
        return self._locate(
            points,
            self.deformed_vertices,
            allow_extrapolation=allow_extrapolation,
        )

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
        return self._layer_normals(self.locate_original(points))

    def inverse_map_properties(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extrapolate nearby toolpaths and return positions, flow ratios, and normals."""
        locations = self.locate_deformed(points, allow_extrapolation=True)
        return (
            self._interpolate(locations, self.original_vertices),
            self._extrusion_multipliers(locations),
            self._layer_normals(locations),
        )

    def _layer_normals(self, locations: BarycentricLocations) -> np.ndarray:
        if self.scalar_values is None:
            raise ValueError("The volume has no solved layer field")
        if self._layer_vertex_gradients is None:
            tetrahedron_gradients = self.scalar_gradients()
            volumes = np.abs(
                _tetrahedron_determinants(self.original_vertices, self.tetrahedra)
            )
            vertex_gradients = np.zeros_like(self.original_vertices)
            vertex_weights = np.zeros(len(self.original_vertices))
            np.add.at(
                vertex_gradients,
                self.tetrahedra.ravel(),
                np.repeat(tetrahedron_gradients * volumes[:, None], 4, axis=0),
            )
            np.add.at(
                vertex_weights,
                self.tetrahedra.ravel(),
                np.repeat(volumes, 4),
            )
            self._layer_vertex_gradients = vertex_gradients / vertex_weights[:, None]
        gradients = self._interpolate(locations, self._layer_vertex_gradients)
        return gradients / np.linalg.norm(gradients, axis=1, keepdims=True)

    def extrusion_multipliers(self, points: np.ndarray) -> np.ndarray:
        """Return original-to-deformed tetrahedron volume ratios at points."""
        return self._extrusion_multipliers(self.locate_deformed(points))

    def _extrusion_multipliers(
        self,
        locations: BarycentricLocations,
    ) -> np.ndarray:
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
        *,
        allow_extrapolation: bool = False,
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
            if allow_extrapolation:
                candidates = np.fromiter(
                    spatial_index.nearest((*point, *point), 256),
                    dtype=np.int32,
                )
            else:
                candidates = np.fromiter(
                    spatial_index.intersection(
                        (
                            *point - locator.point_tolerance,
                            *point + locator.point_tolerance,
                        )
                    ),
                    dtype=np.int32,
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

            if np.any(contained):
                match = int(np.flatnonzero(contained)[0])
                tetrahedron_indices[point_index] = candidates[match]
                point_weights = np.clip(candidate_weights[match], 0.0, 1.0)
                weights[point_index] = point_weights / point_weights.sum()
                continue

            if allow_extrapolation:
                clipped = np.clip(candidate_weights, 0.0, 1.0)
                clipped /= clipped.sum(axis=1, keepdims=True)
                projected = np.einsum(
                    "ni,nij->nj",
                    clipped,
                    tetrahedron_points[candidates],
                )
                distances = np.linalg.norm(projected - point, axis=1)
                match = int(distances.argmin())
                if distances[match] <= MAX_EXTRAPOLATION_DISTANCE:
                    tetrahedron_indices[point_index] = candidates[match]
                    weights[point_index] = candidate_weights[match]
                    continue

            raise ValueError(f"Point {point_index} is outside the tetrahedral volume")

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
    if len(guides) < 2:
        raise ValueError("Add at least two guides to define the flattened layer range")

    positions = np.asarray([guide.position for guide in guides])
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.any(np.isclose(segment_lengths, 0.0)):
        raise ValueError("Adjacent guides must have different positions")
    guide_heights = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    edge_pairs = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    edges = np.unique(
        np.sort(volume.tetrahedra[:, edge_pairs].reshape(-1, 2), axis=1), axis=0
    )
    constraints, targets = _guide_constraints(
        volume.original_vertices,
        edges,
        guides,
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
    tetrahedron_volumes = np.abs(np.linalg.det(edge_matrices)) / 6.0
    preferred_weights = np.sqrt(tetrahedron_volumes)
    rows = np.repeat(np.arange(len(volume.tetrahedra) * 3), 4)
    columns = np.tile(volume.tetrahedra, (1, 3)).ravel()
    smoothness = coo_matrix(
        (
            (basis_gradients * preferred_weights[:, None, None])
            .transpose(0, 2, 1)
            .ravel(),
            (rows, columns),
        ),
        shape=(len(volume.tetrahedra) * 3, len(volume.original_vertices)),
    ).tocsr()

    faces = np.sort(
        volume.tetrahedra[:, ([1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1])].reshape(
            -1, 3
        ),
        axis=1,
    )
    unique_faces, inverse, counts = np.unique(
        faces,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    interior_faces = np.flatnonzero(counts == 2)
    face_order = np.argsort(inverse)
    face_starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    owners = np.repeat(np.arange(len(volume.tetrahedra)), 4)
    neighbours = np.column_stack(
        (
            owners[face_order[face_starts[interior_faces]]],
            owners[face_order[face_starts[interior_faces] + 1]],
        )
    )

    shared_points = volume.original_vertices[unique_faces[interior_faces]]
    shared_areas = 0.5 * np.linalg.norm(
        np.cross(
            shared_points[:, 1] - shared_points[:, 0],
            shared_points[:, 2] - shared_points[:, 0],
        ),
        axis=1,
    )
    centers = points.mean(axis=1)
    center_distances = np.linalg.norm(
        centers[neighbours[:, 0]] - centers[neighbours[:, 1]],
        axis=1,
    )
    # Face area / center distance is the finite-volume coupling: small faces and
    # distant cells contribute less. Normalization keeps it comparable to the
    # volume-weighted preferred-gradient objective at any model scale.
    neighbour_weights = shared_areas / center_distances
    if len(neighbour_weights):
        neighbour_weights *= tetrahedron_volumes.sum() / neighbour_weights.sum()
    neighbour_basis = np.concatenate(
        (
            basis_gradients[neighbours[:, 0]],
            -basis_gradients[neighbours[:, 1]],
        ),
        axis=1,
    )
    neighbour_vertices = np.concatenate(
        (
            volume.tetrahedra[neighbours[:, 0]],
            volume.tetrahedra[neighbours[:, 1]],
        ),
        axis=1,
    )
    continuity = coo_matrix(
        (
            (neighbour_basis * np.sqrt(neighbour_weights)[:, None, None])
            .transpose(0, 2, 1)
            .ravel(),
            (
                np.repeat(np.arange(len(neighbours) * 3), 8),
                np.tile(neighbour_vertices, (1, 3)).ravel(),
            ),
        ),
        shape=(len(neighbours) * 3, len(volume.original_vertices)),
    ).tocsr()

    guide_normals = np.asarray(
        [tf.quaternion_matrix(guide.wxyz)[:3, 2] for guide in guides]
    )
    preferred_gradient = guide_normals.mean(axis=0)
    preferred_gradient /= np.linalg.norm(preferred_gradient)
    smoothness_targets = (preferred_weights[:, None] * preferred_gradient).ravel()

    constraint_weight = 100.0
    system = vstack(
        (smoothness, continuity, constraints * constraint_weight),
        format="csr",
    )
    right_hand_side = np.concatenate(
        (
            smoothness_targets,
            np.zeros(len(neighbours) * 3),
            targets * constraint_weight,
        )
    )
    scalar_values = lsqr(
        system,
        right_hand_side,
        atol=1e-12,
        btol=1e-12,
    )[0]
    volume.scalar_values = scalar_values
    volume._layer_vertex_gradients = None

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
    for guide, height in zip(guides, guide_heights):
        first_row = len(targets)
        for edge, roots in zip(
            edges, _guide_edge_intersections(vertices, edges, guide)
        ):
            for amount in roots:
                row = len(targets)
                row_indices.extend((row, row))
                column_indices.extend(edge)
                values.extend((1.0 - amount, amount))
                targets.append(height)
        if len(targets) == first_row:
            raise ValueError(f"Guide {guide.guide_id} does not intersect the model")

    return (
        coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(targets), len(vertices)),
        ).tocsr(),
        np.asarray(targets),
    )


def _guide_edge_intersections(
    vertices: np.ndarray,
    edges: np.ndarray,
    guide: GuideSurfaceSnapshot,
) -> list[list[float]]:
    """Intersect volume edges with the finite bicubic guide patch."""
    endpoints = vertices[edges]
    local = (endpoints - guide.position) @ guide.rotation
    start = local[:, 0, :2]
    delta = local[:, 1, :2] - start
    half_size = guide.size_mm / 2.0
    moving = np.abs(delta) > 1e-14
    lower = np.divide(
        -half_size - start, delta, out=np.full_like(delta, -np.inf), where=moving
    )
    upper = np.divide(
        half_size - start, delta, out=np.full_like(delta, np.inf), where=moving
    )
    # Clip to the displayed patch before solving, including coplanar edges.
    first = np.maximum(np.minimum(lower, upper).max(axis=1), 0.0)
    last = np.minimum(np.maximum(lower, upper).min(axis=1), 1.0)
    active = np.flatnonzero(
        (last >= first) & np.all(moving | (np.abs(start) <= half_size + 1e-9), axis=1)
    )
    results: list[list[float]] = [[] for _ in edges]
    if len(active) == 0:
        return results

    # A tensor-product cubic restricted to a line has degree at most six.
    # Recover that polynomial, so nearby crossings need no sampling brackets.
    nodes = np.cos(np.pi * (np.arange(7) + 0.5) / 7.0)
    amounts = first[active, None] + (nodes + 1.0) * (
        (last[active] - first[active])[:, None] / 2.0
    )
    points = (
        endpoints[active, :1]
        + (endpoints[active, 1:] - endpoints[active, :1]) * amounts[:, :, None]
    )
    coefficients = np.polynomial.chebyshev.chebfit(
        nodes, guide.signed_height(points).T, 6
    ).T
    for edge_index, coefficients_on_edge in zip(active, coefficients):
        tolerance = 1e-12 * max(1.0, np.abs(coefficients_on_edge).max())
        polynomial = np.polynomial.chebyshev.chebtrim(
            coefficients_on_edge, tol=tolerance
        )
        if np.abs(polynomial).max() <= tolerance:
            roots = np.array([-1.0, 1.0])
        else:
            roots = np.polynomial.chebyshev.chebroots(polynomial)
            roots = roots.real[np.abs(roots.imag) <= 1e-7]
            roots = np.clip(
                roots[(roots >= -1.0 - 1e-9) & (roots <= 1.0 + 1e-9)], -1, 1
            )
        amounts = np.sort(
            first[edge_index]
            + (roots + 1.0) * (last[edge_index] - first[edge_index]) / 2.0
        )
        results[edge_index] = (
            amounts[np.concatenate(([True], np.diff(amounts) > 1e-9))].tolist()
            if len(amounts)
            else []
        )
    return results


def tetrahedralize(mesh: trimesh.Trimesh) -> TetrahedralVolume:
    """Fill a watertight surface mesh with valid tetrahedra and recover its boundary."""
    if not mesh.is_volume:
        raise ValueError(
            "Nonplanar slicing requires a watertight, consistently wound mesh"
        )

    vertices, tetrahedra, _, _ = tetgen.TetGen(
        mesh.vertices,
        mesh.faces,
    ).tetrahedralize()

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
