from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from trimesh import transformations as tf

from services.slicing import (
    DEFAULT_CONFIG_PATH,
    SlicePlane,
    Slicer,
    decompose_mesh,
)


def _read_prusa_float(path: Path, key: str, fallback: float) -> float:
    """Read a numeric PrusaSlicer setting, returning a fallback on failure."""
    try:
        for line in path.read_text().splitlines():
            setting, _, value = line.partition("=")
            if setting.strip() == key:
                return float(value.strip().rstrip("%"))
    except (OSError, ValueError):
        pass

    return fallback


DEFAULT_PRINT_DIRECTION = np.array([0.0, 0.0, 1.0])
EPSILON = 1e-9
BED_Z = 0.0
BED_CONTACT_TOLERANCE = 1e-6
NOZZLE_BED_CLEARANCE = 5.0
OVERHANG_WEIGHT = 0.67
QUALITY_WEIGHT = 0.24
REINDEX_WEIGHT = 0.07
PLANE_REINDEX_WEIGHT = 1.0 / 3.0
CHUNK_REINDEX_WEIGHT = 1.0 / 3.0


@dataclass(frozen=True)
class AutoPlaneConfig:
    # Maximum number of slice planes the search may return.
    max_planes: int = 2
    # Number of best partial solutions retained after each search round.
    beam_width: int = 6
    # Maximum bed tilt that a candidate plane may require.
    max_tilt_degrees: float = 90.0
    # Surface angle below which a downward-facing face counts as an overhang.
    overhang_threshold_degrees: float = field(
        default_factory=lambda: _read_prusa_float(
            DEFAULT_CONFIG_PATH,
            "support_material_threshold",
            40.0,
        )
    )
    # Number of hemisphere directions sampled in addition to cardinal directions.
    normal_samples: int = 12
    # Number of possible cut positions considered for each direction.
    offsets_per_normal: int = 24
    # Number of top first-plane candidates reused in later search rounds.
    deep_candidate_pool: int = 24


@dataclass
class AutoPlaneCandidate:
    position: np.ndarray
    normal: np.ndarray

    @property
    def wxyz(self) -> np.ndarray:
        """Return the quaternion that rotates print-up to the plane normal."""
        return _quaternion_from_z_to(self.normal)

    def offset(self) -> float:
        """Return the plane's signed offset from the origin."""
        return float(self.normal @ self.position)

    def key(self) -> tuple[float, ...]:
        """Return a rounded key used to identify equivalent planes."""
        return tuple(np.round([*self.normal, self.offset()], 4))


@dataclass
class _SearchState:
    planes: list[AutoPlaneCandidate]
    total: float
    chunk_count: int


def overhang_preview_mesh(
    mesh: trimesh.Trimesh,
    planes: list[SlicePlane],
    threshold_degrees: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build a decomposed preview mesh and mark its overhanging faces."""
    # Split intersected triangles so highlighting changes exactly at each plane.
    pieces = decompose_mesh(mesh, planes, cap=False)
    overhang_masks = []

    for piece in pieces:
        normal = piece.print_up_normal
        direction = DEFAULT_PRINT_DIRECTION if normal is None else _normalize(normal)
        floor_height = (
            BED_Z if normal is None else float((piece.mesh.vertices @ direction).min())
        )
        _, _, overhangs = _classify_piece_faces(
            piece.mesh,
            direction,
            floor_height,
            threshold_degrees,
        )
        overhang_masks.append(overhangs)

    return (
        trimesh.util.concatenate([piece.mesh for piece in pieces]),
        np.concatenate(overhang_masks),
    )


def _evaluate_state(
    mesh: trimesh.Trimesh,
    planes: list[AutoPlaneCandidate],
    config: AutoPlaneConfig,
) -> _SearchState:
    """Score a plane set by overhangs, surface quality, and reindexing cost."""
    mesh_pieces = decompose_mesh(mesh, planes, cap=False)
    total_surface_area = float(mesh.area)

    overhang_area = 0.0
    quality = 0.0

    for piece in mesh_pieces:
        direction = (
            DEFAULT_PRINT_DIRECTION
            if piece.print_up_normal is None
            else _normalize(piece.print_up_normal)
        )
        face_areas = piece.mesh.area_faces
        floor_height = BED_Z
        if piece.print_up_normal is not None:
            floor_height = float((piece.mesh.vertices @ direction).min())
        dots, bed_supported, risky = _classify_piece_faces(
            piece.mesh,
            direction,
            floor_height,
            config.overhang_threshold_degrees,
        )

        overhang_area += float(face_areas[risky].sum())

        quality_faces = ~bed_supported
        quality += float(
            (face_areas[quality_faces] * np.abs(dots[quality_faces])).sum()
        )

    plane_count = len(planes)
    chunk_count = len(mesh_pieces)

    overhang = overhang_area / total_surface_area
    quality = quality / total_surface_area
    reindex = (
        PLANE_REINDEX_WEIGHT * plane_count / config.max_planes
        + CHUNK_REINDEX_WEIGHT * (chunk_count - 1) / config.max_planes
    )
    total = (
        OVERHANG_WEIGHT * overhang + QUALITY_WEIGHT * quality + REINDEX_WEIGHT * reindex
    )

    return _SearchState(
        planes=list(planes),
        total=total,
        chunk_count=chunk_count,
    )


class AutoPlaneSelector:
    def __init__(self, config: AutoPlaneConfig | None = None) -> None:
        """Initialize the selector with explicit or default search settings."""
        self.config = AutoPlaneConfig() if config is None else config

    def select(self, mesh: trimesh.Trimesh) -> list[AutoPlaneCandidate]:
        """Use beam search to find the lowest-scoring set of slice planes."""
        if self.config.max_planes <= 0:
            return []

        baseline = _evaluate_state(mesh, [], self.config)
        candidates = self.generate_candidates(mesh)
        ranked = sorted(
            self._expand_beam(mesh, [baseline], candidates),
            key=lambda state: state.total,
        )
        if not ranked:
            return baseline.planes

        beam = ranked[: self.config.beam_width]
        best = min(baseline, beam[0], key=lambda state: state.total)
        deep_candidates = [
            state.planes[0] for state in ranked[: self.config.deep_candidate_pool]
        ]

        for _ in range(1, self.config.max_planes):
            expanded = self._expand_beam(mesh, beam, deep_candidates)
            if not expanded:
                break

            beam = sorted(expanded, key=lambda state: state.total)[
                : self.config.beam_width
            ]
            best = min(best, beam[0], key=lambda state: state.total)

        return best.planes

    def _expand_beam(
        self,
        mesh: trimesh.Trimesh,
        beam: list[_SearchState],
        candidates: list[AutoPlaneCandidate],
    ) -> list[_SearchState]:
        """Extend each search state with every valid unused candidate plane."""
        expanded: list[_SearchState] = []

        for state in beam:
            used_keys = {plane.key() for plane in state.planes}

            for candidate in candidates:
                if candidate.key() in used_keys:
                    continue

                planes = [*state.planes, candidate]
                evaluation = _evaluate_state(mesh, planes, self.config)

                # Make sure actually creating new chunks
                if evaluation.chunk_count <= state.chunk_count:
                    continue

                expanded.append(evaluation)

        return expanded

    def generate_candidates(self, mesh: trimesh.Trimesh) -> list[AutoPlaneCandidate]:
        """Generate unique slice planes that intersect the mesh safely."""
        candidates: list[AutoPlaneCandidate] = []
        seen: set[tuple[float, ...]] = set()
        mesh_center = mesh.bounds.mean(axis=0)

        for normal in self.candidate_normals():
            projections = mesh.vertices @ normal
            for offset in _feature_offsets(
                projections,
                self.config.offsets_per_normal,
            ):
                candidate = AutoPlaneCandidate(
                    position=_plane_position(mesh_center, normal, float(offset)),
                    normal=normal,
                )
                key = candidate.key()
                if key in seen:
                    continue
                seen.add(key)
                if not _plane_intersection_has_bed_clearance(mesh, candidate):
                    continue
                if len(decompose_mesh(mesh, [candidate], False)) >= 2:
                    candidates.append(candidate)

        return candidates

    def candidate_normals(self) -> list[np.ndarray]:
        """Sample unique plane normals within the machine's tilt limit."""
        normals = [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ]

        if self.config.normal_samples > 1:
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            for index in range(self.config.normal_samples):
                z = index / (self.config.normal_samples - 1)
                radius = np.sqrt(max(0.0, 1.0 - z * z))
                angle = index * golden_angle
                normals.append(
                    np.array(
                        [
                            np.cos(angle) * radius,
                            np.sin(angle) * radius,
                            z,
                        ]
                    )
                )

        allowed = []
        seen = set()
        for normal in normals:
            normal = _normalize(normal)
            a_degrees, _ = Slicer.ab_angles(normal)
            if a_degrees > self.config.max_tilt_degrees:
                continue
            key = tuple(np.round(normal, 6))
            if key in seen:
                continue
            seen.add(key)
            allowed.append(normal)
        return allowed


def _classify_piece_faces(
    mesh: trimesh.Trimesh,
    print_direction: np.ndarray,
    floor_height: float,
    threshold_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify faces by print angle, bed contact, and overhang risk."""
    dots = np.clip(mesh.face_normals @ print_direction, -1.0, 1.0)
    face_heights = (mesh.vertices @ print_direction)[mesh.faces]
    floor_faces = np.all(
        np.isclose(face_heights, floor_height, atol=BED_CONTACT_TOLERANCE), axis=1
    )
    overhangs = (dots + np.sin(np.radians(threshold_degrees)) < 0.0) & ~floor_faces
    return dots, floor_faces, overhangs


def _plane_intersection_has_bed_clearance(
    mesh: trimesh.Trimesh,
    plane: AutoPlaneCandidate,
) -> bool:
    """Check that a plane intersects the mesh above the nozzle clearance."""
    lines = trimesh.intersections.mesh_plane(mesh, plane.normal, plane.position)
    return bool(len(lines) and lines[..., 2].min() >= BED_Z + NOZZLE_BED_CLEARANCE)


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of a non-zero vector."""
    vector = np.array(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("Expected a non-zero vector")
    return vector / norm


def _quaternion_from_z_to(normal: np.ndarray) -> np.ndarray:
    """Create a quaternion rotating the positive Z axis onto a normal."""
    normal = _normalize(normal)
    source = DEFAULT_PRINT_DIRECTION
    dot = float(np.clip(source @ normal, -1.0, 1.0))

    if np.isclose(dot, 1.0):
        # Identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    if np.isclose(dot, -1.0):
        # We assume a rotation around X
        return tf.quaternion_about_axis(np.pi, [1.0, 0.0, 0.0])

    axis = np.cross(source, normal)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return tf.quaternion_about_axis(angle, axis)


def _plane_position(
    mesh_center: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray:
    """Find a point on an offset plane nearest the mesh center."""
    normal = _normalize(normal)
    return mesh_center + normal * (offset - float(mesh_center @ normal))


def _feature_offsets(
    projections: np.ndarray,
    limit: int,
) -> list[float]:
    """Choose common interior vertex projections as candidate plane offsets."""
    rounded = np.round(projections.astype(float), decimals=4)
    values, counts = np.unique(rounded, return_counts=True)
    min_projection = float(values.min())
    max_projection = float(values.max())
    candidates = [
        (int(count), float(value))
        for value, count in zip(values, counts)
        if count >= 2
        and not np.isclose(float(value), min_projection)
        and not np.isclose(float(value), max_projection)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [offset for _, offset in candidates[:limit]]
