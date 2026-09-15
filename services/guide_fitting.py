import numpy as np
import trimesh

from models import GuideSurfaceSnapshot
from services.auto_planes import quaternion_from_z_to

# Fitting accuracy is independent of the display mesh's resolution.
GUIDE_FIT_SAMPLES_PER_AXIS = 30


def align_guide_to_surface(
    mesh: trimesh.Trimesh,
    guide: GuideSurfaceSnapshot,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
) -> GuideSurfaceSnapshot | None:
    """Return a fitted candidate, None on a ray miss, or raise on an invalid fit."""
    locations, _, face_indices = mesh.ray.intersects_location(
        [ray_origin], [ray_direction]
    )
    if len(locations) == 0:
        return None

    # Get the closer hit
    hit_index = int(np.linalg.norm(locations - ray_origin, axis=1).argmin())
    face_index = int(face_indices[hit_index])
    surface_faces = _connected_surface_faces(mesh, face_index)
    surface = mesh.submesh([surface_faces], append=True)
    surface_center = np.average(
        mesh.triangles_center[surface_faces],
        axis=0,
        weights=mesh.area_faces[surface_faces],
    )
    normal = np.average(
        mesh.face_normals[surface_faces],
        axis=0,
        weights=mesh.area_faces[surface_faces],
    )
    normal_length = np.linalg.norm(normal)
    # fallback to face normal if averaged normal is too small
    normal = (
        mesh.face_normals[face_index]
        if np.isclose(normal_length, 0.0)
        else normal / normal_length
    )
    # Snap the estimated patch center to the nearest point on the selected surface.
    center_hits, _, _ = surface.ray.intersects_location(
        [surface_center + normal * max(float(mesh.scale), 1.0)],
        [-normal],
    )
    if len(center_hits) == 0:
        raise ValueError("The selected surface has no center intersection")
    surface_center = center_hits[
        np.linalg.norm(center_hits - surface_center, axis=1).argmin()
    ]
    candidate = GuideSurfaceSnapshot(
        position=surface_center,
        wxyz=quaternion_from_z_to(normal),
        guide_id=guide.guide_id,
        size_mm=guide.size_mm.copy(),
    )
    candidate.heights_mm = _fit_surface_heights(surface, candidate)
    # Keep the snapped origin on the selected surface.
    center_height = float(candidate.evaluate_local(np.zeros((1, 2)))[0][0])
    candidate.heights_mm -= center_height
    return candidate


def _connected_surface_faces(mesh: trimesh.Trimesh, face_index: int) -> np.ndarray:
    normal_threshold = np.cos(np.radians(45.0))
    neighbors: list[list[int]] = [[] for _ in range(len(mesh.faces))]
    for first, second in mesh.face_adjacency:
        neighbors[int(first)].append(int(second))
        neighbors[int(second)].append(int(first))

    def collect(*, follow_curvature: bool) -> set[int]:
        surface = {face_index}
        pending = [face_index]
        while pending:
            current = pending.pop()
            reference = current if follow_curvature else face_index
            for candidate in neighbors[current]:
                if candidate in surface:
                    continue
                if (
                    mesh.face_normals[candidate] @ mesh.face_normals[reference]
                    < normal_threshold
                ):
                    continue
                surface.add(candidate)
                pending.append(candidate)
        return surface

    # collect surface faces, first following curvature, then falling back to face normals
    surface = collect(follow_curvature=True)
    if len(surface) == len(mesh.faces):
        surface = collect(follow_curvature=False)
    return np.fromiter(surface, dtype=np.int64)


def _fit_surface_heights(
    surface: trimesh.Trimesh, guide: GuideSurfaceSnapshot
) -> np.ndarray:
    """Fit the guide's 4x4 control grid to sampled mesh-surface heights."""
    rotation = guide.rotation
    normal = rotation[:, 2]
    local_surface = (surface.vertices - guide.position) @ rotation
    half = guide.size_mm / 2.0
    lower = np.maximum(-half, local_surface[:, :2].min(axis=0))
    upper = np.minimum(half, local_surface[:, :2].max(axis=0))
    if np.any(upper <= lower):
        raise ValueError("The guide patch does not overlap the selected surface")
    x = np.linspace(lower[0], upper[0], GUIDE_FIT_SAMPLES_PER_AXIS)
    y = np.linspace(lower[1], upper[1], GUIDE_FIT_SAMPLES_PER_AXIS)
    grid_x, grid_y = np.meshgrid(x, y)
    local_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    ray_height = float(local_surface[:, 2].max()) + 0.01
    ray_origins = guide.position + local_xy @ rotation[:, :2].T + normal * ray_height
    locations, ray_indices, _ = surface.ray.intersects_location(
        ray_origins, np.tile(-normal, (len(ray_origins), 1))
    )
    heights = np.full(len(local_xy), np.inf)
    for ray_index, location in zip(ray_indices, locations):
        candidate = float((location - guide.position) @ normal)
        if abs(candidate) < abs(heights[ray_index]):
            heights[ray_index] = candidate
    valid = np.isfinite(heights)
    return guide.fit_heights(local_xy[valid], heights[valid])
