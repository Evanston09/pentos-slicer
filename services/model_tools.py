from pathlib import Path

import numpy as np
import trimesh
from trimesh import transformations as tf

from machine import BUILD_PLATE_CENTER, BUILD_VOLUME_SIZE
from models import AppState


def normalize_mesh_units(mesh: trimesh.Trimesh) -> None:
    if mesh.units is None or mesh.units.lower() == "none":
        mesh.units = "m" if mesh.extents.max() < 1.0 else "mm"

    if mesh.units != "mm":
        mesh.convert_units("mm")


def load_model(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path)
    normalize_mesh_units(mesh)

    lower, upper = mesh.bounds
    mesh_center_xy = (lower[:2] + upper[:2]) / 2.0
    mesh.apply_translation(
        [
            BUILD_PLATE_CENTER[0] - mesh_center_xy[0],
            BUILD_PLATE_CENTER[1] - mesh_center_xy[1],
            -lower[2],
        ]
    )
    return mesh


def load_uploaded_model(
    name: str,
    content: bytes,
    upload_dir: Path,
) -> tuple[trimesh.Trimesh, str]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / Path(name).name
    path.write_bytes(content)
    return load_model(path), path.stem


def model_center(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.bounds.mean(axis=0)


def model_wxyz(state: AppState) -> np.ndarray:
    return tf.quaternion_from_euler(
        0.0,
        0.0,
        np.radians(state.model_z_degrees),
        axes="sxyz",
    )


def model_frame_position(state: AppState, mesh: trimesh.Trimesh) -> np.ndarray:
    center = model_center(mesh)
    xy_position = state.model_xy_position
    return np.array([xy_position[0], xy_position[1], center[2]])


def transformed_model(state: AppState) -> tuple[trimesh.Trimesh, str] | None:
    if state.current_model is None:
        return None

    base_mesh, source_name = state.current_model
    mesh = base_mesh.copy()
    if not np.isclose(state.model_z_degrees, 0.0):
        mesh.apply_transform(
            tf.rotation_matrix(
                np.radians(state.model_z_degrees),
                [0.0, 0.0, 1.0],
                point=model_center(base_mesh),
            ),
        )

    xy_position = state.model_xy_position
    center = model_center(base_mesh)
    mesh.apply_translation(
        [
            xy_position[0] - center[0],
            xy_position[1] - center[1],
            0.0,
        ]
    )
    return mesh, source_name


def model_within_build_volume(mesh: trimesh.Trimesh) -> bool:
    lower, upper = mesh.bounds
    volume_upper = np.asarray(BUILD_VOLUME_SIZE)
    lower_inside = np.logical_or(lower >= 0.0, np.isclose(lower, 0.0))
    upper_inside = np.logical_or(
        upper <= volume_upper,
        np.isclose(upper, volume_upper),
    )
    return bool(np.all(lower_inside) and np.all(upper_inside))
