import os
from pathlib import Path

import numpy as np
import trimesh
from trimesh import transformations as tf

from machine import BUILD_PLATE_CENTER, BUILD_VOLUME_SIZE
from models import AppState

SUPPORTED_UPLOAD_EXTENSIONS = frozenset({".stl", ".3mf", ".obj", ".ply", ".pentos"})
DEFAULT_MAX_UPLOAD_MB = 50
MAX_MESH_VERTICES = 1_000_000
MAX_MESH_FACES = 1_000_000


def max_upload_bytes() -> int:
    try:
        max_upload_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", str(DEFAULT_MAX_UPLOAD_MB)))
    except ValueError as exc:
        raise ValueError("MAX_UPLOAD_SIZE_MB must be an integer") from exc
    if max_upload_mb < 1:
        raise ValueError("MAX_UPLOAD_SIZE_MB must be at least 1")
    return max_upload_mb * 1024 * 1024


def validate_upload(name: str, content: bytes) -> tuple[str, str]:
    filename = Path(name).name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Choose one of: {supported}")
    if not content:
        raise ValueError("Uploaded file is empty")
    upload_limit = max_upload_bytes()
    if len(content) > upload_limit:
        raise ValueError(f"Upload exceeds the {upload_limit // (1024 * 1024)} MB limit")
    return filename, extension


# TODO: Improve
def source_name_from_filename(name: str) -> str:
    filename = Path(name.replace("\\", "/")).name
    stem = Path(filename).stem
    return stem if stem.strip(".") else "model"


def validate_mesh(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Uploaded file must contain a single mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Uploaded mesh is empty")
    if len(mesh.vertices) > MAX_MESH_VERTICES:
        raise ValueError(f"Mesh exceeds the {MAX_MESH_VERTICES:,} vertex limit")
    if len(mesh.faces) > MAX_MESH_FACES:
        raise ValueError(f"Mesh exceeds the {MAX_MESH_FACES:,} face limit")


def normalize_mesh_units(mesh: trimesh.Trimesh) -> None:
    if mesh.units is None or mesh.units.lower() == "none":
        mesh.units = "m" if mesh.extents.max() < 1.0 else "mm"

    if mesh.units != "mm":
        mesh.convert_units("mm")


def load_model(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path)
    validate_mesh(mesh)
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
    filename, extension = validate_upload(name, content)
    if extension == ".pentos":
        raise ValueError("Use scene loading for .pentos files")

    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename
    path.write_bytes(content)
    try:
        mesh = load_model(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return mesh, source_name_from_filename(filename)


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
