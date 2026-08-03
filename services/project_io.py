import io
import json
import zipfile

import trimesh

from models import AppState, PlaneSnapshot
from services.model_tools import source_name_from_filename, validate_mesh


def load_scene(content: bytes) -> AppState:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        model_bytes = zf.read("model.3mf")

    mesh = trimesh.load_mesh(io.BytesIO(model_bytes), file_type="3mf")
    validate_mesh(mesh)

    return AppState(
        current_model=(
            mesh,
            source_name_from_filename(manifest["original_model_name"]),
        ),
        model_xy_position=list(manifest["model_xy_position"]),
        model_z_degrees=manifest["model_z_degrees"],
        plane_snapshots=[
            PlaneSnapshot.from_dict(snapshot, plane_id)
            for plane_id, snapshot in enumerate(manifest["plane_snapshots"])
        ],
        debug_mode=manifest["debug_mode"],
    )


def save_scene(state: AppState) -> bytes:
    if state.current_model is None:
        raise ValueError("No model loaded")

    model, model_name = state.current_model
    model_bytes = model.export(file_type="3mf")
    assert isinstance(model_bytes, bytes | str)

    manifest = {
        "format": "pentos",
        "version": 1,
        "original_model_name": model_name,
        "model_xy_position": state.model_xy_position,
        "model_z_degrees": state.model_z_degrees,
        "plane_snapshots": [snapshot.as_dict() for snapshot in state.plane_snapshots],
        "debug_mode": state.debug_mode,
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("model.3mf", model_bytes)

    return zip_buffer.getvalue()
