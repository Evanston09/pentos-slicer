from dataclasses import dataclass, field
from pathlib import Path
import io
import json
import zipfile
from typing import Self

import trimesh

from machine import BUILD_PLATE_CENTER
from plane_manager import PlaneSnapshot


@dataclass
class AppState:
    current_model: tuple[trimesh.Trimesh, str] | None = None
    model_xy_position: list[float] = field(
        default_factory=lambda: BUILD_PLATE_CENTER[:2],
    )
    model_z_degrees: float = 0.0
    plane_snapshots: list[PlaneSnapshot] = field(default_factory=list)
    gcode_path: Path | None = None
    debug_mode: bool = False

    @classmethod
    def from_bytes(cls, content: bytes) -> Self:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            model_bytes = zf.read("model.3mf")

        mesh = trimesh.load_mesh(io.BytesIO(model_bytes), file_type="3mf")
        assert isinstance(mesh, trimesh.Trimesh)

        return cls(
            current_model=(mesh, manifest["original_model_name"]),
            model_xy_position=list(manifest["model_xy_position"]),
            model_z_degrees=manifest["model_z_degrees"],
            plane_snapshots=[
                PlaneSnapshot.from_dict(snapshot)
                for snapshot in manifest["plane_snapshots"]
            ],
            gcode_path=None,
            debug_mode=manifest["debug_mode"],
        )

    def save(self) -> bytes:
        if self.current_model is None:
            raise ValueError("No model loaded")

        model, model_name = self.current_model
        model_bytes = model.export(file_type="3mf")
        assert isinstance(model_bytes, bytes | str)

        manifest = {
            "format": "pentos",
            "version": 1,
            "original_model_name": model_name,
            "model_xy_position": self.model_xy_position,
            "model_z_degrees": self.model_z_degrees,
            "plane_snapshots": [
                snapshot.as_dict() for snapshot in self.plane_snapshots
            ],
            "debug_mode": self.debug_mode,
        }

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("model.3mf", model_bytes)

        return zip_buffer.getvalue()
