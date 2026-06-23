from dataclasses import dataclass, field
from pathlib import Path

import trimesh

from plane_manager import PlaneSnapshot


@dataclass
class AppState:
    current_model: tuple[trimesh.Trimesh, str] | None = None
    model_xy_offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    model_z_degrees: float = 0.0
    plane_snapshots: list[PlaneSnapshot] = field(default_factory=list)
    gcode_path: Path | None = None
    debug_mode: bool = False
