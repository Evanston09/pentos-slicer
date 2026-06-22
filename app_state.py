from dataclasses import dataclass, field
from pathlib import Path

import trimesh

from plane_manager import PlaneSnapshot


@dataclass
class AppState:
    current_model: tuple[trimesh.Trimesh, str] | None = None
    plane_snapshots: list[PlaneSnapshot] = field(default_factory=list)
    gcode_path: Path | None = None
    debug_mode: bool = False
