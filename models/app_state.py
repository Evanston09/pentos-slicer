from dataclasses import dataclass, field
from pathlib import Path

import trimesh

from .guide_surface import GuideSurfaceSnapshot
from .machine_config import MachineConfig
from .plane import PlaneSnapshot
from .slicing_settings import SlicingSettings


@dataclass
class AppState:
    slicing_settings: SlicingSettings = field(default_factory=SlicingSettings)
    machine_config: MachineConfig = field(default_factory=MachineConfig)
    current_model: tuple[trimesh.Trimesh, str] | None = None
    model_xy_position: tuple[float, float] = (45.0, 45.0)
    model_z_degrees: float = 0.0
    plane_snapshots: list[PlaneSnapshot] = field(default_factory=list)
    guide_surfaces: list[GuideSurfaceSnapshot] = field(default_factory=list)
    slicing_mode: str = "multiplanar"
    gcode_path: Path | None = None
    debug_mode: bool = False
