from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MachinePose:
    xyz_mm: np.ndarray
    ab_degrees: np.ndarray


@dataclass(frozen=True)
class PreviewMove:
    pose: MachinePose
    preview_segment: np.ndarray | None
    kind: Literal["setup", "travel", "extrusion", "transition"]
    part_index: int | None


@dataclass
class GcodePreview:
    simulation_steps: list[PreviewMove]
