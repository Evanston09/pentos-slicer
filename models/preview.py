from dataclasses import dataclass

import numpy as np


@dataclass
class GcodePreviewPart:
    travel: np.ndarray
    extrusion: np.ndarray
    color: tuple[int, int, int]


@dataclass
class MachinePose:
    xyz: np.ndarray
    ab: np.ndarray


@dataclass
class GcodePreview:
    setup: np.ndarray
    parts: list[GcodePreviewPart]
    machine_poses: list[MachinePose]
