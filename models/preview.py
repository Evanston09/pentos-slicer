from dataclasses import dataclass

import numpy as np


@dataclass
class GcodePreviewPart:
    travel: np.ndarray
    extrusion: np.ndarray
    color: tuple[int, int, int]


@dataclass
class GcodePreview:
    setup: np.ndarray
    parts: list[GcodePreviewPart]
    motion_time_seconds: np.ndarray
    a_degrees: np.ndarray
    b_degrees: np.ndarray
