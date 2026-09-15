from dataclasses import dataclass

import numpy as np
from yourdfpy import URDF


@dataclass
class PrinterModel:
    urdf: URDF
    position_mm: np.ndarray
