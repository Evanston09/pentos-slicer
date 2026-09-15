from pathlib import Path

import numpy as np
from yourdfpy import URDF

from models import MachineConfig
from models.printer import PrinterModel

PENTOS_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "Pentos_URDF"
    / "urdf"
    / "Pentos_URDF.urdf"
)


def load_printer_model(machine_config: MachineConfig) -> PrinterModel:
    urdf = URDF.load(PENTOS_URDF_PATH, mesh_dir=str(PENTOS_URDF_PATH.parent))
    plate_m = urdf.get_transform("plate_center")[:3, 3]
    return PrinterModel(
        urdf=urdf,
        position_mm=np.asarray(machine_config.build_plate_center) - plate_m * 1000.0,
    )
