import numpy as np
from numpy.testing import assert_allclose

from machine import rotation_matrix
from models import DEFAULT_MACHINE_CONFIG
from services.gcode_preview import parse_gcode_preview


def test_preview_uses_each_endpoint_ab_pose() -> None:
    start = np.array([45.0, 45.0, 5.0])
    end = np.array([46.0, 45.0, 5.0])
    center = np.asarray(DEFAULT_MACHINE_CONFIG.rotation_center_local_mm)
    machine_start = np.asarray(DEFAULT_MACHINE_CONFIG.machine_offset) + start
    machine_end = (
        np.asarray(DEFAULT_MACHINE_CONFIG.machine_offset)
        + center
        + rotation_matrix(10.0, 0.0) @ (end - center)
    )
    text = (
        "G90\n"
        "M83\n"
        f"G1 X{machine_start[0]} Y{machine_start[1]} Z{machine_start[2]}\n"
        ";LAYER_CHANGE\n"
        f"G1 X{machine_end[0]} Y{machine_end[1]} Z{machine_end[2]} "
        "A10 B0 E0.1\n"
    )

    preview = parse_gcode_preview(text, DEFAULT_MACHINE_CONFIG)

    assert_allclose(preview.parts[0].extrusion[0], [start, end])
