import numpy as np
from numpy.testing import assert_allclose

from machine import rotation_matrix
from models import DEFAULT_MACHINE_CONFIG
from services.gcode_preview import parse_gcode_preview


def test_simulation_steps_keep_move_order_and_classification() -> None:
    preview = parse_gcode_preview(
        "G90\n"
        "G1 Z5\n"
        "G1 X68 Y7\n"
        ";LAYER_CHANGE\n"
        "G1 X69 Y7 E0.1\n"
        "; --- PENTOS A/B TRANSITION ---\n"
        "G1 A10 B20\n"
        "; --- END PENTOS A/B TRANSITION ---\n"
        "G1 X70 Y7\n",
        DEFAULT_MACHINE_CONFIG,
    )

    assert [step.kind for step in preview.simulation_steps] == [
        "setup",
        "extrusion",
        "transition",
        "travel",
    ]
    assert preview.simulation_steps[0].preview_segment is None
    assert preview.simulation_steps[1].part_index == 0
    assert preview.simulation_steps[3].part_index == 1
    assert_allclose(preview.simulation_steps[2].pose.ab_degrees, [10.0, 20.0])


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

    extrusion = next(
        step.preview_segment
        for step in preview.simulation_steps
        if step.kind == "extrusion"
    )
    assert_allclose(extrusion, [start, end])


def test_preview_tracks_ab_over_nominal_motion_time() -> None:
    preview = parse_gcode_preview(
        "G90\nG1 X0 Y0 Z0 F600\nG1 X10 Y0 Z0 A10 B20\nG1 X20 Y0 Z0\n",
        DEFAULT_MACHINE_CONFIG,
    )

    assert_allclose(preview.motion_time_seconds, [0.0, 1.0, 2.0])
    assert_allclose(preview.a_degrees, [0.0, 10.0, 10.0])
    assert_allclose(preview.b_degrees, [0.0, 20.0, 20.0])


def test_preview_omits_ab_trace_without_rotary_commands() -> None:
    preview = parse_gcode_preview(
        "G90\nG1 X0 Y0 Z0 F600\nG1 X10 Y0 Z0\n",
        DEFAULT_MACHINE_CONFIG,
    )

    assert len(preview.motion_time_seconds) == 0
