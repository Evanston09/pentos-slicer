from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from gcode_tools import iter_gcode_moves
from models import DEFAULT_MACHINE_CONFIG
from services.nonplanar_gcode import _ab_angles, map_gcode_to_original
from services.volumetric_deformation import TetrahedralVolume


def test_ab_angles_hold_b_near_vertical_and_track_real_tilt() -> None:
    tilt = np.radians(20.0)
    normals = np.array(
        [
            [0.005, 0.005, 1.0],
            [-0.005, 0.005, 1.0],
            [0.0, -np.sin(tilt), np.cos(tilt)],
        ]
    )

    angles = []
    previous_b = 0.0
    for normal in normals:
        angle = _ab_angles(normal, previous_b, DEFAULT_MACHINE_CONFIG)
        angles.append(angle)
        previous_b = angle[1]
    angles = np.asarray(angles)

    assert_allclose(angles[:2, 1], 0.0, atol=1e-4)
    assert abs(angles[2, 1]) > 84.0
    assert np.all(np.abs(angles[:, 1]) <= 180.0)
    a = np.radians(angles[:, 0])
    b = np.radians(angles[:, 1])
    commanded = np.column_stack(
        (-np.sin(a) * np.cos(b), -np.sin(a) * np.sin(b), np.cos(a))
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    errors = np.degrees(
        np.arccos(np.clip(np.sum(commanded * normals, axis=1), -1.0, 1.0))
    )
    assert np.all(errors <= DEFAULT_MACHINE_CONFIG.max_normal_error_degrees + 0.01)


def test_map_gcode_inverse_maps_and_compensates_extrusion() -> None:
    original = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    volume = TetrahedralVolume(
        original_vertices=original,
        deformed_vertices=original * [2.0, 1.0, 1.0],
        tetrahedra=np.array([[0, 1, 2, 3]]),
        boundary_faces=np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
        scalar_values=original[:, 2],
    )
    offset = np.asarray(DEFAULT_MACHINE_CONFIG.machine_offset)
    start = offset + [0.0, 0.1, 0.1]
    end = offset + [0.4, 0.1, 0.1]
    text = (
        "ENABLE_FIVE_AXIS\n"
        "G90\n"
        "M83\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]} F600\n"
        ";LAYER_CHANGE\n"
        f"G1 X{end[0]} Y{end[1]} Z{end[2]} E0.9\n"
    )

    machine_config = replace(
        DEFAULT_MACHINE_CONFIG,
        a_max_velocity_deg_s=8.0,
        a_max_acceleration_deg_s2=40.0,
        b_max_velocity_deg_s=16.0,
        b_max_acceleration_deg_s2=80.0,
    )
    mapped = map_gcode_to_original(
        text,
        volume,
        machine_config,
        max_segment_length=0.15,
    )
    moves = list(iter_gcode_moves(mapped.splitlines()))

    assert (
        "MANUAL_STEPPER STEPPER=a_motor GCODE_AXIS=A LIMIT_VELOCITY=8 LIMIT_ACCEL=40\n"
    ) in mapped
    assert (
        "MANUAL_STEPPER STEPPER=b_motor GCODE_AXIS=B LIMIT_VELOCITY=16 LIMIT_ACCEL=80\n"
    ) in mapped
    assert len(moves) == 4
    assert_allclose(moves[-1].end_xyz, offset + [0.2, 0.1, 0.1])
    assert_allclose([move.extrusion_delta for move in moves[1:]], [0.15, 0.15, 0.15])
    assert_allclose([move.feedrate for move in moves[1:]], 300.0)
    assert_allclose(
        [[move.parsed.args["A"], move.parsed.args["B"]] for move in moves[1:]],
        0.0,
    )
