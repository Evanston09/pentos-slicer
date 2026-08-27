import numpy as np
from numpy.testing import assert_allclose

from gcode_tools import iter_gcode_moves
from models import DEFAULT_MACHINE_CONFIG
from services.nonplanar_gcode import (
    _ab_angles,
    _commanded_normal,
    map_gcode_to_original,
)
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
    previous_angles = (0.0, 0.0)
    for normal in normals:
        previous_angles = _ab_angles(
            normal,
            previous_angles,
            DEFAULT_MACHINE_CONFIG,
        )
        angles.append(previous_angles)
    angles = np.asarray(angles)

    assert_allclose(angles[:2, 1], 0.0, atol=1e-4)
    assert abs(angles[2, 1]) > 84.0
    assert np.all(np.abs(angles[:, 1]) <= 180.0)
    commanded = np.asarray([_commanded_normal(tuple(angle)) for angle in angles])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    errors = np.degrees(
        np.arccos(np.clip(np.sum(commanded * normals, axis=1), -1.0, 1.0))
    )
    assert np.all(errors <= DEFAULT_MACHINE_CONFIG.max_normal_error_degrees + 0.01)


def test_ab_angles_hold_pose_and_move_only_as_far_as_needed() -> None:
    target_angles = [(0.0, 0.0), (0.6, 0.0), (1.2, 0.0), (2.0, 0.0)]
    previous_angles = (0.0, 0.0)
    commanded_angles = []

    for target in target_angles:
        previous_angles = _ab_angles(
            _commanded_normal(target),
            previous_angles,
            DEFAULT_MACHINE_CONFIG,
        )
        commanded_angles.append(previous_angles)

    assert_allclose(
        commanded_angles,
        [(0.0, 0.0), (0.0, 0.0), (0.2, 0.0), (1.0, 0.0)],
        atol=1e-6,
    )


def test_ab_angles_remove_small_oscillations() -> None:
    previous_angles = (0.0, 0.0)
    commanded_angles = []

    for target_a in [0.6, -0.6, 0.4, -0.4]:
        previous_angles = _ab_angles(
            _commanded_normal((target_a, 0.0)),
            previous_angles,
            DEFAULT_MACHINE_CONFIG,
        )
        commanded_angles.append(previous_angles)

    assert_allclose(commanded_angles, 0.0, atol=1e-8)


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
        f"G1 X{start[0]} Y{start[1]} Z{start[2]}\n"
        ";LAYER_CHANGE\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]}\n"
        ";LAYER_CHANGE\n"
        f"G1 X{end[0]} Y{end[1]} Z{end[2]} E0.9\n"
    )

    mapped = map_gcode_to_original(
        text,
        volume,
        DEFAULT_MACHINE_CONFIG,
        max_segment_length=0.15,
    )
    moves = list(iter_gcode_moves(mapped.splitlines()))

    assert len(moves) == 6
    assert all("A" not in move.parsed.args for move in moves[1:3])
    assert_allclose(moves[-1].end_xyz, offset + [0.35, 0.1, 0.1])
    assert_allclose(
        [move.extrusion_delta for move in moves[-3:]],
        [0.2625, 0.2625, 0.2625],
    )
    assert_allclose([move.feedrate for move in moves[-3:]], 525.0)
    assert_allclose(
        [[move.parsed.args["A"], move.parsed.args["B"]] for move in moves[-3:]],
        0.0,
    )
