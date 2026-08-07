import numpy as np
from numpy.testing import assert_allclose

from gcode_tools import iter_gcode_moves
from machine import MACHINE_OFFSET
from services.nonplanar_gcode import continuous_ab_angles, map_gcode_to_original
from services.volumetric_deformation import TetrahedralVolume


def test_continuous_ab_angles_avoids_half_turn_at_zero_tilt() -> None:
    previous = (0.0, 0.0)
    angles = []
    for normal in (
        np.array([-0.2, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.2, 0.0, 1.0]),
    ):
        previous = continuous_ab_angles(normal, previous)
        angles.append(previous)

    assert angles[0][0] > 0.0
    assert angles[1][0] == 0.0
    assert angles[2][0] < 0.0
    assert_allclose(np.asarray(angles)[:, 1], 0.0)


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
    offset = np.asarray(MACHINE_OFFSET)
    start = offset + [0.0, 0.1, 0.1]
    end = offset + [0.4, 0.1, 0.1]
    text = (
        "G90\n"
        "M83\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]} F600\n"
        ";LAYER_CHANGE\n"
        f"G1 X{end[0]} Y{end[1]} Z{end[2]} E0.9\n"
    )

    mapped = map_gcode_to_original(
        text,
        volume,
        max_segment_length=0.15,
    )
    moves = list(iter_gcode_moves(mapped.splitlines()))

    assert len(moves) == 4
    assert_allclose(moves[-1].end_xyz, offset + [0.2, 0.1, 0.1])
    assert_allclose([move.extrusion_delta for move in moves[1:]], [0.15, 0.15, 0.15])
    assert_allclose([move.feedrate for move in moves[1:]], 300.0)
    assert_allclose(
        [[move.parsed.args["A"], move.parsed.args["B"]] for move in moves[1:]],
        0.0,
    )
