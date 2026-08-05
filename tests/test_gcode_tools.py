import numpy as np
import pytest
from numpy.testing import assert_allclose

import gcode_tools


def test_gcode_command_parse() -> None:
    parsed = gcode_tools.GcodeCommand.parse("g1 x1.5 Y-2 E.25 ; move")

    assert parsed.command == "G1"
    assert parsed.args == {"X": 1.5, "Y": -2.0, "E": 0.25}
    assert parsed.comment == "move"
    assert (
        parsed.build_with_updated_args({"x": 3.0, "e": 0.5}) == "G1 X3.0 Y-2 E0.5 ;move"
    )


def test_iter_gcode_moves_tracks_modes_and_extrusion() -> None:
    lines = [
        "G90\n",
        "M83\n",
        "G1 X1 Y2 Z3 E0.5 F1200\n",
        "G91\n",
        "G1 X1 Y-1 E0.2\n",
        "M82\n",
        "G92 E10\n",
        "G1 X1 E12\n",
    ]

    moves = list(gcode_tools.iter_gcode_moves(lines))

    assert len(moves) == 3
    assert moves[0].start_xyz is None
    assert moves[0].end_xyz is not None
    assert_allclose(moves[0].end_xyz, np.array([1.0, 2.0, 3.0]))
    assert moves[0].extrusion_delta == pytest.approx(0.5)
    assert moves[0].feedrate == pytest.approx(1200.0)

    assert moves[1].start_xyz is not None
    assert moves[1].end_xyz is not None
    assert_allclose(moves[1].start_xyz, np.array([1.0, 2.0, 3.0]))
    assert_allclose(moves[1].end_xyz, np.array([2.0, 1.0, 3.0]))
    assert moves[1].extrusion_delta == pytest.approx(0.2)
    assert moves[1].feedrate == pytest.approx(1200.0)
    assert not moves[1].is_absolute_xyz

    assert moves[2].start_xyz is not None
    assert moves[2].end_xyz is not None
    assert_allclose(moves[2].start_xyz, np.array([2.0, 1.0, 3.0]))
    assert_allclose(moves[2].end_xyz, np.array([3.0, 1.0, 3.0]))
    assert moves[2].extrusion_delta == pytest.approx(2.0)
    assert moves[2].feedrate == pytest.approx(1200.0)
    assert not moves[2].is_absolute_xyz


def test_iter_gcode_moves_tracks_start_and_end_ab() -> None:
    moves = list(
        gcode_tools.iter_gcode_moves(
            [
                "G90\n",
                "G1 X1 Y2 Z3 A10 B20\n",
                "G1 X2 A15 B25\n",
            ]
        )
    )

    assert_allclose(moves[0].start_ab, [0.0, 0.0])
    assert_allclose(moves[0].end_ab, [10.0, 20.0])
    assert_allclose(moves[1].start_ab, [10.0, 20.0])
    assert_allclose(moves[1].end_ab, [15.0, 25.0])


def test_translate_gcode_shifts_only_absolute_xyz_moves() -> None:
    lines = [
        "G90\n",
        "G1 X1 Y2 Z3 E0.1 ; move\n",
        "G91\n",
        "G1 X1 Y2 Z3\n",
    ]

    translated = gcode_tools.translate_gcode(lines, np.array([10.0, 20.0, 30.0]))

    absolute_move = gcode_tools.GcodeCommand.parse(translated[1])
    assert absolute_move.args["X"] == pytest.approx(11.0)
    assert absolute_move.args["Y"] == pytest.approx(22.0)
    assert absolute_move.args["Z"] == pytest.approx(33.0)
    assert translated[3] == "G1 X1 Y2 Z3\n"


def test_remove_leading_retract_keeps_prime_and_print_retracts() -> None:
    lines = [
        ";LAYER_CHANGE\n",
        ";Z:0.2\n",
        "G1 E-5 F3600\n",
        "G1 Z.2 F9000\n",
        "G1 X10 Y20\n",
        "G1 E5.25 F2400\n",
        ";TYPE:Perimeter\n",
        "G1 X20 Y20 E.5\n",
        "G1 E-3.5 F3600\n",
    ]

    cleaned = gcode_tools.remove_leading_retract(lines)

    assert "G1 E-5 F3600\n" not in cleaned
    assert "G1 E5.25 F2400\n" in cleaned
    assert "G1 X20 Y20 E.5\n" in cleaned
    assert "G1 E-3.5 F3600\n" in cleaned
