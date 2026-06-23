from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gcode_tools


@dataclass
class FakeChunk:
    a_degrees: float = 0.0
    b_degrees: float = 0.0
    z_offset: float = 0.0
    flat_xy_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))


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
        "G1 X1 Y2 Z3 E0.5\n",
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

    assert moves[1].start_xyz is not None
    assert moves[1].end_xyz is not None
    assert_allclose(moves[1].start_xyz, np.array([1.0, 2.0, 3.0]))
    assert_allclose(moves[1].end_xyz, np.array([2.0, 1.0, 3.0]))
    assert moves[1].extrusion_delta == pytest.approx(0.2)
    assert not moves[1].is_absolute_xyz

    assert moves[2].start_xyz is not None
    assert moves[2].end_xyz is not None
    assert_allclose(moves[2].start_xyz, np.array([2.0, 1.0, 3.0]))
    assert_allclose(moves[2].end_xyz, np.array([3.0, 1.0, 3.0]))
    assert moves[2].extrusion_delta == pytest.approx(2.0)
    assert not moves[2].is_absolute_xyz


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


def test_apply_chunk_offsets_adjusts_flattened_absolute_moves() -> None:
    lines = [
        "G90\n",
        "G1 X10 Y20 Z0 F9000\n",
        "G1 X12 Y25 Z1 E0.5\n",
    ]
    chunk = FakeChunk(
        z_offset=5.0,
        flat_xy_offset=np.array([2.0, 3.0]),
    )

    transformed = gcode_tools.apply_chunk_offsets(lines, chunk)

    first = gcode_tools.GcodeCommand.parse(transformed[1])
    second = gcode_tools.GcodeCommand.parse(transformed[2])
    assert first.args["X"] == pytest.approx(8.0)
    assert first.args["Y"] == pytest.approx(17.0)
    assert first.args["Z"] == pytest.approx(5.0)
    assert second.args["X"] == pytest.approx(10.0)
    assert second.args["Y"] == pytest.approx(22.0)
    assert second.args["Z"] == pytest.approx(6.0)
