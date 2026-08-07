from types import SimpleNamespace

import numpy as np
import pytest

from gcode_tools import GcodeCommand
from services.multiplanar_gcode import apply_chunk_offsets


def test_apply_chunk_offsets_adjusts_flattened_absolute_moves() -> None:
    lines = [
        "G90\n",
        "G1 X10 Y20 Z0 F9000\n",
        "G1 X12 Y25 Z1 E0.5\n",
    ]
    chunk = SimpleNamespace(
        z_offset=5.0,
        flat_xy_offset=np.array([2.0, 3.0]),
    )

    transformed = apply_chunk_offsets(lines, chunk)

    first = GcodeCommand.parse(transformed[1])
    second = GcodeCommand.parse(transformed[2])
    assert first.args["X"] == pytest.approx(8.0)
    assert first.args["Y"] == pytest.approx(17.0)
    assert first.args["Z"] == pytest.approx(5.0)
    assert second.args["X"] == pytest.approx(10.0)
    assert second.args["Y"] == pytest.approx(22.0)
    assert second.args["Z"] == pytest.approx(6.0)
