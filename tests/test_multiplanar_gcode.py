from types import SimpleNamespace

import numpy as np
import pytest

from gcode_tools import GcodeCommand
from models import DEFAULT_MACHINE_CONFIG
from services.multiplanar_gcode import apply_chunk_offsets, merge_gcode_files


def test_merge_sums_chunk_time_estimates(tmp_path) -> None:
    paths = []
    for index, estimate in enumerate(("1m 10s", "50s")):
        path = tmp_path / f"chunk_{index}.gcode"
        path.write_text(
            "G90\n"
            ";LAYER_CHANGE\n"
            f"G1 X{index + 1} Y2 Z3 F1200\n"
            ";TYPE:Custom\n"
            f"; estimated printing time (normal mode) = {estimate}\n"
        )
        paths.append(path)
    chunks = [
        SimpleNamespace(
            z_offset=0.0,
            flat_xy_offset=[0.0, 0.0],
            a_degrees=0.0,
            b_degrees=0.0,
        )
        for _ in paths
    ]
    output = tmp_path / "merged.gcode"

    merge_gcode_files(paths, chunks, output, DEFAULT_MACHINE_CONFIG.machine_offset)

    assert output.read_text().startswith(
        "; estimated printing time (normal mode) = 2m\n"
    )


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
