from pathlib import Path
from typing import Protocol, Sequence


class GcodeChunk(Protocol):
    a_degrees: float
    b_degrees: float
    z_offset: float


def merge_gcode_files(
    gcode_paths: Sequence[Path],
    chunks: Sequence[GcodeChunk],
    output_path: Path,
) -> Path:
    if len(gcode_paths) != len(chunks):
        raise ValueError("G-code paths and chunks must have the same length")

    final_gcode = []
    total = len(gcode_paths)

    for index, gcode_path in enumerate(gcode_paths):
        lines = gcode_path.read_text().splitlines(keepends=True)
        lines = trim_gcode(lines, index, total)

        if index > 0:
            final_gcode.extend(
                transition(
                    chunks[index].a_degrees,
                    chunks[index].b_degrees,
                    chunks[index].z_offset,
                )
            )

        final_gcode.extend(lines)

    output_path.write_text("".join(final_gcode))
    return output_path


def trim_gcode(
    lines: list[str],
    index: int,
    total: int,
) -> list[str]:
    if total == 1:
        return lines
    if index == 0:
        return remove_end(lines)
    if index == total - 1:
        return remove_start(lines)
    return remove_start(remove_end(lines))


def remove_end(lines: list[str]) -> list[str]:
    custom_blocks = [
        i for i, line in enumerate(lines) if line.strip() == ";TYPE:Custom"
    ]
    return lines[: custom_blocks[-1]] if custom_blocks else lines


def remove_start(lines: list[str]) -> list[str]:
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == ";LAYER_CHANGE"),
        0,
    )
    return lines[start:]


def transition(a_degrees: float, b_degrees: float, z_offset: float) -> list[str]:
    return [
        "\n; --- PENTOS A/B TRANSITION ---\n",
        "G1 E-5 F3600\n",
        "G91 ; relative movement for safe lift\n",
        "G1 Z10 F3000\n",
        "G90 ; absolute movement\n",
        f"G1 A{a_degrees:.3f} B{b_degrees:.3f} F1200\n",
        "; --- PENTOS Z OFFSET ---\n",
        f"SET_GCODE_OFFSET Z={z_offset:.3f}\n",
        "G92 E0\n",
        "; --- END PENTOS A/B TRANSITION ---\n",
    ]
