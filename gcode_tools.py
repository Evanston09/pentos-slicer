from pathlib import Path
from typing import Callable, Protocol, Sequence
from dataclasses import dataclass


class GcodeChunk(Protocol):
    a_degrees: float
    b_degrees: float
    z_offset: float


@dataclass
class GcodeCommand:
    command: str | None
    args: list[str] | None
    comment: str | None


def parse_gcode_line(line: str) -> GcodeCommand:
    code, separator, raw_comment = line.strip().partition(";")
    tokens = code.split()
    comment = raw_comment.strip() if separator else None

    return GcodeCommand(
        command=tokens[0] if tokens else None,
        args=tokens[1:] or None,
        comment=comment or None,
    )


# See if necessary when we introduce time mashing
def is_comment_line(line: str, matches: Callable[[str | None], bool]) -> bool:
    parsed = parse_gcode_line(line)
    return parsed.command is None and matches(parsed.comment)


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
            initial_xy = find_initial_xy(lines)
            final_gcode.extend(
                transition(
                    initial_xy,
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
        i
        for i, line in enumerate(lines)
        if is_comment_line(line, lambda comment: comment == "TYPE:Custom")
    ]
    return lines[: custom_blocks[-1]] if custom_blocks else lines


def remove_start(lines: list[str]) -> list[str]:
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if is_comment_line(line, lambda comment: comment == "LAYER_CHANGE")
        ),
        0,
    )
    return lines[start:]


def find_initial_xy(lines: list[str]) -> tuple[float, float]:
    for line in lines:
        parsed = parse_gcode_line(line)
        if parsed.command != "G1" or parsed.args is None:
            continue

        x = None
        y = None

        for arg in parsed.args:
            if arg.startswith("X"):
                x = float(arg[1:])
            elif arg.startswith("Y"):
                y = float(arg[1:])

        if x is not None and y is not None:
            return x, y

    raise ValueError("Could not find initial G1 move with X and Y coordinates")


def transition(
    initial_xy: tuple[float, float], a_degrees: float, b_degrees: float, z_offset: float
) -> list[str]:
    return [
        "\n; --- PENTOS A/B TRANSITION ---\n",
        "G1 E-5 F3600\n",
        "G91 ; relative movement for safe lift\n",
        "G1 Z10 F3000\n",
        "G90 ; absolute movement\n",
        f"G1 A{a_degrees} B{b_degrees} F1200\n",
        "; --- PENTOS MOVE TO SAFE XY---\n",
        f"G1 X{initial_xy[0]} Y{initial_xy[1]} F1200\n",
        "; --- PENTOS Z OFFSET ---\n",
        f"SET_GCODE_OFFSET Z={z_offset}\n",
        "G92 E0\n",
        "; --- END PENTOS A/B TRANSITION ---\n",
    ]
