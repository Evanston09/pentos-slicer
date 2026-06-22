from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Self, Sequence

import numpy as np

from machine import BUILD_PLATE_CENTER, rotation_matrix


class GcodeChunk(Protocol):
    a_degrees: float
    b_degrees: float
    z_offset: float


@dataclass
class GcodeCommand:
    command: str = ""
    args: list[str] = field(default_factory=list)
    comment: str | None = None

    @classmethod
    def parse(cls, line: str) -> Self:
        code, separator, raw_comment = line.strip().partition(";")
        tokens = code.split()
        return cls(
            command=tokens[0] if tokens else "",
            args=tokens[1:],
            comment=(raw_comment.strip() or None) if separator else None,
        )

    def build(self) -> str:
        code = " ".join([self.command, *self.args]) if self.command else ""
        if self.comment is None:
            return code
        if code:
            return f"{code} ;{self.comment}"
        return f";{self.comment}"


@dataclass
class GcodeMotionBounds:
    first_position: tuple[float, float, float]
    final_position: tuple[float, float, float]


# See if necessary when we introduce time mashing
def is_comment_line(line: str, matches: Callable[[str | None], bool]) -> bool:
    parsed = GcodeCommand.parse(line)
    return not parsed.command and matches(parsed.comment)


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
            lines = transform_chunk_gcode(lines, chunks[index])
            initial_xyz = find_first_last_xyz(lines).first_position
            final_gcode.extend(
                transition(
                    initial_xyz,
                    chunks[index].a_degrees,
                    chunks[index].b_degrees,
                )
            )

        final_gcode.extend(lines)

    output_path.write_text("".join(final_gcode))
    return output_path


def generate_debug_transition_check(
    gcode_paths: Sequence[Path],
    chunks: Sequence[GcodeChunk],
    output_path: Path,
) -> Path:
    if len(gcode_paths) != len(chunks):
        raise ValueError("G-code paths and chunks must have the same length")

    total = len(gcode_paths)
    bounds = []

    for index, gcode_path in enumerate(gcode_paths):
        lines = gcode_path.read_text().splitlines(keepends=True)
        lines = trim_gcode(lines, index, total)
        if index > 0:
            lines = transform_chunk_gcode(lines, chunks[index])
        bounds.append(find_first_last_xyz(lines))

    debug_gcode = [
        "; --- PENTOS DEBUG TRANSITION CHECK ---\n",
        "G90 ; absolute movement\n",
        "M83 ; extruder relative mode\n",
        "G28 ; home all axis\n",
        "HOME_A\n",
        "HOME_B\n",
        "ENABLE_FIVE_AXIS\n",
        "G1 Z50 F240\n",
    ]

    if total < 2:
        debug_gcode.append("; No transitions to check.\n")

    for index in range(1, total):
        next_chunk = chunks[index]
        prev_bound = bounds[index - 1]
        next_bound = bounds[index]
        prev_x, prev_y, prev_z = prev_bound.final_position

        debug_gcode.extend(
            [
                f"\n; --- DEBUG STAGE CHUNK {index - 1} END STATE ---\n",
                "G90 ; absolute movement\n",
                f"G1 X{prev_x} Y{prev_y} Z{prev_z} F3000\n",
                "; Prev chunk final commanded Z restored before relative lift.\n",
                f"; --- DEBUG TRANSITION {index - 1} TO {index} ---\n",
            ]
        )
        debug_gcode.extend(
            transition(
                next_bound.first_position,
                next_chunk.a_degrees,
                next_chunk.b_degrees,
            )
        )
        debug_gcode.append(f"; --- END DEBUG TRANSITION {index - 1} TO {index} ---\n")

    debug_gcode.extend(
        [
            "\n; --- END PENTOS DEBUG TRANSITION CHECK ---\n",
            "G90 ; absolute movement\n",
            "G1 Z50 F1200\n",
            "G1 X5 Y76.5 F9000 ; present print\n",
            "DISABLE_FIVE_AXIS\n",
            "M107 ; turn off fan\n",
            "M84 X Y E ; disable motors\n",
        ]
    )

    output_path.write_text("".join(debug_gcode))
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


def transform_chunk_gcode(lines: list[str], chunk: GcodeChunk) -> list[str]:
    if not lines:
        return []

    transformed = []
    initial_x, initial_y, initial_z = find_first_last_xyz(lines).first_position
    flat_position: dict[str, float] = {
        "X": initial_x,
        "Y": initial_y,
        "Z": initial_z,
    }
    absolute_xyz = True
    r = rotation_matrix(chunk.a_degrees, chunk.b_degrees)

    for line in lines:
        parsed = GcodeCommand.parse(line)
        command = parsed.command.upper()

        if command == "G90":
            absolute_xyz = True
        elif command == "G91":
            absolute_xyz = False
        elif command in {"G0", "G1"} and absolute_xyz:
            moved_xyz = False
            for arg in parsed.args:
                key = arg[:1].upper()
                if key in flat_position:
                    flat_position[key] = float(arg[1:])
                    moved_xyz = True

            if moved_xyz:
                flat_point = np.array(
                    [
                        flat_position["X"],
                        flat_position["Y"],
                        flat_position["Z"] + chunk.z_offset,
                    ]
                )
                real_point = BUILD_PLATE_CENTER + r @ (flat_point - BUILD_PLATE_CENTER)
                # Preserve the original line ending while rebuilding the command.
                stripped = line.rstrip("\r\n")
                ending = line[len(stripped) :]
                kept_args = [
                    arg for arg in parsed.args if arg[:1].upper() not in {"X", "Y", "Z"}
                ]
                parsed.args = [
                    f"X{real_point[0]}",
                    f"Y{real_point[1]}",
                    f"Z{real_point[2]}",
                    *kept_args,
                ]
                line = parsed.build() + ending

        transformed.append(line)

    return transformed


# TODO: Consider relative motion
def find_first_last_xyz(lines: list[str]) -> GcodeMotionBounds:
    pos: dict[str, float | None] = {"X": None, "Y": None, "Z": None}
    first_position: tuple[float, float, float] | None = None
    final_position: tuple[float, float, float] | None = None

    for line in lines:
        parsed = GcodeCommand.parse(line)
        if parsed.command not in {"G0", "G1"}:
            continue
        for arg in parsed.args:
            key = arg[0].upper()
            if key in pos:
                pos[key] = float(arg[1:])
        x, y, z = pos["X"], pos["Y"], pos["Z"]
        if x is None or y is None or z is None:
            continue
        if first_position is None:
            first_position = (x, y, z)
        final_position = (x, y, z)

    assert first_position is not None and final_position is not None

    return GcodeMotionBounds(
        first_position=first_position,
        final_position=final_position,
    )


def transition(
    initial_xyz: tuple[float, float, float], a_degrees: float, b_degrees: float
) -> list[str]:
    return [
        "\n; --- PENTOS A/B TRANSITION ---\n",
        "G1 E-5 F3600\n",
        "G91 ; relative movement for safe lift\n",
        "G1 Z10 F3000\n",
        "G90 ; absolute movement\n",
        f"G1 A{a_degrees} B{b_degrees} F1200\n",
        "; --- PENTOS MOVE TO NEXT CHUNK ---\n",
        f"G1 X{initial_xyz[0]} Y{initial_xyz[1]} Z{initial_xyz[2]} F1200\n",
        "G92 E0\n",
        "; --- END PENTOS A/B TRANSITION ---\n",
    ]
