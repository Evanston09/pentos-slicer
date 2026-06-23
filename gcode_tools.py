from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol, Self, Sequence

import numpy as np

from machine import MACHINE_OFFSET


class GcodeChunk(Protocol):
    a_degrees: float
    b_degrees: float
    z_offset: float
    flat_xy_offset: np.ndarray


@dataclass
class GcodeCommand:
    command: str = ""
    raw_args: list[str] = field(default_factory=list)
    args: dict[str, float] = field(default_factory=dict)
    comment: str | None = None

    @classmethod
    def parse(cls, line: str) -> Self:
        code, separator, raw_comment = line.strip().partition(";")
        tokens = code.split()
        raw_args = tokens[1:]
        return cls(
            command=tokens[0].upper() if tokens else "",
            raw_args=raw_args,
            args=parse_gcode_args(raw_args),
            comment=(raw_comment.strip() or None) if separator else None,
        )

    def build(self) -> str:
        code = " ".join([self.command, *self.raw_args]) if self.command else ""
        if self.comment is None:
            return code
        if code:
            return f"{code} ;{self.comment}"
        return f";{self.comment}"

    def build_with_updated_args(self, values: dict[str, float]) -> str:
        update_values = {key.upper(): value for key, value in values.items()}
        next_args = []

        for arg in self.raw_args:
            parsed_arg = parse_gcode_arg(arg)
            if parsed_arg is None:
                next_args.append(arg)
                continue

            key, _ = parsed_arg
            if key in update_values:
                next_args.append(f"{key}{update_values[key]}")
            else:
                next_args.append(arg)

        updated = GcodeCommand(
            command=self.command,
            raw_args=next_args,
            args=parse_gcode_args(next_args),
            comment=self.comment,
        )
        return updated.build()


def parse_gcode_arg(arg: str) -> tuple[str, float] | None:
    token = arg.strip()
    if len(token) < 2:
        return None

    key = token[0].upper()
    if not key.isalpha():
        return None

    try:
        return key, float(token[1:])
    except ValueError:
        return None


def parse_gcode_args(args: list[str]) -> dict[str, float]:
    parsed_args = {}
    for arg in args:
        parsed_arg = parse_gcode_arg(arg)
        if parsed_arg is None:
            continue
        key, value = parsed_arg
        parsed_args[key] = value
    return parsed_args


@dataclass
class GcodeBounds:
    first_position: tuple[float, float, float]
    final_position: tuple[float, float, float]


@dataclass
class GcodeMove:
    index: int
    line: str
    parsed: GcodeCommand
    start_xyz: np.ndarray | None
    end_xyz: np.ndarray | None
    extrusion_delta: float
    has_xyz: bool
    is_absolute_xyz: bool


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
        lines = translate_gcode(lines, MACHINE_OFFSET)

        if index > 0:
            lines = apply_chunk_offsets(lines, chunks[index])
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
        lines = translate_gcode(lines, MACHINE_OFFSET)
        if index > 0:
            lines = apply_chunk_offsets(lines, chunks[index])
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
                False,
            )
        )
        debug_gcode.append(f"; --- END DEBUG TRANSITION {index - 1} TO {index} ---\n")

    debug_gcode.extend(
        [
            "\n; --- END PENTOS DEBUG TRANSITION CHECK ---\n",
            "G90 ; absolute movement\n",
            "G1 Z50 F1200\n",
            "G1 X0 Y235 F9000 ; present print\n",
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


def xyz_array(position: dict[str, float | None]) -> np.ndarray | None:
    x, y, z = position["X"], position["Y"], position["Z"]
    if x is None or y is None or z is None:
        return None
    return np.array([x, y, z])


def iter_gcode_moves(lines: Iterable[str]) -> Iterator[GcodeMove]:
    current: dict[str, float | None] = {
        "X": None,
        "Y": None,
        "Z": None,
        "E": 0.0,
        "A": 0.0,
        "B": 0.0,
    }
    absolute_xyz = True
    absolute_extrusion = True

    for index, line in enumerate(lines):
        parsed = GcodeCommand.parse(line)
        command = parsed.command
        args = parsed.args

        if command in {"G90", "G91", "M82", "M83"}:
            if command == "G90":
                absolute_xyz = True
            elif command == "G91":
                absolute_xyz = False
            elif command == "M82":
                absolute_extrusion = True
            elif command == "M83":
                absolute_extrusion = False

            continue

        if command == "G92":
            for key, value in args.items():
                if key in current:
                    current[key] = value

            continue

        if command not in {"G0", "G1"}:
            continue

        start_xyz = xyz_array(current)
        next_position = current.copy()
        extrusion_delta = 0.0
        has_xyz = False
        has_position_arg = False
        has_e = False

        for key, value in args.items():
            if key == "F":
                continue

            if key in {"X", "Y", "Z", "A", "B"}:
                has_position_arg = True
                if key in {"X", "Y", "Z"}:
                    has_xyz = True

                current_value = next_position[key]
                if absolute_xyz or current_value is None:
                    next_position[key] = value
                else:
                    next_position[key] = current_value + value
            elif key == "E":
                has_e = True
                current_e = float(current["E"] or 0.0)
                if absolute_extrusion:
                    extrusion_delta = value - current_e
                    next_position["E"] = value
                else:
                    extrusion_delta = value
                    next_position["E"] = current_e + value

        if has_position_arg or has_e:
            current = next_position

        yield GcodeMove(
            index=index,
            line=line,
            parsed=parsed,
            start_xyz=start_xyz,
            end_xyz=xyz_array(next_position),
            extrusion_delta=extrusion_delta,
            has_xyz=has_xyz,
            is_absolute_xyz=absolute_xyz,
        )


def translate_gcode(lines: list[str], offset: np.ndarray) -> list[str]:
    if not lines:
        return []

    translated = list(lines)
    for move in iter_gcode_moves(lines):
        if move.is_absolute_xyz:
            values = {}
            for key, index in (("X", 0), ("Y", 1), ("Z", 2)):
                if key in move.parsed.args:
                    values[key] = move.parsed.args[key] + offset[index]

            if values:
                stripped = move.line.rstrip("\r\n")
                ending = move.line[len(stripped) :]
                translated[move.index] = (
                    move.parsed.build_with_updated_args(values) + ending
                )

    return translated


def apply_chunk_offsets(lines: list[str], chunk: GcodeChunk) -> list[str]:
    if not lines:
        return []

    transformed = list(lines)
    initial_x, initial_y, initial_z = find_first_last_xyz(lines).first_position
    flat_position = {
        "X": initial_x,
        "Y": initial_y,
        "Z": initial_z,
    }

    for move in iter_gcode_moves(lines):
        if move.is_absolute_xyz and move.has_xyz:
            for key in ("X", "Y", "Z"):
                if key in move.parsed.args:
                    flat_position[key] = move.parsed.args[key]

            flat_point = np.array(
                [
                    flat_position["X"] - chunk.flat_xy_offset[0],
                    flat_position["Y"] - chunk.flat_xy_offset[1],
                    flat_position["Z"] + chunk.z_offset,
                ]
            )
            stripped = move.line.rstrip("\r\n")
            ending = move.line[len(stripped) :]
            transformed[move.index] = (
                move.parsed.build_with_updated_args(
                    {
                        "X": flat_point[0],
                        "Y": flat_point[1],
                        "Z": flat_point[2],
                    },
                )
                + ending
            )

    return transformed


def find_first_last_xyz(lines: list[str]) -> GcodeBounds:
    first_position: tuple[float, float, float] | None = None
    final_position: tuple[float, float, float] | None = None

    for move in iter_gcode_moves(lines):
        if move.end_xyz is None:
            continue

        x, y, z = move.end_xyz
        if first_position is None:
            first_position = (float(x), float(y), float(z))
        final_position = (float(x), float(y), float(z))

    assert first_position is not None and final_position is not None

    return GcodeBounds(
        first_position=first_position,
        final_position=final_position,
    )


def transition(
    initial_xyz: tuple[float, float, float],
    a_degrees: float,
    b_degrees: float,
    extrude=True,
) -> list[str]:
    gcode = [
        "\n; --- PENTOS A/B TRANSITION ---\n",
    ]
    if extrude:
        gcode.append("G1 E-5 F3600\n")

    gcode.extend(
        [
            "G91 ; relative movement for safe lift\n",
            "G1 Z10 F3000\n",
            "G90 ; absolute movement\n",
            f"G1 A{a_degrees} B{b_degrees} F1200\n",
            "; --- PENTOS MOVE TO NEXT CHUNK ---\n",
            f"G1 X{initial_xyz[0]} Y{initial_xyz[1]} F1200\n",
            f"G1 Z{initial_xyz[2]} F1200\n",
        ]
    )
    if extrude:
        gcode.append("G92 E0\n")

    gcode.append("; --- END PENTOS A/B TRANSITION ---\n")
    return gcode
