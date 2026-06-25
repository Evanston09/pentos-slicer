from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

import numpy as np

from .commands import GcodeCommand


class GcodeChunk(Protocol):
    a_degrees: float
    b_degrees: float
    z_offset: float
    flat_xy_offset: list[float]


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
                current_e = current["E"] or 0.0
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

            stripped = move.line.rstrip("\r\n")
            ending = move.line[len(stripped) :]
            transformed[move.index] = (
                move.parsed.build_with_updated_args(
                    {
                        "X": flat_position["X"] - chunk.flat_xy_offset[0],
                        "Y": flat_position["Y"] - chunk.flat_xy_offset[1],
                        "Z": flat_position["Z"] + chunk.z_offset,
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
            first_position = (x, y, z)
        final_position = (x, y, z)

    assert first_position is not None and final_position is not None

    return GcodeBounds(
        first_position=first_position,
        final_position=final_position,
    )
