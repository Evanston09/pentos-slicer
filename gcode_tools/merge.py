from pathlib import Path
from typing import Sequence

from machine import MACHINE_OFFSET

from .moves import (
    GcodeChunk,
    apply_chunk_offsets,
    find_first_last_xyz,
    translate_gcode,
)
from .transitions import transition
from .trimming import remove_leading_retract, trim_gcode


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
            lines = remove_leading_retract(lines)
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
        if index > 0:
            lines = remove_leading_retract(lines)
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
