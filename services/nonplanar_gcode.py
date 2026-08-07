import math

import numpy as np

from gcode_tools import GcodeCommand, iter_gcode_moves, parse_gcode_arg
from machine import MACHINE_OFFSET, ROTATION_CENTER, rotation_matrix
from services.slicing import Slicer
from services.volumetric_deformation import TetrahedralVolume

MAX_EXTRUSION_MULTIPLIER = 10.0


def continuous_ab_angles(
    normal: np.ndarray,
    previous: tuple[float, float],
) -> tuple[float, float]:
    """Choose the equivalent A/B pose nearest the previous commanded pose to prevent jumps."""
    a_degrees, b_degrees = Slicer.ab_angles(normal)
    candidates = ((a_degrees, b_degrees), (-a_degrees, b_degrees + 180.0))
    unwrapped = [
        (a, b + 360.0 * round((previous[1] - b) / 360.0)) for a, b in candidates
    ]
    return min(
        unwrapped,
        key=lambda angles: (
            (angles[0] - previous[0]) ** 2 + (angles[1] - previous[1]) ** 2
        ),
    )


def map_gcode_to_original(
    text: str,
    volume: TetrahedralVolume,
    max_segment_length: float = 0.5,
) -> str:
    """Subdivide and inverse-map printable G-code moves through a tetrahedral volume."""
    if max_segment_length <= 0.0:
        raise ValueError("Maximum segment length must be positive")

    lines = text.splitlines(keepends=True)
    moves = {move.index: move for move in iter_gcode_moves(lines)}
    mapped_lines = []
    has_seen_layer = False
    previous_ab = (0.0, 0.0)
    last_emitted_xyz: np.ndarray | None = None

    for index, line in enumerate(lines):
        parsed = GcodeCommand.parse(line)
        if parsed.comment == "LAYER_CHANGE":
            has_seen_layer = True

        move = moves.get(index)
        if (
            not has_seen_layer
            or move is None
            or not move.is_absolute_xyz
            or not move.has_xyz
            or move.start_xyz is None
            or move.end_xyz is None
        ):
            mapped_lines.append(line)
            if move is not None and move.end_xyz is not None:
                last_emitted_xyz = move.end_xyz
            continue

        distance = float(np.linalg.norm(move.end_xyz - move.start_xyz))
        if distance > 0.0 and move.feedrate is None:
            raise ValueError("Mapped G-code move has no feedrate")
        segment_count = max(1, math.ceil(distance / max_segment_length))
        points = move.start_xyz + (
            np.arange(1, segment_count + 1)[:, None]
            / segment_count
            * (move.end_xyz - move.start_xyz)
        )
        if move.extrusion_delta > 0.0 and move.is_absolute_extrusion:
            raise ValueError(
                "Nonplanar extrusion compensation requires relative extrusion"
            )

        stripped = line.rstrip("\r\n")
        ending = line[len(stripped) :]
        try:
            local_points = points - MACHINE_OFFSET
            original_points = volume.map_to_original(local_points)
            extrusion_multipliers = (
                np.minimum(
                    volume.extrusion_multipliers(local_points),
                    MAX_EXTRUSION_MULTIPLIER,
                )
                if move.extrusion_delta > 0.0
                else np.ones(segment_count)
            )
            normals = volume.layer_normals(original_points)
            angles = []
            for normal in normals:
                previous_ab = continuous_ab_angles(normal, previous_ab)
                angles.append(previous_ab)
            center = np.asarray(ROTATION_CENTER)
            offset = np.asarray(MACHINE_OFFSET)
            mapped = np.asarray(
                [
                    offset
                    + center
                    + rotation_matrix(a_degrees, b_degrees) @ (point - center)
                    for point, (a_degrees, b_degrees) in zip(original_points, angles)
                ]
            )
        except ValueError:
            # keep out-of-volume setup and skirt moves planar for preview.
            previous_ab = (0.0, 0.0)
            mapped = np.asarray([move.end_xyz])
            angles = [previous_ab]
            extrusion_multipliers = np.ones(1)
            segment_count = 1

        if last_emitted_xyz is None:
            last_emitted_xyz = move.start_xyz
        mapped_starts = np.vstack((last_emitted_xyz, mapped[:-1]))
        mapped_lengths = np.linalg.norm(mapped - mapped_starts, axis=1)
        feedrates = (
            move.feedrate * mapped_lengths / (distance / segment_count)
            if distance > 0.0
            else [move.feedrate] * segment_count
        )

        for segment_index, (
            point,
            (a_degrees, b_degrees),
            extrusion_multiplier,
            feedrate,
        ) in enumerate(zip(mapped, angles, extrusion_multipliers, feedrates)):
            mapped_lines.append(
                _mapped_command(
                    parsed,
                    point,
                    segment_index,
                    segment_count,
                    move.extrusion_delta,
                    extrusion_multiplier,
                    feedrate,
                    a_degrees,
                    b_degrees,
                ).build()
                + ending
            )
        last_emitted_xyz = mapped[-1]

    return "".join(mapped_lines)


def _mapped_command(
    command: GcodeCommand,
    point: np.ndarray,
    segment_index: int,
    segment_count: int,
    extrusion_delta: float,
    extrusion_multiplier: float,
    feedrate: float | None,
    a_degrees: float,
    b_degrees: float,
) -> GcodeCommand:
    """Build one mapped segment while preserving non-position G-code arguments."""
    raw_args = [
        argument
        for argument in command.raw_args
        if (parsed := parse_gcode_arg(argument)) is None
        or parsed[0] not in {"X", "Y", "Z", "A", "B", "E", "F"}
    ]
    raw_args.extend(f"{axis}{value:.5f}" for axis, value in zip("XYZ", point))
    raw_args.extend((f"A{a_degrees:.5f}", f"B{b_degrees:.5f}"))
    if feedrate is not None:
        raw_args.append(f"F{feedrate:.5f}")
    if "E" in command.args:
        extrusion = extrusion_delta / segment_count
        if extrusion > 0.0:
            extrusion *= extrusion_multiplier
        raw_args.append(f"E{extrusion:.5f}")

    return GcodeCommand(
        command=command.command,
        raw_args=raw_args,
        comment=command.comment if segment_index == 0 else None,
    )
