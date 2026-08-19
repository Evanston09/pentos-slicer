import math

import numpy as np
from gcode_tools import GcodeCommand, iter_gcode_moves, parse_gcode_arg
from machine import rotation_matrix
from models import MachineConfig
from services.volumetric_deformation import TetrahedralVolume

MAX_EXTRUSION_MULTIPLIER = 10.0


def _ab_angles(
    normal: np.ndarray,
    previous_b: float,
    machine_config: MachineConfig,
) -> tuple[float, float]:
    """Choose the nearest legal pose of a desired normal on a point in model within the allowed normal error."""
    normal /= np.linalg.norm(normal)
    # A unit normal tilted by θ from vertical has horizontal magnitude sin(θ).
    tolerance = np.sin(np.radians(machine_config.max_normal_error_degrees))
    target_b = (np.degrees(np.arctan2(-normal[1], -normal[0])) + 180.0) % 360.0 - 180.0
    horizontal = np.hypot(normal[0], normal[1])
    width = (
        180.0
        if horizontal <= tolerance
        else np.degrees(np.arcsin(tolerance / horizontal))
    )
    min_turn = math.ceil((machine_config.b_degrees_min - target_b) / 180.0)
    max_turn = math.floor((machine_config.b_degrees_max - target_b) / 180.0)
    candidates = [
        target_b + 180.0 * turn for turn in range(min_turn, max_turn + 1)
    ]
    center = min(candidates, key=lambda value: abs(value - previous_b))
    b_degrees = float(
        np.clip(
            previous_b,
            max(machine_config.b_degrees_min, center - width),
            min(machine_config.b_degrees_max, center + width),
        )
    )
    b = np.radians(b_degrees)
    # Positive A points towards -X
    horizontal = -(normal[0] * np.cos(b) + normal[1] * np.sin(b))
    return float(np.degrees(np.arctan2(horizontal, normal[2]))), b_degrees


def map_gcode_to_original(
    text: str,
    volume: TetrahedralVolume,
    machine_config: MachineConfig,
    max_segment_length: float = 0.5,
) -> str:
    """Subdivide and inverse-map printable G-code moves through a tetrahedral volume."""
    if max_segment_length <= 0.0:
        raise ValueError("Maximum segment length must be positive")

    lines = text.splitlines(keepends=True)
    moves = {move.index: move for move in iter_gcode_moves(lines)}
    mapped_lines = []
    has_seen_layer = False
    previous_b = 0.0
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
            if parsed.command == "ENABLE_FIVE_AXIS":
                mapped_lines.extend(
                    (
                        "MANUAL_STEPPER STEPPER=a_motor GCODE_AXIS=A "
                        f"LIMIT_VELOCITY={machine_config.a_max_velocity_deg_s:g} "
                        f"LIMIT_ACCEL={machine_config.a_max_acceleration_deg_s2:g}\n",
                        "MANUAL_STEPPER STEPPER=b_motor GCODE_AXIS=B "
                        f"LIMIT_VELOCITY={machine_config.b_max_velocity_deg_s:g} "
                        f"LIMIT_ACCEL={machine_config.b_max_acceleration_deg_s2:g}\n",
                    )
                )
            if move is not None and move.end_xyz is not None:
                last_emitted_xyz = move.end_xyz
            continue

        distance = float(np.linalg.norm(move.end_xyz - move.start_xyz))
        if distance > 0.0 and move.feedrate is None:
            raise ValueError("Mapped G-code move has no feedrate")
        if move.extrusion_delta > 0.0 and move.is_absolute_extrusion:
            raise ValueError(
                "Nonplanar extrusion compensation requires relative extrusion"
            )

        stripped = line.rstrip("\r\n")
        ending = line[len(stripped) :]
        segment_count = max(1, math.ceil(distance / max_segment_length))
        points = move.start_xyz + (
            np.arange(1, segment_count + 1)[:, None]
            / segment_count
            * (move.end_xyz - move.start_xyz)
        )
        try:
            local_points = points - machine_config.machine_offset
            original_points, extrusion_multipliers, normals = (
                volume.inverse_map_properties(local_points)
            )
            angles = []
            for normal in normals:
                angle = _ab_angles(normal, previous_b, machine_config)
                angles.append(angle)
                previous_b = angle[1]
            if move.extrusion_delta > 0.0:
                extrusion_multipliers = np.minimum(
                    extrusion_multipliers,
                    MAX_EXTRUSION_MULTIPLIER,
                )
            else:
                extrusion_multipliers = np.ones(segment_count)
            center = np.asarray(machine_config.rotation_center_local_mm)
            offset = np.asarray(machine_config.machine_offset)
            mapped = np.asarray(
                [
                    offset
                    + center
                    + rotation_matrix(a_degrees, b_degrees) @ (point - center)
                    for point, (a_degrees, b_degrees) in zip(original_points, angles)
                ]
            )
        except ValueError:
            # Keep out-of-volume setup and skirt moves planar for preview.
            previous_b = 0.0
            mapped = np.asarray([move.end_xyz])
            angles = [(0.0, 0.0)]
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
