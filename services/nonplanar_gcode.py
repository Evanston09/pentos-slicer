import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import bisect
from scipy.spatial import geometric_slerp

from gcode_tools import GcodeCommand, GcodeMove, iter_gcode_moves, parse_gcode_arg
from machine import rotation_matrix
from models import MachineConfig

MAX_EXTRUSION_MULTIPLIER = 10.0
PLANAR_BASE_LAYERS = 2
NONPLANAR_TRANSITION_LAYERS = 4
ORIENTATION_SMOOTHING_TIME_SECONDS = 0.05


@dataclass
class _MappedMoveProperties:
    original_points: np.ndarray
    extrusion_multipliers: np.ndarray
    normals: np.ndarray
    segment_duration: float
    angles: np.ndarray


# Lightweight adapter for tests
class _InverseMappingVolume(Protocol):
    def inverse_map_properties(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def _ab_angles(
    normal: np.ndarray,
    previous_b: float,
    machine_config: MachineConfig,
    max_error_degrees: float,
) -> tuple[float, float]:
    """Choose the nearest legal pose of a desired normal on a point in model within the allowed normal error."""
    normal = normal / np.linalg.norm(normal)
    # A unit normal tilted by θ from vertical has horizontal magnitude sin(θ).
    tolerance = np.sin(np.radians(max_error_degrees))
    target_b = (np.degrees(np.arctan2(-normal[1], -normal[0])) + 180.0) % 360.0 - 180.0
    horizontal = np.hypot(normal[0], normal[1])
    width = (
        180.0
        if horizontal <= tolerance
        else np.degrees(np.arcsin(tolerance / horizontal))
    )
    min_turn = math.ceil((machine_config.b_degrees_min - target_b) / 180.0)
    max_turn = math.floor((machine_config.b_degrees_max - target_b) / 180.0)
    turn = min(max(round((previous_b - target_b) / 180.0), min_turn), max_turn)
    center = target_b + 180.0 * turn
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


def _smooth_normals(
    normals: np.ndarray,
    segment_durations: np.ndarray,
    max_error_degrees: float,
) -> np.ndarray:
    """Time-domain smooth a path's normals within its orientation-error limit."""
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if len(normals) < 2:
        return normals.copy()

    amounts = -np.expm1(
        -np.maximum(segment_durations, 0.0) / ORIENTATION_SMOOTHING_TIME_SECONDS
    )
    forward = normals.copy()
    for index in range(1, len(normals)):
        forward[index] = geometric_slerp(
            forward[index - 1], normals[index], [amounts[index]]
        )[0]
    backward = normals.copy()
    for index in range(len(normals) - 2, -1, -1):
        backward[index] = geometric_slerp(
            backward[index + 1], normals[index], [amounts[index + 1]]
        )[0]
    smoothed = np.asarray(
        [
            geometric_slerp(before, after, [0.5])[0]
            for before, after in zip(forward, backward)
        ]
    )

    maximum = np.radians(max_error_degrees)
    for index, (target, candidate) in enumerate(zip(normals, smoothed)):
        error = math.acos(float(np.clip(np.dot(target, candidate), -1.0, 1.0)))
        if error > maximum:
            smoothed[index] = geometric_slerp(target, candidate, [maximum / error])[0]
    return smoothed


def _commanded_normal(angle: np.ndarray) -> np.ndarray:
    return rotation_matrix(*angle).T @ np.array([0.0, 0.0, 1.0])


def _smooth_angles(
    angles: np.ndarray,
    source_normals: np.ndarray,
    segment_durations: np.ndarray,
    machine_config: MachineConfig,
) -> np.ndarray:
    """Time-domain smooth A/B commands without exceeding normal-error limits."""
    if len(angles) < 2:
        return angles.copy()

    # Filter across the B wrap instead of treating -180 and 180 as far apart.
    unwrapped = angles.copy()
    unwrapped[:, 1] = np.degrees(np.unwrap(np.radians(unwrapped[:, 1])))
    amounts = -np.expm1(
        -np.maximum(segment_durations, 0.0) / ORIENTATION_SMOOTHING_TIME_SECONDS
    )
    forward = unwrapped.copy()
    for index in range(1, len(unwrapped)):
        forward[index] = forward[index - 1] + amounts[index] * (
            unwrapped[index] - forward[index - 1]
        )
    backward = unwrapped.copy()
    for index in range(len(unwrapped) - 2, -1, -1):
        backward[index] = backward[index + 1] + amounts[index + 1] * (
            unwrapped[index] - backward[index + 1]
        )
    smoothed = (forward + backward) / 2.0

    minimum_dot = math.cos(math.radians(machine_config.max_normal_error_degrees))
    source_normals = source_normals / np.linalg.norm(
        source_normals, axis=1, keepdims=True
    )
    for index, (source, original, candidate) in enumerate(
        zip(source_normals, unwrapped, smoothed)
    ):
        delta = candidate - original

        def error_margin(amount: float) -> float:
            return (
                np.dot(source, _commanded_normal(original + amount * delta))
                - minimum_dot
            )

        if error_margin(1.0) < 0.0:
            smoothed[index] = original + bisect(error_margin, 0.0, 1.0) * delta

    # Return equivalent B coordinates in the configured machine range.
    turns = np.round((angles[:, 1] - smoothed[:, 1]) / 360.0)
    smoothed[:, 1] = np.clip(
        smoothed[:, 1] + 360.0 * turns,
        machine_config.b_degrees_min,
        machine_config.b_degrees_max,
    )
    return smoothed


def _prepare_mapped_moves(
    lines: list[str],
    moves: dict[int, GcodeMove],
    volume: _InverseMappingVolume,
    machine_config: MachineConfig,
    max_segment_length: float,
) -> dict[int, _MappedMoveProperties]:
    """Inverse-map moves and smooth normals within continuous motion paths."""
    prepared: dict[int, _MappedMoveProperties] = {}
    groups: list[list[tuple[int, int]]] = []
    current_group: list[tuple[int, int]] | None = None
    layer_index = -1

    for index, line in enumerate(lines):
        parsed = GcodeCommand.parse(line)
        if parsed.comment == "LAYER_CHANGE":
            layer_index += 1

        move = moves.get(index)
        if layer_index < PLANAR_BASE_LAYERS:
            continue
        if move is None:
            if parsed.command in {"G90", "G91"} or (
                parsed.command == "G92" and any(axis in parsed.args for axis in "XYZAB")
            ):
                current_group = None
            continue
        if (
            not move.is_absolute_xyz
            or not move.has_xyz
            or move.start_xyz is None
            or move.end_xyz is None
        ):
            if move.has_xyz:
                current_group = None
            continue

        distance = float(np.linalg.norm(move.end_xyz - move.start_xyz))
        if distance > 0.0 and (move.feedrate is None or move.feedrate <= 0.0):
            raise ValueError("Mapped G-code move has no positive feedrate")
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
        except ValueError:
            current_group = None
            continue

        blend = min(
            1.0,
            (layer_index - PLANAR_BASE_LAYERS + 1) / NONPLANAR_TRANSITION_LAYERS,
        )
        original_points = local_points + blend * (original_points - local_points)
        extrusion_multipliers = 1.0 + blend * (extrusion_multipliers - 1.0)
        normals = np.array([0.0, 0.0, 1.0]) + blend * (normals - [0.0, 0.0, 1.0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        duration = (60.0 * distance) / move.feedrate if distance > 0.0 else 0.0
        prepared[index] = _MappedMoveProperties(
            original_points,
            extrusion_multipliers,
            normals,
            duration / segment_count,
            np.empty((segment_count, 2)),
        )

        if current_group is None:
            current_group = []
            groups.append(current_group)
        current_group.extend((index, segment) for segment in range(segment_count))

    previous_b = 0.0
    mapping_error = machine_config.max_normal_error_degrees / 2.0
    for group in groups:
        source_normals = np.asarray(
            [prepared[index].normals[segment] for index, segment in group]
        )
        durations = np.asarray([prepared[index].segment_duration for index, _ in group])
        smoothed_normals = _smooth_normals(source_normals, durations, mapping_error)
        angles = []
        for normal in smoothed_normals:
            angle = _ab_angles(normal, previous_b, machine_config, mapping_error)
            angles.append(angle)
            previous_b = angle[1]
        angles = _smooth_angles(
            np.asarray(angles), source_normals, durations, machine_config
        )
        previous_b = float(angles[-1, 1])
        for (index, segment), angle in zip(group, angles):
            prepared[index].angles[segment] = angle

    return prepared


def map_gcode_to_original(
    text: str,
    volume: _InverseMappingVolume,
    machine_config: MachineConfig,
    max_segment_length: float = 0.5,
) -> str:
    """Subdivide and inverse-map printable G-code moves through a tetrahedral volume."""
    if max_segment_length <= 0.0:
        raise ValueError("Maximum segment length must be positive")

    lines = text.splitlines(keepends=True)
    moves = {move.index: move for move in iter_gcode_moves(lines)}
    prepared_moves = _prepare_mapped_moves(
        lines,
        moves,
        volume,
        machine_config,
        max_segment_length,
    )
    mapped_lines = []
    layer_index = -1
    last_emitted_xyz: np.ndarray | None = None
    last_emitted_ab = (0.0, 0.0)

    for index, line in enumerate(lines):
        parsed = GcodeCommand.parse(line)
        if parsed.comment == "LAYER_CHANGE":
            layer_index += 1

        move = moves.get(index)
        if (
            layer_index < PLANAR_BASE_LAYERS
            or move is None
            or not move.is_absolute_xyz
            or not move.has_xyz
            or move.start_xyz is None
            or move.end_xyz is None
        ):
            mapped_lines.append(line)
            if move is not None and move.has_xyz and move.end_xyz is not None:
                last_emitted_xyz = move.end_xyz
            continue

        if last_emitted_xyz is None:
            last_emitted_xyz = move.start_xyz
        distance = float(np.linalg.norm(move.end_xyz - move.start_xyz))
        if move.extrusion_delta > 0.0 and move.is_absolute_extrusion:
            raise ValueError(
                "Nonplanar extrusion compensation requires relative extrusion"
            )

        stripped = line.rstrip("\r\n")
        ending = line[len(stripped) :]
        properties = prepared_moves.get(index)
        if properties is not None:
            original_points = properties.original_points
            extrusion_multipliers = properties.extrusion_multipliers
            angles = properties.angles
            segment_count = len(original_points)
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
        else:
            # Anchor unmapped travel to the last mapped point in the held bed pose.
            mapped = np.asarray(
                [
                    last_emitted_xyz
                    + rotation_matrix(*last_emitted_ab)
                    @ (move.end_xyz - move.start_xyz)
                ]
            )
            angles = [last_emitted_ab]
            extrusion_multipliers = np.ones(1)
            segment_count = 1

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
        last_emitted_ab = tuple(angles[-1])

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
