import numpy as np

from gcode_tools import GcodeCommand, iter_gcode_moves
from machine import rotation_matrix
from models import GcodePreview, MachineConfig, MachinePose, PreviewMove


def transform_preview_point(
    point: np.ndarray,
    a_degrees: float,
    b_degrees: float,
    machine_config: MachineConfig,
) -> np.ndarray:
    local_point = point - machine_config.machine_offset
    if np.isclose(a_degrees, 0.0) and np.isclose(b_degrees, 0.0):
        return local_point

    # Merged G-code is in the rotated machine pose; preview in object space.
    rotation = rotation_matrix(a_degrees, b_degrees)
    center = np.asarray(machine_config.rotation_center_local_mm)
    return center + rotation.T @ (local_point - center)


def parse_gcode_preview(
    text: str,
    machine_config: MachineConfig,
) -> GcodePreview:
    has_seen_layer = False
    in_transition = False
    part_index = 0
    part_has_segments = False
    simulation_steps: list[PreviewMove] = []
    motion_time_seconds = 0.0
    motion_times = [0.0]
    a_degrees = [0.0]
    b_degrees = [0.0]
    has_rotary_commands = False

    lines = text.splitlines()
    moves_by_index = {move.index: move for move in iter_gcode_moves(lines)}

    for index, line in enumerate(lines):
        parsed = GcodeCommand.parse(line)
        comment = parsed.comment

        if comment == "LAYER_CHANGE":
            in_transition = False
            has_seen_layer = True
            continue
        if comment == "--- PENTOS A/B TRANSITION ---":
            if part_has_segments:
                part_index += 1
                part_has_segments = False
            in_transition = True
            continue
        if comment == "--- END PENTOS A/B TRANSITION ---":
            in_transition = False
            continue

        move = moves_by_index.get(index)
        if move is None:
            continue

        if "A" in move.parsed.args or "B" in move.parsed.args:
            has_rotary_commands = True
        duration = 0.0
        if (
            move.start_xyz is not None
            and move.end_xyz is not None
            and move.feedrate is not None
            and move.feedrate > 0.0
        ):
            distance = float(np.linalg.norm(move.end_xyz - move.start_xyz))
            duration = (distance / move.feedrate) * 60.0
        motion_time_seconds += duration
        if duration > 0.0 or not np.array_equal(
            move.end_ab, [a_degrees[-1], b_degrees[-1]]
        ):
            motion_times.append(motion_time_seconds)
            a_degrees.append(float(move.end_ab[0]))
            b_degrees.append(float(move.end_ab[1]))

        if move.end_xyz is None:
            continue

        has_ab = not np.allclose(move.start_ab, move.end_ab)
        if not move.has_xyz and not has_ab:
            continue

        segment = None
        step_part_index = None
        if in_transition:
            kind = "transition"
        elif not has_seen_layer:
            kind = "setup"
        elif move.extrusion_delta > 0:
            kind = "extrusion"
            step_part_index = part_index
        else:
            kind = "travel"
            step_part_index = part_index

        if not in_transition and move.has_xyz and move.start_xyz is not None:
            start = transform_preview_point(
                move.start_xyz, *move.start_ab, machine_config
            )
            end = transform_preview_point(move.end_xyz, *move.end_ab, machine_config)
            segment = np.asarray([start, end])
            if kind in {"extrusion", "travel"}:
                part_has_segments = True

        simulation_steps.append(
            PreviewMove(
                pose=MachinePose(move.end_xyz, move.end_ab),
                preview_segment=segment,
                kind=kind,
                part_index=step_part_index,
            )
        )
    if not has_rotary_commands:
        motion_times = []
        a_degrees = []
        b_degrees = []
    return GcodePreview(
        simulation_steps=simulation_steps,
        motion_time_seconds=np.asarray(motion_times),
        a_degrees=np.asarray(a_degrees),
        b_degrees=np.asarray(b_degrees),
    )
