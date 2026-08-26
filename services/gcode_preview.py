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

        if move.has_xyz and move.start_xyz is not None:
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

    return GcodePreview(simulation_steps)
