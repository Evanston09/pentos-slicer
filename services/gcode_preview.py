import numpy as np

from gcode_tools import GcodeCommand, iter_gcode_moves
from machine import MACHINE_OFFSET, ROTATION_CENTER, rotation_matrix
from models import GcodePreview, GcodePreviewPart

SETUP_COLOR = (255, 130, 0)
PART_COLORS = [
    (47, 153, 238),
    (255, 130, 0),
    (34, 197, 94),
    (236, 72, 153),
    (168, 85, 247),
    (20, 184, 166),
]


def transform_preview_point(
    point: np.ndarray,
    a_degrees: float,
    b_degrees: float,
) -> np.ndarray:
    local_point = point - MACHINE_OFFSET
    if np.isclose(a_degrees, 0.0) and np.isclose(b_degrees, 0.0):
        return local_point

    # Merged G-code is in the rotated machine pose; preview in object space.
    rotation = rotation_matrix(a_degrees, b_degrees)
    return ROTATION_CENTER + rotation.T @ (local_point - ROTATION_CENTER)


def parse_gcode_preview(text: str) -> GcodePreview:
    has_seen_layer = False
    in_transition = False
    setup_segments: list[list[np.ndarray]] = []
    part_travel_segments: list[list[np.ndarray]] = []
    part_extrusion_segments: list[list[np.ndarray]] = []
    parts: list[GcodePreviewPart] = []

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
            if part_travel_segments or part_extrusion_segments:
                part_index = len(parts)
                parts.append(
                    GcodePreviewPart(
                        travel=np.asarray(part_travel_segments),
                        extrusion=np.asarray(part_extrusion_segments),
                        color=PART_COLORS[part_index % len(PART_COLORS)],
                    )
                )
                part_travel_segments = []
                part_extrusion_segments = []
            in_transition = True
            continue
        if comment == "--- END PENTOS A/B TRANSITION ---":
            in_transition = False
            continue

        move = moves_by_index.get(index)
        if move is None:
            continue

        if move.has_xyz and move.start_xyz is not None and move.end_xyz is not None:
            start = transform_preview_point(move.start_xyz, *move.start_ab)
            end = transform_preview_point(move.end_xyz, *move.end_ab)
            segment = [start, end]
            if not in_transition:
                if not has_seen_layer:
                    setup_segments.append(segment)
                elif move.extrusion_delta > 0:
                    part_extrusion_segments.append(segment)
                else:
                    part_travel_segments.append(segment)

    if part_travel_segments or part_extrusion_segments:
        part_index = len(parts)
        parts.append(
            GcodePreviewPart(
                travel=np.asarray(part_travel_segments),
                extrusion=np.asarray(part_extrusion_segments),
                color=PART_COLORS[part_index % len(PART_COLORS)],
            )
        )
    return GcodePreview(setup=np.asarray(setup_segments), parts=parts)
