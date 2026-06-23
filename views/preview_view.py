from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import viser

from app_state import AppState
from gcode_tools import GcodeCommand, iter_gcode_moves
from integrations import send_to_moonraker
from machine import MACHINE_OFFSET, ROTATION_CENTER, rotation_matrix
from theming import PENTOS_BLUE, PENTOS_ORANGE

SETUP_COLOR = np.array(PENTOS_ORANGE)
PART_COLORS = [
    np.array(PENTOS_BLUE),
    np.array(PENTOS_ORANGE),
    np.array([34, 197, 94]),
    np.array([236, 72, 153]),
    np.array([168, 85, 247]),
    np.array([20, 184, 166]),
]


@dataclass
class GcodePreviewPart:
    travel: np.ndarray
    extrusion: np.ndarray
    color: np.ndarray


@dataclass
class GcodePreview:
    setup: np.ndarray
    parts: list[GcodePreviewPart]


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
    a_degrees = 0.0
    b_degrees = 0.0
    setup_segments: list[list[np.ndarray]] = []
    part_travel_segments: list[list[np.ndarray]] = []
    part_extrusion_segments: list[list[np.ndarray]] = []
    parts: list[GcodePreviewPart] = []

    lines = text.splitlines()
    moves_by_index = {move.index: move for move in iter_gcode_moves(lines)}

    for index, line in enumerate(lines):
        parsed = GcodeCommand.parse(line)
        comment = parsed.comment

        if parsed.command in {"G0", "G1"}:
            if "A" in parsed.args:
                a_degrees = parsed.args["A"]
            if "B" in parsed.args:
                b_degrees = parsed.args["B"]

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
            start = transform_preview_point(
                move.start_xyz,
                a_degrees,
                b_degrees,
            )
            end = transform_preview_point(
                move.end_xyz,
                a_degrees,
                b_degrees,
            )
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


class PreviewView:
    def __init__(
        self,
        server: viser.ViserServer,
        state: AppState,
        show_setup: Callable[[], None],
    ) -> None:
        self.server = server
        self.state = state
        self.show_setup = show_setup
        self.status: Any | None = None
        self.output_path: Any | None = None
        self.show_travel: Any | None = None
        self.line_width: Any | None = None
        self.back_button: Any | None = None
        self.setup_handle: Any | None = None
        self.send_handle: Any | None = None
        self.send_print_handle: Any | None = None
        self.travel_handles: list[Any] = []
        self.extrusion_handles: list[Any] = []

    def mount(self) -> None:
        self.status = self.server.gui.add_text(
            "Status",
            "Saved G-code",
            disabled=True,
        )
        self.output_path = self.server.gui.add_text(
            "Output G-code",
            ("" if self.state.gcode_path is None else str(self.state.gcode_path)),
            disabled=True,
        )
        self.show_travel = self.server.gui.add_checkbox("Travel", True)
        self.line_width = self.server.gui.add_number(
            "Line width",
            2.0,
            min=1.0,
            max=10.0,
        )
        self.send_handle = self.server.gui.add_button(
            "Send to Moonraker",
            icon=viser.Icon.UPLOAD,
        )
        self.send_print_handle = self.server.gui.add_button(
            "Send and Print",
            icon=viser.Icon.PLAYER_PLAY,
        )
        self.back_button = self.server.gui.add_button("Back to Setup")

        self.load_preview()

        @self.show_travel.on_update
        def _(_) -> None:
            visible = bool(self.show_travel.value)
            if self.setup_handle is not None:
                self.setup_handle.visible = visible
            for travel_handle in self.travel_handles:
                travel_handle.visible = visible

        @self.line_width.on_update
        def _(_) -> None:
            line_width = float(self.line_width.value)
            for extrusion_handle in self.extrusion_handles:
                extrusion_handle.line_width = line_width
            if self.setup_handle is not None:
                self.setup_handle.line_width = max(1.0, line_width * 0.5)
            for travel_handle in self.travel_handles:
                travel_handle.line_width = max(1.0, line_width * 0.5)

        @self.back_button.on_click
        def _(_) -> None:
            self.show_setup()

        @self.send_handle.on_click
        def _(_) -> None:
            self.handle_send_to_moonraker(start_print=False)

        @self.send_print_handle.on_click
        def _(_) -> None:
            self.handle_send_to_moonraker(start_print=True)

    def load_preview(self) -> None:
        if self.state.gcode_path is None:
            if self.status is not None:
                self.status.value = "No G-code generated"
            return

        try:
            text = self.state.gcode_path.read_text()
            preview = parse_gcode_preview(text)
        except Exception as exc:
            if self.status is not None:
                self.status.value = f"Failed to preview G-code: {exc}"
            return

        self.show_preview(preview)
        extrusion_count = sum(len(part.extrusion) for part in preview.parts)
        travel_count = sum(len(part.travel) for part in preview.parts)
        if self.status is not None:
            self.status.value = (
                f"Preview: {len(preview.parts)} parts, "
                f"{extrusion_count} extrusion, "
                f"{travel_count} travel, "
                f"{len(preview.setup)} setup"
            )

    def handle_send_to_moonraker(self, start_print: bool) -> None:
        if self.state.gcode_path is None:
            if self.status is not None:
                self.status.value = "No G-code generated"
            return

        if self.status is not None:
            self.status.value = "Sending print"

        try:
            result = send_to_moonraker(
                self.state.gcode_path,
                start_print=start_print,
            )
        except Exception as exc:
            if self.status is not None:
                self.status.value = f"Moonraker upload failed: {exc}"
            return

        if self.status is not None:
            if start_print:
                self.status.value = f"Sent {self.state.gcode_path.name}"

    def show_preview(self, preview: GcodePreview) -> None:
        line_width = (
            float(self.line_width.value) if self.line_width is not None else 2.0
        )
        travel_visible = (
            bool(self.show_travel.value) if self.show_travel is not None else True
        )

        if len(preview.setup):
            self.setup_handle = self.server.scene.add_line_segments(
                "/preview/setup",
                points=preview.setup,
                colors=SETUP_COLOR,
                line_width=max(1.0, line_width * 0.5),
                visible=travel_visible,
            )

        for index, part in enumerate(preview.parts):
            if len(part.extrusion):
                self.extrusion_handles.append(
                    self.server.scene.add_line_segments(
                        f"/preview/part_{index}/extrusion",
                        points=part.extrusion,
                        colors=part.color,
                        line_width=line_width,
                    )
                )

            if len(part.travel):
                self.travel_handles.append(
                    self.server.scene.add_line_segments(
                        f"/preview/part_{index}/travel",
                        points=part.travel,
                        colors=part.color,
                        line_width=max(1.0, line_width * 0.5),
                        visible=travel_visible,
                    )
                )

    def unmount(self) -> None:
        for handle in (
            *self.extrusion_handles,
            *self.travel_handles,
            self.setup_handle,
            self.back_button,
            self.line_width,
            self.show_travel,
            self.send_print_handle,
            self.send_handle,
            self.output_path,
            self.status,
        ):
            if handle is not None:
                handle.remove()

        self.status = None
        self.output_path = None
        self.show_travel = None
        self.line_width = None
        self.back_button = None
        self.send_handle = None
        self.send_print_handle = None
        self.setup_handle = None
        self.travel_handles = []
        self.extrusion_handles = []
