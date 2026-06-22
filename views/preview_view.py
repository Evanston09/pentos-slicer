from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import viser

from app_state import AppState
from gcode_tools import GcodeCommand, parse_gcode_args
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


def parse_gcode_preview(text: str) -> GcodePreview:
    current: dict[str, float | None] = {
        "X": None,
        "Y": None,
        "Z": None,
        "E": 0.0,
        "A": 0.0,
        "B": 0.0,
    }
    absolute_positioning = True
    absolute_extrusion = True
    motion_mode: str | None = None
    has_seen_layer = False
    in_transition = False
    setup_segments: list[list[np.ndarray]] = []
    part_travel_segments: list[list[np.ndarray]] = []
    part_extrusion_segments: list[list[np.ndarray]] = []
    parts: list[GcodePreviewPart] = []

    for line in text.splitlines():
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

        command_word = parsed.command
        args = parsed.args
        if command_word and command_word[:1] not in {"G", "M"}:
            args = parse_gcode_args([parsed.command, *parsed.raw_args])
            command_word = ""

        if command_word in {"G0", "G1"}:
            motion_mode = command_word
        elif command_word == "G90":
            absolute_positioning = True
            continue
        elif command_word == "G91":
            absolute_positioning = False
            continue
        elif command_word == "M82":
            absolute_extrusion = True
            continue
        elif command_word == "M83":
            absolute_extrusion = False
            continue

        if command_word not in {"G0", "G1"} and motion_mode not in {"G0", "G1"}:
            continue

        next_position = current.copy()
        extrusion_delta = 0.0
        has_xyz = False
        has_motion_word = False

        for key, value in args.items():
            if key == "F":
                continue

            if key in {"X", "Y", "Z", "A", "B"}:
                has_motion_word = True
                if key in {"X", "Y", "Z"}:
                    has_xyz = True

                current_value = next_position[key]
                if absolute_positioning or current_value is None:
                    next_position[key] = value
                else:
                    next_position[key] = current_value + value
            elif key == "E":
                current_e = float(current["E"] or 0.0)
                if absolute_extrusion:
                    extrusion_delta = value - current_e
                    next_position["E"] = value
                else:
                    extrusion_delta = value
                    next_position["E"] = current_e + value

        if has_xyz and all(current[key] is not None for key in ("X", "Y", "Z")):
            start = np.array(
                [current["X"], current["Y"], current["Z"]],
            )
            end = np.array(
                [next_position["X"], next_position["Y"], next_position["Z"]],
            )
            segment = [start, end]
            if not in_transition:
                if not has_seen_layer:
                    setup_segments.append(segment)
                elif extrusion_delta > 0:
                    part_extrusion_segments.append(segment)
                else:
                    part_travel_segments.append(segment)

        if has_motion_word or extrusion_delta != 0:
            current = next_position

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
        self.setup_handle = None
        self.travel_handles = []
        self.extrusion_handles = []
