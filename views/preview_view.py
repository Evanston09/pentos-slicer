from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import viser

from models import GcodePreview
from views.theming import PENTOS_ORANGE

if TYPE_CHECKING:
    from controllers.preview_controller import PreviewController

SETUP_COLOR = PENTOS_ORANGE


class PreviewView:
    def __init__(self, client: viser.ClientHandle) -> None:
        self.client = client
        self.controller: PreviewController | None = None
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

    def bind_controller(self, controller: PreviewController) -> None:
        self.controller = controller

    def mount(self, gcode_path: Path | None) -> None:
        self.status = self.client.gui.add_text(
            "Status",
            "Saved G-code",
            disabled=True,
        )
        self.output_path = self.client.gui.add_text(
            "Output G-code",
            "" if gcode_path is None else str(gcode_path),
            disabled=True,
        )
        self.show_travel = self.client.gui.add_checkbox("Travel", True)
        self.line_width = self.client.gui.add_number(
            "Line width",
            2.0,
            min=1.0,
            max=10.0,
        )
        self.send_handle = self.client.gui.add_button(
            "Send to Moonraker",
            icon=viser.Icon.UPLOAD,
        )
        self.send_print_handle = self.client.gui.add_button(
            "Send and Print",
            icon=viser.Icon.PLAYER_PLAY,
        )
        self.back_button = self.client.gui.add_button("Back to Setup")

        @self.show_travel.on_update
        def _(_) -> None:
            visible = self.show_travel.value
            if self.setup_handle is not None:
                self.setup_handle.visible = visible
            for travel_handle in self.travel_handles:
                travel_handle.visible = visible

        @self.line_width.on_update
        def _(_) -> None:
            line_width = self.line_width.value
            for extrusion_handle in self.extrusion_handles:
                extrusion_handle.line_width = line_width
            if self.setup_handle is not None:
                self.setup_handle.line_width = max(1.0, line_width * 0.5)
            for travel_handle in self.travel_handles:
                travel_handle.line_width = max(1.0, line_width * 0.5)

        @self.back_button.on_click
        def _(_) -> None:
            if self.controller is not None:
                self.controller.show_setup()

        @self.send_handle.on_click
        def _(_) -> None:
            if self.controller is not None:
                self.controller.send_to_moonraker(start_print=False)

        @self.send_print_handle.on_click
        def _(_) -> None:
            if self.controller is not None:
                self.controller.send_to_moonraker(start_print=True)

    def set_status(self, message: str) -> None:
        if self.status is not None:
            self.status.value = message

    def show_preview(self, preview: GcodePreview) -> None:
        line_width = self.line_width.value if self.line_width is not None else 2.0
        travel_visible = (
            self.show_travel.value if self.show_travel is not None else True
        )

        if len(preview.setup):
            self.setup_handle = self.client.scene.add_line_segments(
                "/preview/setup",
                points=preview.setup,
                colors=SETUP_COLOR,
                line_width=max(1.0, line_width * 0.5),
                visible=travel_visible,
            )

        for index, part in enumerate(preview.parts):
            if len(part.extrusion):
                self.extrusion_handles.append(
                    self.client.scene.add_line_segments(
                        f"/preview/part_{index}/extrusion",
                        points=part.extrusion,
                        colors=part.color,
                        line_width=line_width,
                    )
                )

            if len(part.travel):
                self.travel_handles.append(
                    self.client.scene.add_line_segments(
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
