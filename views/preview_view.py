from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import viser

from models import GcodePreview
from views.theming import PENTOS_ORANGE

if TYPE_CHECKING:
    from controllers.preview_controller import PreviewController

SETUP_COLOR = PENTOS_ORANGE


@dataclass(frozen=True)
class PreviewControls:
    status: viser.GuiTextHandle
    output_path: viser.GuiTextHandle
    show_travel: viser.GuiCheckboxHandle
    line_width: viser.GuiNumberHandle[float]
    back_button: viser.GuiButtonHandle
    download_button: viser.GuiButtonHandle


class PreviewView:
    def __init__(self, client: viser.ClientHandle) -> None:
        self.client = client
        self.controller: PreviewController
        self.controls: PreviewControls | None = None
        self.travel_handles: list[viser.LineSegmentsHandle] = []
        self.extrusion_handles: list[viser.LineSegmentsHandle] = []

    def bind_controller(self, controller: PreviewController) -> None:
        self.controller = controller

    def mount(self, gcode_path: Path | None) -> None:
        status = self.client.gui.add_text(
            "Status",
            "Saved G-code",
            disabled=True,
        )
        output_path = self.client.gui.add_text(
            "Output G-code",
            "" if gcode_path is None else gcode_path.name,
            disabled=True,
        )
        show_travel = self.client.gui.add_checkbox("Travel", True)
        line_width = self.client.gui.add_number(
            "Line width",
            2.0,
            min=1.0,
            max=10.0,
        )
        download_button = self.client.gui.add_button(
            "Download G-code",
            icon=viser.Icon.DOWNLOAD,
        )
        back_button = self.client.gui.add_button("Back to Setup")
        controls = PreviewControls(
            status=status,
            output_path=output_path,
            show_travel=show_travel,
            line_width=line_width,
            back_button=back_button,
            download_button=download_button,
        )
        self.controls = controls

        @controls.show_travel.on_update
        def _(_) -> None:
            visible = controls.show_travel.value
            for travel_handle in self.travel_handles:
                travel_handle.visible = visible

        @controls.line_width.on_update
        def _(_) -> None:
            width = controls.line_width.value
            for extrusion_handle in self.extrusion_handles:
                extrusion_handle.line_width = width
            for travel_handle in self.travel_handles:
                travel_handle.line_width = max(1.0, width * 0.5)

        @controls.back_button.on_click
        def _(_) -> None:
            self.controller.show_setup()

        @controls.download_button.on_click
        def _(event) -> None:
            download = self.controller.download_gcode()
            if download is None:
                return
            filename, content = download
            assert event.client is not None
            event.client.send_file_download(
                filename,
                content,
                save_immediately=True,
            )

    def set_status(self, message: str) -> None:
        self._mounted().status.value = message

    def show_preview(self, preview: GcodePreview) -> None:
        controls = self._mounted()
        line_width = controls.line_width.value
        travel_visible = controls.show_travel.value

        if len(preview.setup):
            self.travel_handles.append(
                self.client.scene.add_line_segments(
                    "/preview/setup",
                    points=preview.setup,
                    colors=SETUP_COLOR,
                    line_width=max(1.0, line_width * 0.5),
                    visible=travel_visible,
                )
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
        if self.controls is None:
            return

        controls = self.controls
        for handle in (
            *self.extrusion_handles,
            *self.travel_handles,
            controls.back_button,
            controls.line_width,
            controls.show_travel,
            controls.download_button,
            controls.output_path,
            controls.status,
        ):
            handle.remove()

        self.controls = None
        self.travel_handles = []
        self.extrusion_handles = []

    def _mounted(self) -> PreviewControls:
        if self.controls is None:
            raise RuntimeError("Preview view is not mounted")
        return self.controls
