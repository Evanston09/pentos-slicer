from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import viser

from models import GcodePreview
from views.printer_simulation import PrinterSimulation
from views.theming import PENTOS_ORANGE

if TYPE_CHECKING:
    from controllers.preview_controller import PreviewController

SETUP_COLOR = PENTOS_ORANGE
PART_COLORS = [
    (47, 153, 238),
    (255, 130, 0),
    (34, 197, 94),
    (236, 72, 153),
    (168, 85, 247),
    (20, 184, 166),
]


@dataclass(frozen=True)
class PreviewControls:
    status: viser.GuiTextHandle
    estimated_time: viser.GuiTextHandle
    output_path: viser.GuiTextHandle
    show_travel: viser.GuiCheckboxHandle
    line_width: viser.GuiNumberHandle[float]
    simulation: viser.GuiCheckboxHandle
    move: viser.GuiSliderHandle[int]
    back_button: viser.GuiButtonHandle
    download_button: viser.GuiButtonHandle


class PreviewView:
    def __init__(self, client: viser.ClientHandle) -> None:
        self.client = client
        self.controller: PreviewController
        self.controls: PreviewControls | None = None
        self.preview: GcodePreview | None = None
        self.printer: PrinterSimulation | None = None
        self.travel_handle: viser.LineSegmentsHandle | None = None
        self.extrusion_handle: viser.LineSegmentsHandle | None = None

    def bind_controller(self, controller: PreviewController) -> None:
        self.controller = controller

    def mount(self, gcode_path: Path | None) -> None:
        status = self.client.gui.add_text(
            "Status",
            "Saved G-code",
            disabled=True,
        )
        estimated_time = self.client.gui.add_text(
            "Estimated time",
            "",
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
        simulation = self.client.gui.add_checkbox("Simulation", False)
        move = self.client.gui.add_slider(
            "Simulation move",
            min=0,
            max=0,
            step=1,
            initial_value=0,
            disabled=True,
            visible=False,
        )
        download_button = self.client.gui.add_button(
            "Download G-code",
            icon=viser.Icon.DOWNLOAD,
        )
        back_button = self.client.gui.add_button("Back to Setup")
        controls = PreviewControls(
            status=status,
            estimated_time=estimated_time,
            output_path=output_path,
            show_travel=show_travel,
            line_width=line_width,
            simulation=simulation,
            move=move,
            back_button=back_button,
            download_button=download_button,
        )
        self.controls = controls

        @controls.show_travel.on_update
        def _(_) -> None:
            if self.travel_handle is not None:
                self.travel_handle.visible = controls.show_travel.value

        @controls.line_width.on_update
        def _(_) -> None:
            width = controls.line_width.value
            if self.travel_handle is not None:
                self.travel_handle.line_width = max(1.0, width * 0.5)
            if self.extrusion_handle is not None:
                self.extrusion_handle.line_width = width

        @controls.simulation.on_update
        def _(_) -> None:
            self._update_simulation()

        @controls.move.on_update
        def _(_) -> None:
            self._show_step(controls.move.value)

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

    def set_estimated_time(self, estimate: str) -> None:
        self._mounted().estimated_time.value = estimate

    def show_preview(self, preview: GcodePreview) -> None:
        controls = self._mounted()
        line_width = controls.line_width.value
        travel_visible = controls.show_travel.value
        empty_points = np.empty((0, 2, 3), dtype=np.float32)
        empty_colors = np.empty((0, 2, 3), dtype=np.uint8)
        self.preview = preview
        self.printer = PrinterSimulation(
            self.client, self.controller.state.machine_config
        )
        self.travel_handle = self.client.scene.add_line_segments(
            "/preview/toolpath/travel",
            points=empty_points,
            colors=empty_colors,
            line_width=max(1.0, line_width * 0.5),
            visible=travel_visible,
        )
        self.extrusion_handle = self.client.scene.add_line_segments(
            "/preview/toolpath/extrusion",
            points=empty_points,
            colors=empty_colors,
            line_width=line_width,
        )

        if preview.simulation_steps:
            controls.move.max = len(preview.simulation_steps) - 1
            controls.move.disabled = False
            self._update_simulation()

    def unmount(self) -> None:
        if self.controls is None:
            return

        controls = self.controls
        for handle in (self.travel_handle, self.extrusion_handle):
            if handle is not None:
                handle.remove()
        for handle in (
            controls.back_button,
            controls.move,
            controls.simulation,
            controls.line_width,
            controls.show_travel,
            controls.download_button,
            controls.output_path,
            controls.estimated_time,
            controls.status,
        ):
            handle.remove()
        if self.printer is not None:
            self.printer.remove()

        self.controls = None
        self.preview = None
        self.printer = None
        self.travel_handle = None
        self.extrusion_handle = None

    def _update_simulation(self) -> None:
        controls = self._mounted()
        controls.move.visible = controls.simulation.value
        if self.printer is not None:
            self.printer.root.visible = controls.simulation.value
        if self.preview is not None and self.preview.simulation_steps:
            index = (
                controls.move.value
                if controls.simulation.value
                else len(self.preview.simulation_steps) - 1
            )
            self._show_step(index)
            if not controls.simulation.value:
                self.printer.reset_bed_pose()

    def _show_step(self, index: int) -> None:
        if (
            self.preview is None
            or self.printer is None
            or self.travel_handle is None
            or self.extrusion_handle is None
        ):
            return

        steps = self.preview.simulation_steps[: index + 1]
        travel_points = []
        travel_colors = []
        extrusion_points = []
        extrusion_colors = []
        for step in steps:
            if step.preview_segment is None:
                continue
            if step.kind == "setup":
                color = SETUP_COLOR
            else:
                assert step.part_index is not None
                color = PART_COLORS[step.part_index % len(PART_COLORS)]

            if step.kind == "extrusion":
                extrusion_points.append(step.preview_segment)
                extrusion_colors.append([color, color])
            elif step.kind in {"setup", "travel"}:
                travel_points.append(step.preview_segment)
                travel_colors.append([color, color])

        self.travel_handle.points = np.asarray(travel_points, dtype=np.float32).reshape(
            -1, 2, 3
        )
        self.travel_handle.colors = np.asarray(travel_colors, dtype=np.uint8).reshape(
            -1, 2, 3
        )
        self.extrusion_handle.points = np.asarray(
            extrusion_points, dtype=np.float32
        ).reshape(-1, 2, 3)
        self.extrusion_handle.colors = np.asarray(
            extrusion_colors, dtype=np.uint8
        ).reshape(-1, 2, 3)
        self.printer.set_pose(steps[-1].pose)

    def _mounted(self) -> PreviewControls:
        if self.controls is None:
            raise RuntimeError("Preview view is not mounted")
        return self.controls
