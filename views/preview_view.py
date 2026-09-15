from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import viser

from models import GcodePreview, MachineConfig
from views.printer_simulation import PrinterSimulation
from views.theming import PART_COLORS, PENTOS_ORANGE

if TYPE_CHECKING:
    from controllers.preview_controller import PreviewController


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


@dataclass(frozen=True)
class ToolpathDrawing:
    handle: viser.LineSegmentsHandle
    points: np.ndarray
    colors: np.ndarray
    segment_counts: np.ndarray


class PreviewView:
    def __init__(self, client: viser.ClientHandle) -> None:
        self.client = client
        self.controller: PreviewController
        self.controls: PreviewControls | None = None
        self.preview: GcodePreview | None = None
        self.machine_config: MachineConfig | None = None
        self.toolpaths: list[ToolpathDrawing] = []
        self.printer: PrinterSimulation | None = None
        self.travel_handle: viser.LineSegmentsHandle | None = None
        self.extrusion_handle: viser.LineSegmentsHandle | None = None
        self.rotary_plot: viser.GuiUplotHandle | None = None

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

    def show_preview(
        self, preview: GcodePreview, machine_config: MachineConfig
    ) -> None:
        controls = self._mounted()
        line_width = controls.line_width.value
        travel_visible = controls.show_travel.value
        empty_points = np.empty((0, 2, 3))
        empty_colors = np.empty((0, 2, 3))
        self.preview = preview
        self.machine_config = machine_config
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

        for handle, kinds in (
            (self.travel_handle, {"setup", "travel"}),
            (self.extrusion_handle, {"extrusion"}),
        ):
            visible = [
                step.preview_segment is not None and step.kind in kinds
                for step in preview.simulation_steps
            ]
            segments = [
                step
                for step, included in zip(preview.simulation_steps, visible)
                if included
            ]
            points = np.asarray(
                [step.preview_segment for step in segments], dtype=np.float32
            ).reshape(-1, 2, 3)
            colors = np.asarray(
                [
                    [
                        PENTOS_ORANGE
                        if step.kind == "setup"
                        else PART_COLORS[step.part_index % len(PART_COLORS)]
                    ]
                    * 2
                    for step in segments
                ],
            ).reshape(-1, 2, 3)
            self.toolpaths.append(
                ToolpathDrawing(
                    handle=handle,
                    points=points,
                    colors=colors,
                    segment_counts=np.cumsum(visible),
                )
            )

        controls.simulation.disabled = not preview.simulation_steps
        controls.move.max = max(0, len(preview.simulation_steps) - 1)
        controls.move.disabled = not preview.simulation_steps
        self._update_simulation()

        if len(preview.motion_time_seconds):
            self.rotary_plot = self.client.gui.add_uplot(
                data=(
                    preview.motion_time_seconds,
                    preview.a_degrees,
                    preview.b_degrees,
                ),
                series=(
                    {},
                    {"label": "A", "stroke": "#f58214", "width": 2.0},
                    {"label": "B", "stroke": "#2f99ee", "width": 2.0},
                ),
                title="A/B movement (click legend to toggle)",
                scales={"x": {"time": False}},
                axes=(
                    {"label": "Nominal motion time (s)"},
                    {"label": "Commanded angle (degrees)"},
                ),
                height=260,
            )

    def unmount(self) -> None:
        if self.controls is None:
            return

        controls = self.controls
        for handle in (self.travel_handle, self.extrusion_handle):
            if handle is not None:
                handle.remove()
        if self.rotary_plot is not None:
            self.rotary_plot.remove()
            self.rotary_plot = None
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
        self.machine_config = None
        self.toolpaths = []
        self.printer = None
        self.travel_handle = None
        self.extrusion_handle = None

    def _update_simulation(self) -> None:
        controls = self._mounted()
        enabled = controls.simulation.value
        if enabled and self.preview is not None and self.preview.simulation_steps:
            if self.printer is None:
                assert self.machine_config is not None
                self.printer = PrinterSimulation(
                    self.client,
                    self.machine_config,
                    self.controller.load_printer_model(),
                )
        controls.move.visible = enabled
        if self.printer is not None:
            self.printer.root.visible = enabled
            if not enabled:
                self.printer.reset_bed_pose()
        if self.preview is not None and self.preview.simulation_steps:
            self._show_step(
                controls.move.value
                if enabled
                else len(self.preview.simulation_steps) - 1
            )

    def _show_step(self, index: int) -> None:
        for toolpath in self.toolpaths:
            count = toolpath.segment_counts[index]
            toolpath.handle.points = toolpath.points[:count]
            toolpath.handle.colors = toolpath.colors[:count]
        if self._mounted().simulation.value and self.printer is not None:
            self.printer.set_pose(self.preview.simulation_steps[index].pose)

    def _mounted(self) -> PreviewControls:
        if self.controls is None:
            raise RuntimeError("Preview view is not mounted")
        return self.controls
