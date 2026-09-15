from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from gcode_tools import format_print_time, parse_estimated_print_time
from models import AppState, GcodePreview, MachineConfig
from models.printer import PrinterModel
from services.gcode_preview import parse_gcode_preview
from services.printer_model import load_printer_model


class PreviewViewPort(Protocol):
    def mount(self, gcode_path: Path | None) -> None: ...

    def unmount(self) -> None: ...

    def set_status(self, message: str) -> None: ...

    def set_estimated_time(self, estimate: str) -> None: ...

    def show_preview(
        self, preview: GcodePreview, machine_config: MachineConfig
    ) -> None: ...


class PreviewController:
    def __init__(
        self,
        state: AppState,
        view: PreviewViewPort,
        show_setup: Callable[[], None],
    ) -> None:
        self.state = state
        self.view = view
        self._show_setup = show_setup

    def mount(self) -> None:
        self.view.mount(self.state.gcode_path)
        self.load_preview()

    def unmount(self) -> None:
        self.view.unmount()

    def load_preview(self) -> None:
        if self.state.gcode_path is None:
            self.view.set_status("No G-code generated")
            return

        try:
            text = self.state.gcode_path.read_text()
            preview = parse_gcode_preview(text, self.state.machine_config)
        except Exception as exc:
            self.view.set_status(f"Failed to preview G-code: {exc}")
            return

        self.view.show_preview(preview, self.state.machine_config)
        visible_steps = [
            step
            for step in preview.simulation_steps
            if step.preview_segment is not None
        ]
        counts = Counter(step.kind for step in visible_steps)
        part_count = len(
            {step.part_index for step in visible_steps if step.part_index is not None}
        )
        estimate = parse_estimated_print_time(text)
        if estimate is not None:
            self.view.set_estimated_time(format_print_time(estimate))
        self.view.set_status(
            f"Preview: {part_count} parts, "
            f"{counts['extrusion']} extrusion, "
            f"{counts['travel']} travel, "
            f"{counts['setup']} setup"
        )

    def load_printer_model(self) -> PrinterModel:
        return load_printer_model(self.state.machine_config)

    def download_gcode(self) -> tuple[str, bytes] | None:
        if self.state.gcode_path is None:
            self.view.set_status("No G-code generated")
            return None

        try:
            content = self.state.gcode_path.read_bytes()
        except Exception as exc:
            self.view.set_status(f"Failed to download G-code: {exc}")
            return None

        filename = self.state.gcode_path.name
        self.view.set_status(f"Downloading {filename}")
        return filename, content

    def show_setup(self) -> None:
        self._show_setup()
