from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from models import AppState, GcodePreview
from services.gcode_preview import parse_gcode_preview


class PreviewViewPort(Protocol):
    def mount(self, gcode_path: Path | None) -> None: ...

    def unmount(self) -> None: ...

    def set_status(self, message: str) -> None: ...

    def show_preview(self, preview: GcodePreview) -> None: ...


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
            preview = parse_gcode_preview(text)
        except Exception as exc:
            self.view.set_status(f"Failed to preview G-code: {exc}")
            return

        self.view.show_preview(preview)
        extrusion_count = sum(len(part.extrusion) for part in preview.parts)
        travel_count = sum(len(part.travel) for part in preview.parts)
        self.view.set_status(
            f"Preview: {len(preview.parts)} parts, "
            f"{extrusion_count} extrusion, "
            f"{travel_count} travel, "
            f"{len(preview.setup)} setup"
        )

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
