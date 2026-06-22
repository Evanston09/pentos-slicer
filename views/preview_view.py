from typing import Any, Callable

import viser

from app_state import AppState


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
        self.back_button: Any | None = None

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
        self.back_button = self.server.gui.add_button("Back to Setup")

        @self.back_button.on_click
        def _(_) -> None:
            self.show_setup()

    def unmount(self) -> None:
        for handle in (self.back_button, self.output_path, self.status):
            if handle is not None:
                handle.remove()

        self.status = None
        self.output_path = None
        self.back_button = None
