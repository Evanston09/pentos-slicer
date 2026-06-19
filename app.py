from typing import Protocol

import viser

from app_state import AppState
from slice_tools import Slicer
from views import PreviewView, SetupView


class SceneView(Protocol):
    def mount(self) -> None: ...

    def unmount(self) -> None: ...


class PentosApp:
    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        self.state = AppState()
        self.slicer = Slicer()
        self.setup_view = SetupView(
            server,
            self.state,
            self.slicer,
            self.show_preview,
        )
        self.preview_view = PreviewView(
            server,
            self.state,
            self.show_setup,
        )
        self.active_view: SceneView | None = None

    def show_setup(self) -> None:
        if self.active_view is self.setup_view:
            return

        if self.active_view is not None:
            self.active_view.unmount()

        self.setup_view.mount()
        self.active_view = self.setup_view

    def show_preview(self) -> None:
        if self.active_view is self.preview_view:
            return

        if self.active_view is not None:
            self.active_view.unmount()

        self.preview_view.mount()
        self.active_view = self.preview_view
