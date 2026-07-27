from typing import Any, Protocol

from controllers.preview_controller import PreviewController
from controllers.setup_controller import SetupController
from machine import BUILD_PLATE_CENTER
from models import AppState
from services.slicing import Slicer
from views import PreviewView, SetupView


class SceneController(Protocol):
    def mount(self) -> None: ...

    def unmount(self) -> None: ...


class AppController:
    def __init__(self, client: Any) -> None:
        self.state = AppState(model_xy_position=BUILD_PLATE_CENTER[:2])
        self.slicer = Slicer()
        self.setup_view = SetupView(client)
        self.preview_view = PreviewView(client)
        self.setup_controller = SetupController(
            self.state,
            self.slicer,
            self.setup_view,
            self.show_preview,
        )
        self.preview_controller = PreviewController(
            self.state,
            self.preview_view,
            self.show_setup,
        )
        self.setup_view.bind_controller(self.setup_controller)
        self.preview_view.bind_controller(self.preview_controller)
        self.active_controller: SceneController | None = None

    def show_setup(self) -> None:
        if self.active_controller is self.setup_controller:
            return

        if self.active_controller is not None:
            self.active_controller.unmount()

        self.setup_controller.mount()
        self.active_controller = self.setup_controller

    def show_preview(self) -> None:
        if self.active_controller is self.preview_controller:
            return

        if self.active_controller is not None:
            self.active_controller.unmount()

        self.preview_controller.mount()
        self.active_controller = self.preview_controller

    def close(self) -> None:
        if self.active_controller is not None:
            self.active_controller.unmount()
            self.active_controller = None
