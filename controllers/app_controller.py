from threading import BoundedSemaphore
from typing import Any, Protocol

from controllers.preview_controller import PreviewController
from controllers.setup_controller import SetupController
from models import AppState, MachineConfig, SlicingSettings
from services.machine_config_io import save_machine_config
from services.session_workspace import SessionWorkspace
from services.slicing import Slicer
from services.slicing_config import SLICING_SETTINGS_STORAGE_KEY, save_slicing_settings
from views import PreviewView, SetupView

MACHINE_CONFIG_STORAGE_KEY = "pentos-machine-config"


class SceneController(Protocol):
    def mount(self) -> None: ...

    def unmount(self) -> None: ...


class AppController:
    def __init__(
        self,
        client: Any,
        workspace: SessionWorkspace,
        slicing_slots: BoundedSemaphore,
        machine_config: MachineConfig,
        slicing_settings: SlicingSettings,
    ) -> None:
        self.state = AppState(
            slicing_settings=slicing_settings,
            machine_config=machine_config,
            model_xy_position=machine_config.build_plate_center[:2],
        )
        self.slicer = Slicer(
            out_dir=workspace.path / "output",
            temp_dir=workspace.path / "temp",
            machine_config=self.state.machine_config,
        )
        self.setup_view = SetupView(client)
        self.preview_view = PreviewView(client)
        self.setup_controller = SetupController(
            self.state,
            self.slicer,
            self.setup_view,
            self.show_preview,
            workspace,
            slicing_slots,
            lambda config: client.local_storage.set_item(
                MACHINE_CONFIG_STORAGE_KEY,
                save_machine_config(config).decode(),
            ),
            lambda settings: client.local_storage.set_item(
                SLICING_SETTINGS_STORAGE_KEY, save_slicing_settings(settings)
            ),
        )
        self.preview_controller = PreviewController(
            self.state,
            self.preview_view,
            self.show_setup,
        )
        self.setup_view.bind_controller(self.setup_controller)
        self.preview_view.bind_controller(self.preview_controller)
        self.active_controller: SceneController | None = None
        self.closed = False

    def show_setup(self) -> None:
        if self.closed:
            return
        if self.active_controller is self.setup_controller:
            return

        if self.active_controller is not None:
            self.active_controller.unmount()

        self.setup_controller.mount()
        self.active_controller = self.setup_controller

    def show_preview(self) -> None:
        if self.closed:
            return
        if self.active_controller is self.preview_controller:
            return

        if self.active_controller is not None:
            self.active_controller.unmount()

        self.preview_controller.mount()
        self.active_controller = self.preview_controller

    def close(self) -> None:
        self.closed = True
        if self.active_controller is not None:
            self.active_controller.unmount()
            self.active_controller = None
