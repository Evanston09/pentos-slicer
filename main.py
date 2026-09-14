import asyncio
import os
from dataclasses import dataclass
from threading import BoundedSemaphore

import viser

from controllers.app_controller import MACHINE_CONFIG_STORAGE_KEY, AppController
from models import DEFAULT_MACHINE_CONFIG, MachineConfig, SlicingSettings
from services.machine_config_io import load_machine_config
from services.session_workspace import SessionWorkspace
from services.slicing_config import SLICING_SETTINGS_STORAGE_KEY, load_slicing_settings
from views.theming import add_build_plate_scene, configure_theme


@dataclass
class ClientSession:
    app: AppController
    workspace: SessionWorkspace

    def close(self) -> None:
        try:
            self.app.close()
        finally:
            self.workspace.close()


def max_concurrent_slices() -> int:
    value = int(os.getenv("MAX_CONCURRENT_SLICES", "2"))
    if value < 1:
        raise ValueError("MAX_CONCURRENT_SLICES must be at least 1")
    return value


def load_client_machine_config(client: viser.ClientHandle) -> MachineConfig:
    try:
        value = client.local_storage.get_item(MACHINE_CONFIG_STORAGE_KEY)
        if value is None:
            return DEFAULT_MACHINE_CONFIG
        return load_machine_config(value.encode())
    except (RuntimeError, TimeoutError, ValueError) as exc:
        client.local_storage.remove_item(MACHINE_CONFIG_STORAGE_KEY)
        print(f"Failed to load saved machine configuration: {exc}")
        client.add_notification(
            "Error", f"Failed to load saved machine configuration: {exc}"
        )
        return DEFAULT_MACHINE_CONFIG


def load_client_slicing_settings(client: viser.ClientHandle) -> SlicingSettings:
    try:
        value = client.local_storage.get_item(SLICING_SETTINGS_STORAGE_KEY)
        return SlicingSettings() if value is None else load_slicing_settings(value)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        client.local_storage.remove_item(SLICING_SETTINGS_STORAGE_KEY)
        client.add_notification(
            "Error", f"Failed to load saved slicing settings: {exc}"
        )
        return SlicingSettings()


def register_client_sessions(
    server: viser.ViserServer,
) -> dict[int, ClientSession]:
    slicing_slots = BoundedSemaphore(max_concurrent_slices())
    sessions: dict[int, ClientSession] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        configure_theme(client)
        machine_config, slicing_settings = await asyncio.gather(
            asyncio.to_thread(load_client_machine_config, client),
            asyncio.to_thread(load_client_slicing_settings, client),
        )
        add_build_plate_scene(client, machine_config)
        workspace = SessionWorkspace(client.client_id)
        try:
            app = AppController(
                client, workspace, slicing_slots, machine_config, slicing_settings
            )
        except Exception:
            workspace.close()
            raise
        session = ClientSession(app, workspace)
        sessions[client.client_id] = session
        try:
            app.show_setup()
        except Exception:
            sessions.pop(client.client_id, None)
            session.close()
            raise

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle) -> None:
        session = sessions.pop(client.client_id, None)
        if session is not None:
            session.close()

    return sessions


def main() -> None:
    server = viser.ViserServer(label="Pentos")
    register_client_sessions(server)

    print(f"Open your browser to http://localhost:{server.get_port()}")
    print("Press Ctrl+C to exit")
    server.sleep_forever()


if __name__ == "__main__":
    main()
