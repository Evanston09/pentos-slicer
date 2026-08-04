import os
from dataclasses import dataclass
from threading import BoundedSemaphore

import viser

from controllers.app_controller import AppController
from services.session_workspace import SessionWorkspace
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


def register_client_sessions(
    server: viser.ViserServer,
) -> dict[int, ClientSession]:
    slicing_slots = BoundedSemaphore(max_concurrent_slices())
    sessions: dict[int, ClientSession] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        configure_theme(client)
        add_build_plate_scene(client)
        workspace = SessionWorkspace(client.client_id)
        try:
            app = AppController(client, workspace, slicing_slots)
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
