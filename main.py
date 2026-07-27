from dataclasses import dataclass
from pathlib import Path
import tempfile

import viser

from controllers.app_controller import AppController
from views.theming import add_build_plate_scene, configure_theme


@dataclass
class ClientSession:
    app: AppController
    temporary_directory: tempfile.TemporaryDirectory[str]

    def close(self) -> None:
        try:
            self.app.close()
        finally:
            self.temporary_directory.cleanup()


def register_client_sessions(
    server: viser.ViserServer,
) -> dict[int, ClientSession]:
    runtime_dir = Path(tempfile.gettempdir()) / "pentos-slicer"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sessions: dict[int, ClientSession] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        configure_theme(client)
        add_build_plate_scene(client)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix=f"{client.client_id}-",
            dir=runtime_dir,
        )
        app = AppController(client, Path(temporary_directory.name))
        session = ClientSession(app, temporary_directory)
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
