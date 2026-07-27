import viser

from controllers.app_controller import AppController
from views.theming import add_build_plate_scene, configure_theme


def register_client_sessions(
    server: viser.ViserServer,
) -> dict[int, AppController]:
    sessions: dict[int, AppController] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        configure_theme(client)
        add_build_plate_scene(client)
        app = AppController(client)
        sessions[client.client_id] = app
        app.show_setup()

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle) -> None:
        app = sessions.pop(client.client_id, None)
        if app is not None:
            app.close()

    return sessions


def main() -> None:
    server = viser.ViserServer(label="Pentos")
    register_client_sessions(server)

    print(f"Open your browser to http://localhost:{server.get_port()}")
    print("Press Ctrl+C to exit")
    server.sleep_forever()


if __name__ == "__main__":
    main()
