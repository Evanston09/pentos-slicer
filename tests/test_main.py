import asyncio
from types import SimpleNamespace

import main


class FakeServer:
    def on_client_connect(self, callback):
        self.connect = callback
        return callback

    def on_client_disconnect(self, callback):
        self.disconnect = callback
        return callback


class FakeApp:
    def __init__(self, client, workspace) -> None:
        self.client = client
        self.workspace = workspace
        self.state = object()
        self.shown = False
        self.closed = False

    def show_setup(self) -> None:
        self.shown = True

    def close(self) -> None:
        self.closed = True


def test_client_connections_have_independent_apps(monkeypatch) -> None:
    themed_clients = []
    scene_clients = []
    monkeypatch.setattr(main, "AppController", FakeApp)
    monkeypatch.setattr(main, "configure_theme", themed_clients.append)
    monkeypatch.setattr(main, "add_build_plate_scene", scene_clients.append)
    server = FakeServer()
    sessions = main.register_client_sessions(server)
    first_client = SimpleNamespace(client_id=1)
    second_client = SimpleNamespace(client_id=2)

    asyncio.run(server.connect(first_client))
    asyncio.run(server.connect(second_client))

    assert sessions[1] is not sessions[2]
    assert sessions[1].app.state is not sessions[2].app.state
    assert sessions[1].app.workspace != sessions[2].app.workspace
    assert sessions[1].app.workspace.is_dir()
    assert sessions[2].app.workspace.is_dir()
    assert sessions[1].app.shown
    assert sessions[2].app.shown
    assert themed_clients == [first_client, second_client]
    assert scene_clients == [first_client, second_client]

    asyncio.run(server.disconnect(first_client))
    asyncio.run(server.disconnect(second_client))


def test_disconnect_closes_only_matching_client_app(monkeypatch) -> None:
    monkeypatch.setattr(main, "AppController", FakeApp)
    monkeypatch.setattr(main, "configure_theme", lambda client: None)
    monkeypatch.setattr(main, "add_build_plate_scene", lambda client: None)
    server = FakeServer()
    sessions = main.register_client_sessions(server)
    first_client = SimpleNamespace(client_id=1)
    second_client = SimpleNamespace(client_id=2)
    asyncio.run(server.connect(first_client))
    asyncio.run(server.connect(second_client))
    first_session = sessions[1]
    second_session = sessions[2]
    first_workspace = first_session.app.workspace
    second_workspace = second_session.app.workspace
    (first_workspace / "uploaded.stl").write_bytes(b"model")

    asyncio.run(server.disconnect(first_client))

    assert first_session.app.closed
    assert not second_session.app.closed
    assert not first_workspace.exists()
    assert second_workspace.exists()
    assert sessions == {2: second_session}

    asyncio.run(server.disconnect(second_client))
