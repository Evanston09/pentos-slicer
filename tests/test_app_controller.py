from controllers.app_controller import AppController


class FakeController:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def mount(self) -> None:
        self.events.append(f"mount {self.name}")

    def unmount(self) -> None:
        self.events.append(f"unmount {self.name}")


def test_navigation_mounts_each_screen_once_and_unmounts_previous() -> None:
    events = []
    app = AppController.__new__(AppController)
    app.setup_controller = FakeController(events, "setup")
    app.preview_controller = FakeController(events, "preview")
    app.active_controller = None

    app.show_setup()
    app.show_setup()
    app.show_preview()
    app.show_preview()

    assert events == [
        "mount setup",
        "unmount setup",
        "mount preview",
    ]


def test_close_unmounts_active_controller() -> None:
    events = []
    app = AppController.__new__(AppController)
    active = FakeController(events, "active")
    app.active_controller = active

    app.close()
    app.close()

    assert events == ["unmount active"]
    assert app.active_controller is None


def test_app_uses_workspace_for_all_runtime_files(tmp_path) -> None:
    app = AppController(object(), tmp_path)

    assert app.setup_controller.upload_dir == tmp_path / "uploads"
    assert app.slicer.temp_dir == tmp_path / "temp"
    assert app.slicer.out_dir == tmp_path / "output"
    assert app.slicer.temp_dir.is_dir()
    assert app.slicer.out_dir.is_dir()
