from pathlib import Path
from types import SimpleNamespace

from views.preview_view import PreviewView


class FakeHandle:
    def __init__(self, value=None) -> None:
        self.value = value
        self.click_callback = None

    def on_update(self, callback):
        return callback

    def on_click(self, callback):
        self.click_callback = callback
        return callback

    def remove(self) -> None:
        pass


class FakeGui:
    def __init__(self) -> None:
        self.buttons = {}
        self.text = {}

    def add_text(self, label, value, **kwargs):
        handle = FakeHandle(value)
        self.text[label] = handle
        return handle

    def add_checkbox(self, label, value):
        return FakeHandle(value)

    def add_number(self, label, value, **kwargs):
        return FakeHandle(value)

    def add_button(self, label, **kwargs):
        handle = FakeHandle()
        self.buttons[label] = handle
        return handle


class FakeController:
    def download_gcode(self):
        return "model.gcode", b"G90\n"

    def show_setup(self) -> None:
        pass


class FakeDownloadClient:
    def __init__(self) -> None:
        self.downloads = []

    def send_file_download(self, filename, content, save_immediately) -> None:
        self.downloads.append((filename, content, save_immediately))


def test_download_button_sends_gcode_to_initiating_client() -> None:
    gui = FakeGui()
    view = PreviewView(SimpleNamespace(gui=gui, scene=None))
    view.bind_controller(FakeController())
    view.mount(Path("/tmp/private-session/model.gcode"))
    download_client = FakeDownloadClient()

    gui.buttons["Download G-code"].click_callback(
        SimpleNamespace(client=download_client)
    )

    assert download_client.downloads == [("model.gcode", b"G90\n", True)]
    assert gui.text["Output G-code"].value == "model.gcode"
