from controllers.preview_controller import PreviewController
from models import AppState


class FakePreviewView:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.preview = None
        self.path = None

    def mount(self, gcode_path) -> None:
        self.path = gcode_path

    def unmount(self) -> None:
        pass

    def set_status(self, message: str) -> None:
        self.statuses.append(message)

    def show_preview(self, preview) -> None:
        self.preview = preview


def test_missing_gcode_sets_status() -> None:
    view = FakePreviewView()
    controller = PreviewController(AppState(), view, lambda: None)

    controller.load_preview()

    assert view.statuses[-1] == "No G-code generated"


def test_load_preview_reads_and_parses_gcode(tmp_path) -> None:
    path = tmp_path / "preview.gcode"
    path.write_text("G90\nM83\n;LAYER_CHANGE\nG1 X68 Y7 Z1\nG1 X69 Y7 Z1 E0.5\n")
    view = FakePreviewView()
    controller = PreviewController(
        AppState(gcode_path=path),
        view,
        lambda: None,
    )

    controller.load_preview()

    assert view.preview is not None
    assert len(view.preview.parts) == 1
    assert view.statuses[-1].startswith("Preview: 1 parts")


def test_download_returns_gcode_and_reports_status(tmp_path) -> None:
    path = tmp_path / "preview.gcode"
    path.write_bytes(b"G90\n")

    view = FakePreviewView()
    controller = PreviewController(
        AppState(gcode_path=path),
        view,
        lambda: None,
    )

    download = controller.download_gcode()

    assert download == ("preview.gcode", b"G90\n")
    assert view.statuses[-1] == "Downloading preview.gcode"


def test_download_failure_reports_error(tmp_path) -> None:
    path = tmp_path / "preview.gcode"

    view = FakePreviewView()
    controller = PreviewController(
        AppState(gcode_path=path),
        view,
        lambda: None,
    )

    download = controller.download_gcode()

    assert download is None
    assert view.statuses[-1].startswith("Failed to download G-code:")
