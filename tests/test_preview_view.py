from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal

from models import DEFAULT_MACHINE_CONFIG, GcodePreview
from services.gcode_preview import parse_gcode_preview
from views.preview_view import PreviewView
from views.theming import PART_COLORS, PENTOS_ORANGE


class FakeHandle:
    def __init__(self, value=None, **kwargs) -> None:
        self.value = value
        self.click_callback = None
        self.update_callback = None
        for name, setting in kwargs.items():
            setattr(self, name, setting)

    def on_update(self, callback):
        self.update_callback = callback
        return callback

    def on_click(self, callback):
        self.click_callback = callback
        return callback

    def remove(self) -> None:
        pass


class FakeGui:
    def __init__(self) -> None:
        self.buttons = {}
        self.checkboxes = {}
        self.sliders = {}
        self.text = {}
        self.uplot = None

    def add_text(self, label, value, **kwargs):
        handle = FakeHandle(value)
        self.text[label] = handle
        return handle

    def add_checkbox(self, label, value):
        handle = FakeHandle(value)
        self.checkboxes[label] = handle
        return handle

    def add_number(self, label, value, **kwargs):
        return FakeHandle(value)

    def add_slider(self, label, initial_value, **kwargs):
        handle = FakeHandle(initial_value, **kwargs)
        self.sliders[label] = handle
        return handle

    def add_button(self, label, **kwargs):
        handle = FakeHandle()
        self.buttons[label] = handle
        return handle

    def add_uplot(self, **kwargs):
        self.uplot = kwargs
        return FakeHandle()


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


def test_simulation_loads_once_and_scrubs_prepared_paths(monkeypatch) -> None:
    gui = FakeGui()
    scene = SimpleNamespace(
        add_line_segments=lambda *args, **kwargs: FakeHandle(**kwargs)
    )
    view = PreviewView(SimpleNamespace(gui=gui, scene=scene))
    controller = FakeController()
    loaded = []
    controller.load_printer_model = lambda: loaded.append(True) or "printer model"
    view.bind_controller(controller)
    view.mount(None)
    poses, resets = [], []
    printer = SimpleNamespace(
        root=SimpleNamespace(visible=True),
        set_pose=poses.append,
        reset_bed_pose=lambda: resets.append(True),
        remove=lambda: None,
    )

    def make_printer(client, config, model):
        assert config is DEFAULT_MACHINE_CONFIG
        assert model == "printer model"
        return printer

    monkeypatch.setattr("views.preview_view.PrinterSimulation", make_printer)
    preview = parse_gcode_preview(
        "G90\nG1 X68 Y7 Z1\nG1 X69\n;LAYER_CHANGE\nG1 X70 E0.1\n"
        "; --- PENTOS A/B TRANSITION ---\n"
        "G1 Z15\nG1 A90 B0\nG1 X71 Y8\nG1 Z2\n"
        "; --- END PENTOS A/B TRANSITION ---\nG1 X72\nG1 X73 E0.2\n",
        DEFAULT_MACHINE_CONFIG,
    )
    view.show_preview(preview, DEFAULT_MACHINE_CONFIG)
    assert loaded == []
    assert poses == []
    assert view.travel_handle.points.shape == (2, 2, 3)
    assert view.extrusion_handle.points.shape == (2, 2, 3)
    assert_array_equal(
        view.travel_handle.colors, [[PENTOS_ORANGE] * 2, [PART_COLORS[1]] * 2]
    )
    assert_array_equal(
        view.extrusion_handle.colors, [[PART_COLORS[0]] * 2, [PART_COLORS[1]] * 2]
    )

    simulation = gui.checkboxes["Simulation"]
    move = gui.sliders["Simulation move"]
    simulation.value = True
    simulation.update_callback(None)
    assert loaded == [True]
    assert move.visible
    assert poses[-1] is preview.simulation_steps[0].pose
    prepared = [(toolpath.points, toolpath.colors) for toolpath in view.toolpaths]
    for index in list(range(len(preview.simulation_steps))) + [2, 0]:
        move.value = index
        move.update_callback(None)
        assert poses[-1] is preview.simulation_steps[index].pose
        for handle, kinds in [
            (view.travel_handle, {"setup", "travel"}),
            (view.extrusion_handle, {"extrusion"}),
        ]:
            expected = [
                s.preview_segment
                for s in preview.simulation_steps[: index + 1]
                if s.preview_segment is not None and s.kind in kinds
            ]
            assert_array_equal(handle.points, np.asarray(expected).reshape(-1, 2, 3))
        for toolpath, (old_points, old_colors) in zip(view.toolpaths, prepared):
            assert toolpath.points is old_points
            assert toolpath.colors is old_colors
    assert all(
        step.preview_segment is None
        for step in preview.simulation_steps
        if step.kind == "transition"
    )

    pose_count = len(poses)
    simulation.value = False
    simulation.update_callback(None)
    assert not move.visible
    assert not printer.root.visible
    assert resets == [True]
    assert len(poses) == pose_count
    assert view.extrusion_handle.points.shape == (2, 2, 3)
    simulation.value = True
    simulation.update_callback(None)
    assert loaded == [True]
    assert printer.root.visible
    view.unmount()
    assert view.toolpaths == []


def test_download_button_sends_gcode_to_initiating_client() -> None:
    gui = FakeGui()
    view = PreviewView(SimpleNamespace(gui=gui, scene=None))
    view.bind_controller(FakeController())
    view.mount(Path("/tmp/private-session/model.gcode"))
    view.set_estimated_time("1h 2m")
    download_client = FakeDownloadClient()

    gui.buttons["Download G-code"].click_callback(
        SimpleNamespace(client=download_client)
    )

    assert download_client.downloads == [("model.gcode", b"G90\n", True)]
    assert gui.text["Output G-code"].value == "model.gcode"
    assert gui.text["Estimated time"].value == "1h 2m"


def test_ab_plot_has_toggleable_a_and_b_series(monkeypatch) -> None:
    gui = FakeGui()
    view = PreviewView(SimpleNamespace(gui=gui, scene=None))
    view.bind_controller(FakeController())
    view.mount(Path("/tmp/private-session/model.gcode"))

    monkeypatch.setattr(
        "views.preview_view.PrinterSimulation", lambda *args: SimpleNamespace()
    )
    view.client.scene = SimpleNamespace(
        add_line_segments=lambda *args, **kwargs: FakeHandle(**kwargs)
    )
    view.show_preview(
        GcodePreview(
            simulation_steps=[],
            motion_time_seconds=np.asarray([0.0, 1.0]),
            a_degrees=np.asarray([0.0, 10.0]),
            b_degrees=np.asarray([0.0, 20.0]),
        ),
        DEFAULT_MACHINE_CONFIG,
    )

    assert gui.uplot is not None
    assert gui.uplot["series"][1]["label"] == "A"
    assert gui.uplot["series"][2]["label"] == "B"
    assert gui.uplot["scales"] == {"x": {"time": False}}
    assert "click legend to toggle" in gui.uplot["title"]
