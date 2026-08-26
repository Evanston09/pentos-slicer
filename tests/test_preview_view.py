from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal

from views.preview_view import PART_COLORS, SETUP_COLOR, PreviewView


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


def test_simulation_toggle_switches_between_playhead_and_full_print() -> None:
    gui = FakeGui()
    view = PreviewView(SimpleNamespace(gui=gui, scene=None))
    view.bind_controller(FakeController())
    view.mount(None)
    view.preview = SimpleNamespace(simulation_steps=[1, 2, 3])
    resets = []
    view.printer = SimpleNamespace(
        root=SimpleNamespace(visible=True), reset_bed_pose=lambda: resets.append(True)
    )
    shown = []
    view._show_step = shown.append

    simulation = gui.checkboxes["Simulation"]
    move = gui.sliders["Simulation move"]
    simulation.update_callback(None)
    assert not move.visible
    assert not view.printer.root.visible
    assert shown[-1] == 2
    assert resets == [True]

    simulation.value = True
    simulation.update_callback(None)
    assert move.visible
    assert view.printer.root.visible
    assert shown[-1] == 0


def test_show_step_splits_ordered_moves_by_kind() -> None:
    view = PreviewView(SimpleNamespace())
    view.travel_handle = FakeHandle()
    view.extrusion_handle = FakeHandle()
    steps = [
        SimpleNamespace(
            kind="setup",
            part_index=None,
            preview_segment=np.array([[0, 0, 0], [1, 0, 0]]),
            pose="setup",
        ),
        SimpleNamespace(
            kind="travel",
            part_index=0,
            preview_segment=np.array([[1, 0, 0], [2, 0, 0]]),
            pose="travel",
        ),
        SimpleNamespace(
            kind="extrusion",
            part_index=0,
            preview_segment=np.array([[2, 0, 0], [3, 0, 0]]),
            pose="extrusion",
        ),
    ]
    view.preview = SimpleNamespace(simulation_steps=steps)
    poses = []
    view.printer = SimpleNamespace(set_pose=poses.append)

    view._show_step(2)

    assert view.travel_handle.points.shape == (2, 2, 3)
    assert view.extrusion_handle.points.shape == (1, 2, 3)
    assert_array_equal(
        view.travel_handle.colors,
        [[SETUP_COLOR, SETUP_COLOR], [PART_COLORS[0], PART_COLORS[0]]],
    )
    assert_array_equal(
        view.extrusion_handle.colors,
        [[PART_COLORS[0], PART_COLORS[0]]],
    )
    assert poses == ["extrusion"]


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
