from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import trimesh

import controllers.setup_controller as setup_controller_module
from controllers.setup_controller import SetupController
from models import AppState


class FakeSetupView:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.planes = []
        self.mesh = None
        self.removed_plane_ids: list[int] = []
        self.debug_mode = False
        self.mounted = False

    def mount(self, state: AppState) -> None:
        self.mounted = True

    def unmount(self) -> None:
        self.mounted = False

    def set_status(self, message: str) -> None:
        self.statuses.append(message)

    def show_mesh(self, mesh, center, position, wxyz) -> None:
        self.mesh = mesh

    def clear_model_scene(self) -> None:
        self.mesh = None

    def show_overhang_faces(self, overhang_mask) -> None:
        pass

    def set_model_controls_enabled(self, enabled: bool) -> None:
        pass

    def update_model_placement(self, xy_position, z_degrees, position, wxyz) -> None:
        self.placement = (list(xy_position), z_degrees, position, wxyz)

    def replace_planes(self, planes) -> None:
        self.planes = list(planes)

    def add_plane(self, plane) -> None:
        self.planes.append(plane)

    def remove_plane(self, plane_id: int) -> None:
        self.removed_plane_ids.append(plane_id)
        self.planes = [plane for plane in self.planes if plane.plane_id != plane_id]

    def set_debug_mode_value(self, enabled: bool) -> None:
        self.debug_mode = enabled


class FakeSlicer:
    def __init__(self) -> None:
        self.calls = []
        self.error: Exception | None = None

    def slice(self, mesh, planes, source_name) -> Path:
        if self.error is not None:
            raise self.error
        self.calls.append(("slice", mesh, list(planes), source_name))
        return Path("output/model.gcode")

    def debug_transition_check(self, mesh, planes, source_name) -> Path:
        if self.error is not None:
            raise self.error
        self.calls.append(("debug", mesh, list(planes), source_name))
        return Path("output/model_debug.gcode")


def make_controller(state: AppState | None = None):
    view = FakeSetupView()
    slicer = FakeSlicer()
    navigations = []
    controller = SetupController(
        AppState() if state is None else state,
        slicer,
        view,
        lambda: navigations.append("preview"),
    )
    return controller, view, slicer, navigations


def test_upload_and_placement_update_state(monkeypatch) -> None:
    mesh = trimesh.creation.box()
    monkeypatch.setattr(
        setup_controller_module,
        "load_uploaded_model",
        lambda name, content: (mesh, "uploaded"),
    )
    controller, view, _, _ = make_controller()

    controller.handle_upload("uploaded.stl", b"mesh")
    controller.set_model_placement([12.0, 34.0], 45.0)

    assert controller.state.current_model == (mesh, "uploaded")
    assert controller.state.model_xy_position == [12.0, 34.0]
    assert controller.state.model_z_degrees == 45.0
    assert view.mesh is mesh
    assert view.statuses[-1] == "Loaded uploaded"


def test_plane_changes_update_canonical_state_immediately() -> None:
    controller, view, _, _ = make_controller()

    controller.add_plane()
    plane = controller.state.plane_snapshots[0]
    assert plane.plane_id == 0

    controller.update_plane(
        0,
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 0.0, 0.0, 0.0]),
    )
    assert_allclose(plane.position, [1.0, 2.0, 3.0])
    assert_allclose(plane.wxyz, [1.0, 0.0, 0.0, 0.0])

    controller.remove_plane(0)
    assert controller.state.plane_snapshots == []
    assert view.removed_plane_ids == [0]


def test_auto_planes_replace_existing_planes(monkeypatch) -> None:
    state = AppState(current_model=(trimesh.creation.box(), "box"))
    controller, view, _, _ = make_controller(state)

    class FakeSelector:
        def __init__(self, config) -> None:
            assert config.max_planes == 2

        def select(self, mesh):
            return [
                SimpleNamespace(
                    position=np.array([1.0, 2.0, 3.0]),
                    wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                )
            ]

    monkeypatch.setattr(
        setup_controller_module,
        "AutoPlaneSelector",
        FakeSelector,
    )
    controller.select_auto_planes(2)

    assert len(controller.state.plane_snapshots) == 1
    assert view.planes == controller.state.plane_snapshots
    assert view.statuses[-1] == "Auto Planes: selected 1 plane(s)"


def test_slice_dispatches_normal_and_debug_modes() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, _, slicer, navigations = make_controller(state)

    controller.slice_model()
    state.debug_mode = True
    controller.slice_model()

    assert [call[0] for call in slicer.calls] == ["slice", "debug"]
    assert navigations == ["preview", "preview"]
    assert state.gcode_path == Path("output/model_debug.gcode")


def test_slice_failure_does_not_navigate() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, view, slicer, navigations = make_controller(state)
    slicer.error = RuntimeError("slicer failed")

    controller.slice_model()

    assert navigations == []
    assert view.statuses[-1] == "Failed to slice: slicer failed"


def test_export_returns_scene_filename_and_bytes() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, view, _, _ = make_controller(state)

    result = controller.export_scene()

    assert result is not None
    filename, content = result
    assert filename == "model.pentos"
    assert content.startswith(b"PK")
    assert view.statuses[-1] == "Exported model.pentos"
