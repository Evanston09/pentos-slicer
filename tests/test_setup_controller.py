from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest
import trimesh

import controllers.setup_controller as setup_controller_module
from controllers.setup_controller import SetupController
from models import AppState, GuideSurfaceSnapshot, MachineConfig
from services.machine_config_io import save_machine_config
from services.model_tools import transformed_model
from services.project_io import load_scene, save_scene


class FakeSetupView:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.slice_progress: list[tuple[float | None, str | None]] = []
        self.planes = []
        self.mesh = None
        self.removed_plane_ids: list[int] = []
        self.plane_poses = {}
        self.debug_mode = False
        self.mounted = False
        self.model_out_of_bounds = False
        self.slice_enabled = []
        self.slicing_mode = "multiplanar"
        self.guides = []
        self.machine_config = None

    def mount(self, state: AppState) -> None:
        self.mounted = True

    def unmount(self) -> None:
        self.mounted = False

    def set_status(self, message: str) -> None:
        self.statuses.append(message)

    def set_slice_progress(self, progress, message=None) -> None:
        self.slice_progress.append((progress, message))
        if message is not None:
            self.statuses.append(message)

    def show_machine_config(self, config) -> None:
        self.machine_config = config

    def set_slice_enabled(self, enabled: bool) -> None:
        self.slice_enabled.append(enabled)

    def set_slicing_mode(self, mode: str) -> None:
        self.slicing_mode = mode

    def show_mesh(self, mesh, center, position, wxyz) -> None:
        self.mesh = mesh

    def clear_model_scene(self) -> None:
        self.mesh = None

    def show_overhang_faces(self, mesh, overhang_mask) -> None:
        pass

    def set_model_out_of_bounds(self, out_of_bounds: bool) -> None:
        self.model_out_of_bounds = out_of_bounds

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

    def set_plane_pose(self, plane_id, position, wxyz) -> None:
        self.plane_poses[plane_id] = (position, wxyz)

    def set_debug_mode_value(self, enabled: bool) -> None:
        self.debug_mode = enabled

    def replace_guide_surfaces(self, guides) -> None:
        self.guides = list(guides)

    def add_guide_surface(self, guide) -> None:
        self.guides.append(guide)

    def remove_guide_surface(self, guide_id: int) -> None:
        self.guides = [guide for guide in self.guides if guide.guide_id != guide_id]

    def update_guide_surface(self, guide) -> None:
        self.guides = [
            guide if previous.guide_id == guide.guide_id else previous
            for previous in self.guides
        ]


class FakeSlicer:
    def __init__(self) -> None:
        self.calls = []
        self.error: Exception | None = None

    def slice(self, mesh, planes, source_name, *, progress) -> Path:
        if self.error is not None:
            raise self.error
        self.calls.append(("slice", mesh, list(planes), source_name))
        progress(0.5, "Slicing chunk 1 of 1...")
        return Path("output/model.gcode")

    def debug_transition_check(self, mesh, planes, source_name, *, progress) -> Path:
        if self.error is not None:
            raise self.error
        self.calls.append(("debug", mesh, list(planes), source_name))
        progress(0.5, "Slicing chunk 1 of 1...")
        return Path("output/model_debug.gcode")


class FakeWorkspace:
    def __init__(self, path: Path = Path(".")) -> None:
        self.path = path

    @contextmanager
    def active_job(self):
        yield


def make_controller(
    state: AppState | None = None,
    workspace_path: Path = Path("."),
    slicer=None,
    slicing_slots=None,
    persist_machine_config=lambda config: None,
):
    view = FakeSetupView()
    slicer = FakeSlicer() if slicer is None else slicer
    slicing_slots = BoundedSemaphore(2) if slicing_slots is None else slicing_slots
    navigations = []
    controller = SetupController(
        AppState() if state is None else state,
        slicer,
        view,
        lambda: navigations.append("preview"),
        FakeWorkspace(workspace_path),
        slicing_slots,
        persist_machine_config,
    )
    return controller, view, slicer, navigations


def test_machine_config_upload_applies_to_session() -> None:
    persisted = []
    controller, view, slicer, _ = make_controller(
        persist_machine_config=persisted.append
    )
    config = MachineConfig(
        name="Large Pentos",
        build_volume_mm=(120.0, 100.0, 150.0),
        machine_plate_center_mm=(130.0, 70.0, 0.0),
        rotation_center_machine_mm=(129.0, 69.0, 3.0),
    )

    controller.import_machine_config(save_machine_config(config))

    assert controller.state.machine_config == config
    assert controller.state.model_xy_position == (60.0, 50.0)
    assert slicer.machine_config == config
    assert view.machine_config == config
    assert persisted == [config]
    assert view.statuses[-1] == "Loaded machine Large Pentos"


def test_upload_and_placement_update_state(monkeypatch, tmp_path) -> None:
    mesh = trimesh.creation.box()

    def load_model(name, content, upload_dir):
        assert upload_dir == tmp_path / "uploads"
        return mesh, "uploaded"

    monkeypatch.setattr(
        setup_controller_module,
        "load_uploaded_model",
        load_model,
    )
    controller, view, _, _ = make_controller(workspace_path=tmp_path)

    controller.handle_upload("uploaded.stl", b"mesh")
    controller.set_model_placement([12.0, 34.0], 45.0)

    assert controller.state.current_model == (mesh, "uploaded")
    assert controller.state.model_xy_position == (12.0, 34.0)
    assert controller.state.model_z_degrees == 45.0
    assert view.mesh is mesh
    assert view.statuses[-1] == "Loaded uploaded"


def test_upload_rejects_unsupported_file_type(tmp_path) -> None:
    controller, view, _, _ = make_controller(workspace_path=tmp_path)

    controller.handle_upload("model.exe", b"content")

    assert controller.state.current_model is None
    assert "Unsupported file type" in view.statuses[-1]


def test_upload_rejects_oversized_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    controller, view, _, _ = make_controller(workspace_path=tmp_path)

    controller.handle_upload("model.stl", b"x" * (1024 * 1024 + 1))

    assert controller.state.current_model is None
    assert "exceeds the 1 MB limit" in view.statuses[-1]


def test_model_turns_red_when_placement_leaves_build_volume() -> None:
    mesh = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    mesh.apply_translation([45.0, 45.0, 5.0])
    state = AppState(current_model=(mesh, "box"))
    controller, view, _, _ = make_controller(state)

    controller.mount()
    assert not view.model_out_of_bounds

    controller.set_model_placement([86.0, 45.0])
    assert view.model_out_of_bounds

    controller.set_model_placement([45.0, 45.0])
    assert not view.model_out_of_bounds


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


def test_plane_snaps_to_clicked_model_face() -> None:
    mesh = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    mesh.apply_translation([45.0, 45.0, 5.0])
    controller, view, _, _ = make_controller(AppState(current_model=(mesh, "box")))
    controller.add_plane()

    snapped = controller.snap_plane_to_face(
        0,
        np.array([60.0, 45.0, 5.0]),
        np.array([-1.0, 0.0, 0.0]),
    )

    assert snapped
    plane = controller.state.plane_snapshots[0]
    assert_allclose(plane.position, [50.0, 45.0, 5.0])
    rotation = trimesh.transformations.quaternion_matrix(plane.wxyz)[:3, :3]
    assert_allclose(rotation @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    assert 0 in view.plane_poses


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


def test_added_guides_default_to_parallel_ordered_surfaces() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "box"))
    controller, view, _, _ = make_controller(state)

    controller.nonplanar.add_guide()
    controller.nonplanar.add_guide()

    first, second = controller.state.guide_surfaces
    assert second.position[2] > first.position[2]
    assert_allclose(second.wxyz, first.wxyz)
    assert_allclose(first.size_mm, [100.0, 100.0])
    assert_allclose(second.size_mm, first.size_mm)
    assert_allclose(second.heights_mm, first.heights_mm)
    assert not any("cross" in status for status in view.statuses)


def test_guides_retain_edits_when_snapped() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "box"))
    controller, _, _, _ = make_controller(state)
    controller.nonplanar.add_guide()
    controller.nonplanar.add_guide()
    first, _second = state.guide_surfaces

    controller.nonplanar.set_control_height(first.guide_id, 0, 0, 3.0)
    edited = first.heights_mm.copy()
    controller.nonplanar.snap_guide_to_face(
        first.guide_id, np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0])
    )
    assert_allclose(first.heights_mm, edited)


def test_align_guide_conforms_to_curved_face() -> None:
    state = AppState(current_model=(trimesh.creation.icosphere(radius=5.0), "sphere"))
    controller, _, _, _ = make_controller(state)
    controller.nonplanar.add_guide()
    guide = state.guide_surfaces[0]
    # Keep the accuracy check inside the selected cap; the patch corners of a
    # 6 mm square lie outside the region within 45 degrees of the clicked normal.
    guide.size_mm = np.array([4.0, 4.0])

    snapped = controller.nonplanar.align_guide_to_face(
        guide.guide_id, np.array([45.0, 45.0, 10.0]), np.array([0.0, 0.0, -1.0])
    )
    assert snapped
    hit_mesh = transformed_model(state)[0]
    surface, _, _ = hit_mesh.nearest.on_surface([guide.position])
    assert_allclose(guide.position, surface[0], atol=1e-6)

    center_height = guide.evaluate_local(np.zeros((1, 2)))[0][0]
    world_center = guide.position + center_height * guide.rotation[:, 2]
    assert_allclose(world_center, guide.position, atol=1e-6)
    assert np.ptp(guide.heights_mm) > 0.0, "align should warp heights to curvature"

    tangent = guide.rotation[:, 2]
    half = guide.size_mm / 2.0
    x = np.linspace(-half[0], half[0], 9)
    y = np.linspace(-half[1], half[1], 9)
    grid_x, grid_y = np.meshgrid(x, y)
    local_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    heights, _, _ = guide.evaluate_local(local_xy)
    world = (
        guide.position + local_xy @ guide.rotation[:, :2].T + heights[:, None] * tangent
    )
    surface, _, _ = transformed_model(state)[0].nearest.on_surface(world)
    assert np.max(np.linalg.norm(world - surface, axis=1)) < 0.1


def test_align_guide_to_flat_face_keeps_flat() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "box"))
    controller, _, _, _ = make_controller(state)
    controller.nonplanar.add_guide()
    guide = state.guide_surfaces[0]

    snapped = controller.nonplanar.align_guide_to_face(
        guide.guide_id, np.array([45.3, 45.2, 2.0]), np.array([0.0, 0.0, -1.0])
    )
    assert snapped
    box_mesh = transformed_model(state)[0]
    expected_center = box_mesh.bounds.mean(axis=0)
    expected_center[2] = box_mesh.bounds[1, 2]
    assert_allclose(guide.position, expected_center)
    assert_allclose(guide.heights_mm, np.zeros((4, 4)), atol=1e-6)


def test_guide_fit_heights_reproduces_bicubic_field() -> None:
    guide = GuideSurfaceSnapshot(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0)
    heights = np.array(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 2.0],
            [2.0, 3.0, 4.0, 3.0],
            [1.0, 2.0, 3.0, 2.0],
        ]
    )
    guide.heights_mm = heights

    x = np.linspace(-guide.size_mm[0] / 2.0, guide.size_mm[0] / 2.0, 9)
    y = np.linspace(-guide.size_mm[1] / 2.0, guide.size_mm[1] / 2.0, 9)
    grid_x, grid_y = np.meshgrid(x, y)
    xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    values, _, _ = guide.evaluate_local(xy)
    fitted = guide.fit_heights(xy, values)

    assert_allclose(fitted, heights, atol=1e-6)


def test_align_guide_conforms_to_parabolic_top() -> None:
    content = Path("samples/flat_base_curved_top.pentos").read_bytes()
    state = load_scene(content)
    controller, _, _, _ = make_controller(state)
    controller.nonplanar.add_guide()
    guide = state.guide_surfaces[-1]

    snapped = controller.nonplanar.align_guide_to_face(
        guide.guide_id, np.array([45.0, 45.0, 20.0]), np.array([0.0, 0.0, -1.0])
    )
    assert snapped
    assert np.min(guide.heights_mm) < -1.0, "guide should conform to the dome"

    curvature = 0.0025
    expected_columns = (
        -curvature
        * np.linspace(-guide.size_mm[0] / 2.0, guide.size_mm[0] / 2.0, 4) ** 2
    )
    for row in guide.heights_mm:
        assert_allclose(row, expected_columns, atol=0.1)


def test_scalar_field_surface_colors_tetrahedral_boundary() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, _, _, _ = make_controller(state)

    with pytest.raises(ValueError, match="at least two guide surfaces"):
        controller.nonplanar.scalar_field_surface()

    controller.nonplanar.add_guide()
    controller.nonplanar.add_guide()

    vertices, faces, values = controller.nonplanar.scalar_field_surface()

    assert len(vertices) > 0
    assert len(faces) > 0
    assert len(values) == len(vertices)
    assert np.ptp(values) > 0.0


def test_slice_dispatches_normal_and_debug_modes() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, view, slicer, navigations = make_controller(state)

    controller.slice_model()
    state.debug_mode = True
    controller.slice_model()

    assert [call[0] for call in slicer.calls] == ["slice", "debug"]
    assert navigations == ["preview", "preview"]
    assert state.gcode_path == Path("output/model_debug.gcode")
    assert view.slice_progress[-1] == (None, None)
    assert (1.0, "Opening preview...") in view.slice_progress


def test_nonplanar_slice_uses_deformed_mesh(tmp_path) -> None:
    class NonplanarFakeSlicer(FakeSlicer):
        def slice(self, mesh, planes, source_name, *, progress) -> Path:
            super().slice(mesh, planes, source_name, progress=progress)
            path = tmp_path / f"{source_name}.gcode"
            path.write_text("G90\n")
            return path

    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, view, slicer, navigations = make_controller(
        state,
        slicer=NonplanarFakeSlicer(),
    )
    controller.nonplanar.add_guide()
    controller.nonplanar.add_guide()

    controller.set_slicing_mode("nonplanar")
    controller.slice_model()

    operation, mesh, planes, source_name = slicer.calls[0]
    assert operation == "slice"
    assert mesh.is_volume
    assert planes == []
    assert source_name == "model_deformed"
    assert view.slicing_mode == "nonplanar"
    assert controller.state.gcode_path == tmp_path / "model_mapped.gcode"
    assert navigations == ["preview"]
    assert (0.05, "Deforming model...") in view.slice_progress
    assert (0.85, "Inverse-mapping G-code...") in view.slice_progress
    assert view.slice_progress[-1] == (None, None)


def test_slice_failure_does_not_navigate() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, view, slicer, navigations = make_controller(state)
    slicer.error = RuntimeError("slicer failed")

    controller.slice_model()

    assert navigations == []
    assert view.statuses[-1] == "Failed to slice: slicer failed"
    assert view.slice_enabled == [False, True]
    assert view.slice_progress[-1] == (None, None)


def test_slice_reports_busy_server() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    slicing_slots = BoundedSemaphore(1)
    slicing_slots.acquire()
    controller, view, slicer, navigations = make_controller(
        state,
        slicing_slots=slicing_slots,
    )

    controller.slice_model()

    assert slicer.calls == []
    assert navigations == []
    assert view.statuses[-1] == "Server is busy slicing other models"
    assert view.slice_enabled == [False, True]
    assert view.slice_progress[-1] == (None, None)


def test_loading_nonplanar_project_restores_guides_and_mode() -> None:
    project = AppState(
        current_model=(trimesh.creation.box(), "project"),
        guide_surfaces=[
            GuideSurfaceSnapshot(
                position=np.array([0.0, 0.0, 0.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                guide_id=4,
            )
        ],
        slicing_mode="nonplanar",
    )
    controller, view, _, _ = make_controller()

    controller.handle_upload("project.pentos", save_scene(project))

    assert controller.state.slicing_mode == "nonplanar"
    assert len(controller.state.guide_surfaces) == 1
    assert controller.state.guide_surfaces[0].guide_id == 0
    assert view.guides == controller.state.guide_surfaces
    assert view.slicing_mode == "nonplanar"
    assert controller.nonplanar.next_guide_id == 1


def test_export_returns_scene_filename_and_bytes() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "model"))
    controller, view, _, _ = make_controller(state)

    result = controller.export_scene()

    assert result is not None
    filename, content = result
    assert filename == "model.pentos"
    assert content.startswith(b"PK")
    assert view.statuses[-1] == "Exported model.pentos"


def test_failed_alignment_preserves_guide_and_reports_reason(monkeypatch) -> None:
    state = AppState(current_model=(trimesh.creation.box(), "box"))
    controller, view, _, _ = make_controller(state)
    controller.nonplanar.add_guide()
    guide = state.guide_surfaces[0]
    guide.apply_bend_preset(0.1, 0.2)
    before = guide.as_dict()

    def failed_fit(self, xy, heights):
        raise ValueError("Surface samples do not determine a full 4x4 guide fit")

    monkeypatch.setattr(GuideSurfaceSnapshot, "fit_heights", failed_fit)
    assert not controller.nonplanar.align_guide_to_face(
        guide.guide_id, np.array([45.3, 45.2, 2.0]), np.array([0.0, 0.0, -1.0])
    )
    assert guide.as_dict() == before
    assert "alignment failed" in view.statuses[-1]
    assert "full 4x4 guide fit" in view.statuses[-1]
