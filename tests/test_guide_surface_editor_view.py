import asyncio
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from controllers.nonplanar_controller import NonplanarController
from models import AppState
from views.guide_surface_editor_view import GuideSurfaceEditorView


class FakeHandle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.removed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_update(self, callback):
        self.update_callback = callback
        return callback

    def on_click(self, callback):
        self.click_callback = callback
        return callback

    def remove(self):
        self.removed = True


class FakeGui:
    def add_folder(self, label, **kwargs):
        return FakeHandle(label=label, **kwargs)

    def add_number(self, label, value, **kwargs):
        return FakeHandle(label=label, value=value, **kwargs)

    add_vector3 = add_number
    add_button = add_folder


class FakeScene:
    def add_frame(self, name, **kwargs):
        return FakeHandle(
            **{"position": np.zeros(3), "wxyz": np.array([1, 0, 0, 0]), **kwargs}
        )

    add_transform_controls = add_frame
    add_mesh_simple = add_frame


def make_editor():
    state = AppState()
    client = SimpleNamespace(gui=FakeGui(), scene=FakeScene())
    editor = GuideSurfaceEditorView(
        client,
        lambda *args: controller.update_guide(*args),
        lambda guide_id: controller.remove_guide(guide_id),
        lambda guide_id: None,
        lambda guide_id: None,
        lambda guide_id, row, column, height, finished: controller.set_control_height(
            guide_id, row, column, height, finished=finished
        ),
        lambda *args: controller.apply_bend(*args),
    )
    editor.pose_editor.gui_container = FakeHandle()
    editor.set_visible(True)
    controller = NonplanarController(
        state,
        SimpleNamespace(
            add_guide_surface=editor.add_guide,
            update_guide_surface=editor.update_guide,
            remove_guide_surface=editor.remove_guide,
        ),
    )
    controller.add_guide()
    return controller, editor


def test_surface_drag_keeps_handles_and_pose_until_end():
    controller, editor = make_editor()
    guide = controller.state.guide_surfaces[0]
    editor.pose_editor.gui["edit_surface"].click_callback(None)
    handles = list(editor.guides[guide.guide_id].control_handles)
    handle = handles[0]
    original_position = guide.position.copy()
    asyncio.run(handle.update_callback(SimpleNamespace(phase="start")))
    for height in (1.0, 2.0, 3.0):
        handle.position = (*handle.position[:2], height)
        asyncio.run(handle.update_callback(SimpleNamespace(phase="update")))
        assert_allclose(guide.position, original_position)
        assert guide.heights_mm[0, 0] == height
        assert editor.guides[guide.guide_id].control_handles == handles
        assert not any(control.removed for control in handles)

    before_height = guide.evaluate_local([[0.0, 0.0], [10.0, 20.0]])[0]
    asyncio.run(handle.update_callback(SimpleNamespace(phase="end")))
    assert_allclose(guide.evaluate_local([[0.0, 0.0]])[0], 0.0, atol=1e-12)
    after_height = guide.evaluate_local([[0.0, 0.0], [10.0, 20.0]])[0]
    assert_allclose(
        after_height + guide.position[2], before_height + original_position[2]
    )
    assert editor.guides[guide.guide_id].control_handles == handles
    assert not any(control.removed for control in handles)
    assert_allclose(handle.position[2], guide.heights_mm[0, 0])


@pytest.mark.parametrize("selection_action", ["add", "select", "clear", "delete"])
def test_selection_changes_close_surface_controls(selection_action):
    controller, editor = make_editor()
    first_id = controller.state.guide_surfaces[0].guide_id
    controller.add_guide()
    editor.pose_editor.select(first_id)
    editor.pose_editor.gui["edit_surface"].click_callback(None)
    handles = list(editor.guides[first_id].control_handles)

    if selection_action == "add":
        controller.add_guide()
    elif selection_action == "select":
        editor.pose_editor.select(controller.state.guide_surfaces[1].guide_id)
    elif selection_action == "clear":
        editor.pose_editor.clear_selection()
    else:
        controller.remove_guide(first_id)

    assert editor.editing_guide_id is None
    assert all(handle.removed for handle in handles)
    assert editor.pose_editor.gui["edit_surface"].label == "Edit Surface"


@pytest.mark.parametrize("replace_guide", [False, True])
def test_render_updates_refresh_mesh_and_surface_controls(replace_guide):
    controller, editor = make_editor()
    guide = controller.state.guide_surfaces[0]
    rendered = editor.guides[guide.guide_id]
    if replace_guide:
        guide = type(guide).from_dict(guide.as_dict(), guide.guide_id)
    guide.heights_mm[0, 0] = 5.0
    editor.update_guide(guide)
    assert_allclose(rendered.mesh.vertices, guide.preview_mesh()[0])
    editor.pose_editor.gui["edit_surface"].click_callback(None)
    assert rendered.control_handles[0].position[2] == 5.0


def test_pose_drag_updates_do_not_reset_active_gizmo():
    controller, editor = make_editor()
    guide = controller.state.guide_surfaces[0]
    pose_editor = editor.pose_editor
    rendered = pose_editor.items[guide.guide_id]
    start = np.array(rendered.pose.position)
    pose_editor._on_gizmo_update(guide.guide_id, SimpleNamespace(phase="start"))
    for distance in (1.0, 2.0):
        rendered.gizmo.position = np.array([distance, 0.0, 0.0])
        pose_editor._on_gizmo_update(guide.guide_id, SimpleNamespace(phase="update"))
        assert_allclose(rendered.anchor.position, start)
        assert_allclose(rendered.gizmo.position, [distance, 0.0, 0.0])
        assert_allclose(guide.position, start + [distance, 0.0, 0.0])
    pose_editor._on_gizmo_update(guide.guide_id, SimpleNamespace(phase="end"))
    assert_allclose(rendered.anchor.position, guide.position)
    assert_allclose(rendered.gizmo.position, np.zeros(3))
