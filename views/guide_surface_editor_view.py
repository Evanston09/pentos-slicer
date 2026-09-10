from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from viser import ClientHandle

from models import GuideSurfaceSnapshot
from views.pose_editor_view import PoseEditorView
from views.theming import PENTOS_ORANGE


@dataclass
class GuideSurfaceState:
    guide: GuideSurfaceSnapshot
    mesh: Any
    control_handles: list[Any] = field(default_factory=list)


class GuideSurfaceEditorView:
    def __init__(
        self,
        client: ClientHandle,
        on_guide_changed: Callable[[int, np.ndarray, np.ndarray], None],
        on_guide_deleted: Callable[[int], None],
        on_guide_snap_requested: Callable[[int], None],
        on_guide_align_requested: Callable[[int], None],
        on_control_changed: Callable[[int, int, int, float, bool], None],
        on_apply_bend: Callable[[int, float, float], None],
        scene_prefix: str = "/guides",
    ) -> None:
        self.client = client
        self.on_control_changed = on_control_changed
        self.on_apply_bend = on_apply_bend
        self.on_guide_align_requested = on_guide_align_requested
        self.scene_prefix = scene_prefix
        self.pose_editor = PoseEditorView(
            client,
            on_guide_changed,
            on_guide_deleted,
            on_guide_snap_requested,
            item_label="Guide",
            delete_label="Delete Guide",
            scene_prefix=scene_prefix,
            gizmo_size=50.0,
            extras_factory=self._build_extras,
            extras_populate=self._populate_extras,
            on_selection_changed=self._on_selection_changed,
        )
        self.pose_editor.set_visible(False)
        self.guides: dict[int, GuideSurfaceState] = {}
        self.editing_guide_id: int | None = None
        self.selection_enabled = True

    def add_guide(self, guide: GuideSurfaceSnapshot, *, select: bool = True) -> None:
        guide_id = guide.guide_id

        self.pose_editor.add(guide_id, guide.position, guide.wxyz, select=select)
        vertices, faces = guide.preview_mesh()
        mesh = self.client.scene.add_mesh_simple(
            f"{self.scene_prefix}/{guide_id}/pose/mesh",
            vertices=vertices,
            faces=faces,
            color=PENTOS_ORANGE,
            opacity=0.35,
            side="double",
        )
        self.guides[guide_id] = GuideSurfaceState(guide, mesh)

        @mesh.on_click
        def _(_) -> None:
            if self.selection_enabled:
                self._toggle_select(guide_id)

    def _build_extras(self) -> dict[str, Any]:
        bend_x = self.client.gui.add_number("Bend X", 0.0, step=0.001)
        bend_y = self.client.gui.add_number("Bend Y", 0.0, step=0.001)
        apply_bend = self.client.gui.add_button("Apply")
        reset_flat = self.client.gui.add_button("Reset Flat")
        align_to_face = self.client.gui.add_button("Align to Face")
        edit_surface = self.client.gui.add_button("Edit Surface")

        @apply_bend.on_click
        def _(_) -> None:
            guide_id = self.pose_editor.selected_id
            if guide_id is not None:
                self.on_apply_bend(guide_id, bend_x.value, bend_y.value)

        @reset_flat.on_click
        def _(_) -> None:
            guide_id = self.pose_editor.selected_id
            if guide_id is not None:
                self.on_apply_bend(guide_id, 0.0, 0.0)

        @align_to_face.on_click
        def _(_) -> None:
            guide_id = self.pose_editor.selected_id
            if guide_id is not None:
                self.on_guide_align_requested(guide_id)

        @edit_surface.on_click
        def _(_) -> None:
            guide_id = self.pose_editor.selected_id
            if guide_id is None:
                return
            if self.editing_guide_id == guide_id:
                self._stop_editing()
            else:
                self._edit_surface(guide_id)

        return {
            "bend_x": bend_x,
            "bend_y": bend_y,
            "edit_surface": edit_surface,
        }

    def _populate_extras(self) -> None:
        self.pose_editor.gui["bend_x"].value = 0.0
        self.pose_editor.gui["bend_y"].value = 0.0

    def _edit_surface(self, guide_id: int) -> None:
        if self.editing_guide_id is not None and self.editing_guide_id != guide_id:
            self._remove_control_handles(self.editing_guide_id)
        self.editing_guide_id = guide_id
        state = self.guides[guide_id]
        x, y = state.guide.control_xy
        for row, y_value in enumerate(y):
            for column, x_value in enumerate(x):
                handle = self.client.scene.add_transform_controls(
                    f"{self.scene_prefix}/{guide_id}/pose/control_{row}_{column}",
                    position=(x_value, y_value, state.guide.heights_mm[row, column]),
                    scale=4.0,
                    active_axes=(False, False, True),
                    disable_sliders=True,
                    disable_rotations=True,
                    depth_test=False,
                )

                @handle.on_update
                async def _(event, row=row, column=column, handle=handle) -> None:
                    if event.phase != "start":
                        self.on_control_changed(
                            guide_id,
                            row,
                            column,
                            float(handle.position[2]),
                            event.phase == "end",
                        )

                state.control_handles.append(handle)
        self._sync_edit_button()

    def _stop_editing(self) -> None:
        if self.editing_guide_id is None:
            return
        self._remove_control_handles(self.editing_guide_id)
        self.editing_guide_id = None
        self._sync_edit_button()

    def _sync_edit_button(self) -> None:
        button = self.pose_editor.gui.get("edit_surface")
        if button is not None:
            button.label = (
                "Stop Editing Surface"
                if self.editing_guide_id is not None
                else "Edit Surface"
            )

    def _remove_control_handles(self, guide_id: int) -> None:
        state = self.guides.get(guide_id)
        if state is None:
            return
        for handle in state.control_handles:
            handle.remove()
        state.control_handles.clear()

    def clear(self) -> None:
        for guide_id in list(self.guides):
            self._remove_control_handles(guide_id)
            state = self.guides.pop(guide_id)
            state.mesh.remove()
        self.pose_editor.clear()

    def replace_guides(self, guides: list[GuideSurfaceSnapshot]) -> None:
        selected = self.pose_editor.selected_id
        self.clear()
        for guide in guides:
            self.add_guide(guide, select=False)
        if selected in self.guides:
            self.pose_editor.select(selected)

    def remove_guide(self, guide_id: int) -> None:
        state = self.guides.get(guide_id)
        if state is None:
            return
        self._remove_control_handles(guide_id)
        self.guides.pop(guide_id)
        state.mesh.remove()
        self.pose_editor.remove(guide_id)

    def set_visible(self, visible: bool) -> None:
        self.pose_editor.set_visible(visible)
        if not visible:
            self._stop_editing()

    def _toggle_select(self, guide_id: int) -> None:
        if guide_id not in self.guides:
            return
        if self.pose_editor.selected_id == guide_id:
            self.pose_editor.clear_selection()
            return
        self.pose_editor.select(guide_id)

    def _on_selection_changed(self, guide_id: int | None) -> None:
        if self.editing_guide_id != guide_id:
            self._stop_editing()

    def update_guide(self, guide: GuideSurfaceSnapshot) -> None:
        state = self.guides.get(guide.guide_id)
        if state is not None:
            state.guide = guide
            pose = self.pose_editor.items[guide.guide_id].pose
            # A pose drag already moved the view; don't reset its active gizmo.
            if not (
                np.array_equal(pose.position, guide.position)
                and np.array_equal(pose.wxyz, guide.wxyz)
            ):
                self.pose_editor.set_pose(guide.guide_id, guide.position, guide.wxyz)
            vertices, faces = guide.preview_mesh()
            state.mesh.vertices = vertices
            state.mesh.faces = faces
            x, y = guide.control_xy
            for index, handle in enumerate(state.control_handles):
                row, column = divmod(index, len(x))
                handle.position = (x[column], y[row], guide.heights_mm[row, column])
