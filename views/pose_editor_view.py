from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from trimesh import transformations as tf
from viser import ClientHandle

GIZMO_LINE_WIDTH = 5.0
GIZMO_SCALE = 0.45


def neutral_wxyz() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def neutral_position() -> np.ndarray:
    return np.zeros(3)


@dataclass
class PoseState:
    pose: Any
    anchor: Any
    gizmo: Any
    start_position: np.ndarray | None = None
    start_wxyz: np.ndarray | None = None


class PoseEditorView:
    def __init__(
        self,
        client: ClientHandle,
        on_changed: Callable[[int, np.ndarray, np.ndarray], None],
        on_deleted: Callable[[int], None],
        on_snap_requested: Callable[[int], None],
        *,
        item_label: str,
        delete_label: str,
        scene_prefix: str,
        gizmo_size: float,
        extras_factory: Callable[[], dict[str, Any]] | None = None,
        extras_populate: Callable[[], None] | None = None,
        on_selection_changed: Callable[[int | None], None] | None = None,
    ) -> None:
        self.client = client
        self.on_changed = on_changed
        self.on_deleted = on_deleted
        self.on_snap_requested = on_snap_requested
        self.item_label = item_label
        self.delete_label = delete_label
        self.scene_prefix = scene_prefix
        self.gizmo_size = gizmo_size
        self.extras_factory = extras_factory
        self.extras_populate = extras_populate
        self.on_selection_changed = on_selection_changed
        self.gui_container: Any | None = None
        self.items: dict[int, PoseState] = {}
        self.syncing_gui = False
        self.visible = True
        self.selected_id: int | None = None
        self.panel: Any | None = None
        self.gui: dict[str, Any] = {}

    def add(
        self,
        item_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
        *,
        select: bool = True,
    ) -> PoseState:
        if self.gui_container is None:
            raise RuntimeError("Set a GUI container before adding items")
        self._ensure_panel()

        pose = self.client.scene.add_frame(
            f"{self.scene_prefix}/{item_id}/pose",
            position=position,
            wxyz=self._normalize_quaternion(wxyz),
            visible=self.visible,
        )
        anchor = self.client.scene.add_frame(
            f"{self.scene_prefix}/{item_id}/gizmo_anchor",
            position=pose.position,
            visible=self.visible,
        )
        gizmo = self.client.scene.add_transform_controls(
            f"{self.scene_prefix}/{item_id}/gizmo_anchor/controls",
            scale=self.gizmo_size * GIZMO_SCALE,
            line_width=GIZMO_LINE_WIDTH,
            depth_test=False,
        )

        state = PoseState(pose=pose, anchor=anchor, gizmo=gizmo)
        self.items[item_id] = state

        @gizmo.on_update
        async def _(event) -> None:
            self._on_gizmo_update(item_id, event)

        if select:
            self.select(item_id)
        else:
            self._refresh_gizmos()

        return state

    def _ensure_panel(self) -> None:
        if self.panel is not None:
            return
        with self.gui_container:
            panel = self.client.gui.add_folder(
                self.item_label,
                expand_by_default=True,
                visible=False,
            )
        with panel:
            position_control = self.client.gui.add_vector3(
                "Position", np.zeros(3), step=0.001
            )
            rotation_control = self.client.gui.add_vector3(
                "Rotation", [0.0, 0.0, 0.0], step=1.0
            )
            extras = {} if self.extras_factory is None else self.extras_factory()
            snap_button = self.client.gui.add_button("Snap to Face")
            delete_button = self.client.gui.add_button(self.delete_label)

        self.panel = panel
        self.gui = {
            "position": position_control,
            "rotation": rotation_control,
            **extras,
        }

        @position_control.on_update
        def _(_) -> None:
            if self.syncing_gui or self.selected_id is None:
                return
            self._set_pose(self.selected_id, position=self.gui["position"].value)
            self._notify_changed(self.selected_id)

        @rotation_control.on_update
        def _(_) -> None:
            if self.syncing_gui or self.selected_id is None:
                return
            rx, ry, rz = self.gui["rotation"].value
            self._set_pose(
                self.selected_id,
                wxyz=tf.quaternion_from_euler(
                    np.radians(rx),
                    np.radians(ry),
                    np.radians(rz),
                    axes="sxyz",
                ),
            )
            self._notify_changed(self.selected_id)

        @snap_button.on_click
        def _(_) -> None:
            if self.selected_id is not None:
                self.on_snap_requested(self.selected_id)

        @delete_button.on_click
        def _(_) -> None:
            if self.selected_id is not None:
                self.on_deleted(self.selected_id)

    def select(self, item_id: int) -> None:
        if item_id not in self.items:
            return
        changed = self.selected_id != item_id
        self.selected_id = item_id
        if changed and self.on_selection_changed is not None:
            self.on_selection_changed(item_id)
        if self.panel is not None:
            self.panel.visible = self.visible
            self.panel.label = f"{self.item_label} {item_id}"
            self._sync_gui_from_pose(item_id)
            if self.extras_populate is not None:
                self.extras_populate()
        self._refresh_gizmos()

    def clear_selection(self) -> None:
        changed = self.selected_id is not None
        self.selected_id = None
        if changed and self.on_selection_changed is not None:
            self.on_selection_changed(None)
        if self.panel is not None:
            self.panel.visible = False
        self._refresh_gizmos()

    def clear(self) -> None:
        for item_id in list(self.items):
            self._remove_item(item_id)
        self.clear_selection()

    def remove(self, item_id: int) -> None:
        was_selected = self.selected_id == item_id
        self._remove_item(item_id)
        if not was_selected:
            return
        if self.items:
            self.select(next(iter(self.items)))
        else:
            self.clear_selection()

    def _remove_item(self, item_id: int) -> None:
        state = self.items.pop(item_id, None)
        if state is None:
            return
        state.gizmo.remove()
        state.anchor.remove()
        state.pose.remove()

    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        for state in self.items.values():
            state.pose.visible = visible
            state.anchor.visible = visible
        self._refresh_gizmos()
        if self.panel is not None:
            self.panel.visible = visible and self.selected_id is not None

    def set_pose(
        self,
        item_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        if item_id not in self.items:
            return
        self._set_pose(item_id, position, wxyz)
        self._sync_gui_from_pose(item_id)

    def _set_pose(self, item_id: int, position=None, wxyz=None) -> None:
        state = self.items[item_id]
        if position is not None:
            state.pose.position = np.array(position)
        if wxyz is not None:
            state.pose.wxyz = self._normalize_quaternion(wxyz)
        self._reset_gizmo(state)

    def _on_gizmo_update(self, item_id: int, event) -> None:
        state = self.items.get(item_id)
        if state is None:
            return

        if event.phase == "start":
            state.start_position = np.array(state.pose.position)
            state.start_wxyz = self._normalize_quaternion(state.pose.wxyz)
            return

        position_delta = np.array(state.gizmo.position)
        rotation_delta = self._normalize_quaternion(state.gizmo.wxyz)
        if not self._is_neutral_wxyz(rotation_delta):
            state.pose.wxyz = self._normalize_quaternion(
                tf.quaternion_multiply(rotation_delta, state.start_wxyz)
            )
        else:
            state.pose.position = state.start_position + position_delta

        if event.phase == "end":
            state.start_position = None
            state.start_wxyz = None
            self._reset_gizmo(state)

        self._sync_gui_from_pose(item_id)
        self._notify_changed(item_id)

    def _sync_gui_from_pose(self, item_id: int) -> None:
        if item_id != self.selected_id or self.panel is None:
            return
        state = self.items.get(item_id)
        if state is None:
            return
        self.syncing_gui = True
        try:
            rx, ry, rz = self._euler_degrees(state.pose.wxyz)
            self.gui["position"].value = state.pose.position
            self.gui["rotation"].value = (rx, ry, rz)
        finally:
            self.syncing_gui = False

    def _notify_changed(self, item_id: int) -> None:
        state = self.items.get(item_id)
        if state is not None:
            self.on_changed(
                item_id,
                np.array(state.pose.position),
                np.array(state.pose.wxyz),
            )

    def _refresh_gizmos(self) -> None:
        for item_id, state in self.items.items():
            state.gizmo.visible = self.visible and item_id == self.selected_id

    @staticmethod
    def _reset_gizmo(state: PoseState) -> None:
        state.anchor.position = state.pose.position
        state.gizmo.position = neutral_position()
        state.gizmo.wxyz = neutral_wxyz()

    @staticmethod
    def _normalize_quaternion(wxyz) -> np.ndarray:
        wxyz = np.array(wxyz, dtype=float)
        norm = np.linalg.norm(wxyz)
        if np.isclose(norm, 0.0):
            return neutral_wxyz()
        return wxyz / norm

    @staticmethod
    def _is_neutral_wxyz(wxyz) -> bool:
        wxyz = np.array(wxyz, dtype=float)
        return bool(
            np.isclose(np.linalg.norm(wxyz[1:]), 0.0) and np.isclose(abs(wxyz[0]), 1.0)
        )

    @staticmethod
    def _euler_degrees(wxyz) -> np.ndarray:
        radians = tf.euler_from_quaternion(wxyz, axes="sxyz")
        return np.round(np.degrees(radians), 3)
