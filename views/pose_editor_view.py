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
    gui: dict[str, Any]
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
    ) -> None:
        self.client = client
        self.on_changed = on_changed
        self.on_deleted = on_deleted
        self.on_snap_requested = on_snap_requested
        self.item_label = item_label
        self.delete_label = delete_label
        self.scene_prefix = scene_prefix
        self.gizmo_size = gizmo_size
        self.gui_container: Any | None = None
        self.items: dict[int, PoseState] = {}
        self.syncing_gui = False
        self.visible = True

    def add(
        self,
        item_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
        add_extra_controls: Callable[[], dict[str, Any]] | None = None,
    ) -> PoseState:
        if self.gui_container is None:
            raise RuntimeError("Set a GUI container before adding items")

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

        rx, ry, rz = self._euler_degrees(pose.wxyz)
        with self.gui_container:
            folder = self.client.gui.add_folder(
                f"{self.item_label} {item_id}",
                expand_by_default=True,
            )
        with folder:
            position_control = self.client.gui.add_vector3(
                "Position", pose.position, step=0.001
            )
            rotation_x = self.client.gui.add_number("Rotation X", rx, step=1.0)
            rotation_y = self.client.gui.add_number("Rotation Y", ry, step=1.0)
            rotation_z = self.client.gui.add_number("Rotation Z", rz, step=1.0)
            extra_controls = {} if add_extra_controls is None else add_extra_controls()
            snap_button = self.client.gui.add_button("Snap to Face")
            delete_button = self.client.gui.add_button(self.delete_label)

        state = PoseState(
            pose=pose,
            anchor=anchor,
            gizmo=gizmo,
            gui={
                "folder": folder,
                "position": position_control,
                "rotation_x": rotation_x,
                "rotation_y": rotation_y,
                "rotation_z": rotation_z,
                **extra_controls,
                "snap_button": snap_button,
                "delete_button": delete_button,
            },
        )
        self.items[item_id] = state

        @gizmo.on_update
        async def _(event) -> None:
            self._on_gizmo_update(item_id, event)

        @position_control.on_update
        def _(_) -> None:
            if not self.syncing_gui and item_id in self.items:
                self._set_pose(item_id, position=position_control.value)
                self._notify_changed(item_id)

        def update_rotation() -> None:
            if self.syncing_gui or item_id not in self.items:
                return
            self._set_pose(
                item_id,
                wxyz=tf.quaternion_from_euler(
                    np.radians(rotation_x.value),
                    np.radians(rotation_y.value),
                    np.radians(rotation_z.value),
                    axes="sxyz",
                ),
            )
            self._notify_changed(item_id)

        @rotation_x.on_update
        def _(_) -> None:
            update_rotation()

        @rotation_y.on_update
        def _(_) -> None:
            update_rotation()

        @rotation_z.on_update
        def _(_) -> None:
            update_rotation()

        @snap_button.on_click
        def _(_) -> None:
            self.on_snap_requested(item_id)

        @delete_button.on_click
        def _(_) -> None:
            self.on_deleted(item_id)

        return state

    def clear(self) -> None:
        for item_id in list(self.items):
            self.remove(item_id)

    def remove(self, item_id: int) -> None:
        state = self.items.pop(item_id, None)
        if state is None:
            return
        for handle in reversed(state.gui.values()):
            handle.remove()
        state.gui.clear()
        state.gizmo.remove()
        state.anchor.remove()
        state.pose.remove()

    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        for state in self.items.values():
            state.pose.visible = visible
            state.anchor.visible = visible

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
        state = self.items.get(item_id)
        if state is None:
            return
        self.syncing_gui = True
        try:
            rx, ry, rz = self._euler_degrees(state.pose.wxyz)
            state.gui["position"].value = state.pose.position
            state.gui["rotation_x"].value = rx
            state.gui["rotation_y"].value = ry
            state.gui["rotation_z"].value = rz
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
