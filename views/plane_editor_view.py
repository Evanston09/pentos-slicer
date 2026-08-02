from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from trimesh import transformations as tf
from viser import ClientHandle

from models import PlaneSnapshot
from views.theming import PENTOS_BLUE, PENTOS_ORANGE

PLANE_HALF_SIZE = 50.0
GIZMO_SCALE = 0.45
GIZMO_LINE_WIDTH = 5.0


def neutral_wxyz() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def neutral_position() -> np.ndarray:
    return np.zeros(3)


@dataclass
class PlaneState:
    pose: Any
    mesh: Any
    normal: Any
    anchor: Any
    gizmo: Any
    gui: dict[str, Any]
    start_position: np.ndarray | None = None
    start_wxyz: np.ndarray | None = None


class PlaneEditorView:
    def __init__(
        self,
        client: ClientHandle,
        on_plane_changed: Callable[[int, np.ndarray, np.ndarray], None],
        on_plane_deleted: Callable[[int], None],
        on_plane_snap_requested: Callable[[int], None],
        gui_container: Any | None = None,
        scene_prefix: str = "/planes",
    ):
        self.client = client
        self.on_plane_changed = on_plane_changed
        self.on_plane_deleted = on_plane_deleted
        self.on_plane_snap_requested = on_plane_snap_requested
        self.gui_container = gui_container
        self.scene_prefix = scene_prefix
        self.planes: dict[int, PlaneState] = {}
        self.synching_gui = False

    def add_plane(self, plane: PlaneSnapshot) -> None:
        if plane.plane_id is None:
            raise ValueError("PlaneSnapshot requires a plane_id")
        plane_id = plane.plane_id

        pose = self.client.scene.add_frame(
            f"{self.scene_prefix}/{plane_id}/pose",
            position=plane.position,
            wxyz=self._normalize_quaternion(plane.wxyz),
        )
        anchor = self.client.scene.add_frame(
            f"{self.scene_prefix}/{plane_id}/gizmo_anchor",
            position=pose.position,
        )
        gizmo = self.client.scene.add_transform_controls(
            f"{self.scene_prefix}/{plane_id}/gizmo_anchor/controls",
            scale=PLANE_HALF_SIZE * GIZMO_SCALE,
            line_width=GIZMO_LINE_WIDTH,
            depth_test=False,
        )

        half = PLANE_HALF_SIZE
        mesh = self.client.scene.add_mesh_simple(
            f"{self.scene_prefix}/{plane_id}/pose/mesh",
            vertices=np.array(
                [
                    [-half, -half, 0.0],
                    [half, -half, 0.0],
                    [half, half, 0.0],
                    [-half, half, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            color=PENTOS_ORANGE,
            opacity=0.35,
            side="double",
        )
        normal = self.client.scene.add_arrows(
            f"{self.scene_prefix}/{plane_id}/pose/normal",
            points=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, half * GIZMO_SCALE]]]),
            colors=PENTOS_BLUE,
            shaft_radius=half * 0.012,
            head_radius=half * 0.04,
            head_length=half * 0.1,
        )

        rx, ry, rz = self._euler_degrees(pose.wxyz)

        if self.gui_container is None:
            raise RuntimeError("Set a GUI container before adding planes")

        with self.gui_container:
            folder = self.client.gui.add_folder(
                f"Plane {plane_id}",
                expand_by_default=True,
            )
        with folder:
            position = self.client.gui.add_vector3(
                "Position", pose.position, step=0.001
            )
            rotation_x = self.client.gui.add_number("Rotation X", rx, step=1.0)
            rotation_y = self.client.gui.add_number("Rotation Y", ry, step=1.0)
            rotation_z = self.client.gui.add_number("Rotation Z", rz, step=1.0)
            snap_button = self.client.gui.add_button("Snap to Face")
            delete_button = self.client.gui.add_button("Delete Plane")

        self.planes[plane_id] = PlaneState(
            pose=pose,
            mesh=mesh,
            normal=normal,
            anchor=anchor,
            gizmo=gizmo,
            gui={
                "folder": folder,
                "position": position,
                "rotation_x": rotation_x,
                "rotation_y": rotation_y,
                "rotation_z": rotation_z,
                "snap_button": snap_button,
                "delete_button": delete_button,
            },
        )

        @gizmo.on_update
        async def _(event):
            self._on_gizmo_update(plane_id, event)

        @position.on_update
        def _(_):
            if not self.synching_gui and plane_id in self.planes:
                self._set_plane_pose(plane_id, position=position.value)
                self._notify_plane_changed(plane_id)

        def update_rotation() -> None:
            if self.synching_gui or plane_id not in self.planes:
                return
            rx = np.radians(rotation_x.value)
            ry = np.radians(rotation_y.value)
            rz = np.radians(rotation_z.value)
            self._set_plane_pose(
                plane_id,
                wxyz=tf.quaternion_from_euler(
                    rx,
                    ry,
                    rz,
                    axes="sxyz",
                ),
            )
            self._notify_plane_changed(plane_id)

        @rotation_x.on_update
        def _(_):
            update_rotation()

        @rotation_y.on_update
        def _(_):
            update_rotation()

        @rotation_z.on_update
        def _(_):
            update_rotation()

        @snap_button.on_click
        def _(_):
            self.on_plane_snap_requested(plane_id)

        @delete_button.on_click
        def _(_):
            self.on_plane_deleted(plane_id)

    def clear(self) -> None:
        for plane_id in list(self.planes):
            self.remove_plane(plane_id)

    def set_visible(self, visible: bool) -> None:
        for state in self.planes.values():
            state.pose.visible = visible
            state.anchor.visible = visible

    def replace_planes(self, planes: list[PlaneSnapshot]) -> None:
        self.clear()
        for plane in planes:
            self.add_plane(plane)

    def set_plane_pose(
        self,
        plane_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        if plane_id not in self.planes:
            return
        self._set_plane_pose(plane_id, position, wxyz)
        self._sync_gui_from_pose(plane_id)

    def remove_plane(self, plane_id: int) -> None:
        del_state = self.planes.pop(plane_id, None)
        if del_state is None:
            return

        self._remove_gui(del_state)
        del_state.gizmo.remove()
        del_state.anchor.remove()
        del_state.normal.remove()
        del_state.mesh.remove()
        del_state.pose.remove()

    def _set_plane_pose(
        self,
        plane_id: int,
        position=None,
        wxyz=None,
    ):
        state = self.planes[plane_id]

        if position is not None:
            state.pose.position = np.array(position)

        if wxyz is not None:
            state.pose.wxyz = self._normalize_quaternion(wxyz)

        self._reset_gizmo(state)

    def _on_gizmo_update(self, plane_id: int, event) -> None:
        state = self.planes.get(plane_id)
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

        self._sync_gui_from_pose(plane_id)
        self._notify_plane_changed(plane_id)

    @staticmethod
    def _reset_gizmo(state: PlaneState) -> None:
        state.anchor.position = state.pose.position
        state.gizmo.position = neutral_position()
        state.gizmo.wxyz = neutral_wxyz()

    def _sync_gui_from_pose(self, plane_id: int) -> None:
        state = self.planes.get(plane_id)
        if state is None:
            return
        self.synching_gui = True
        try:
            rx, ry, rz = self._euler_degrees(state.pose.wxyz)
            state.gui["position"].value = state.pose.position
            state.gui["rotation_x"].value = rx
            state.gui["rotation_y"].value = ry
            state.gui["rotation_z"].value = rz
        finally:
            self.synching_gui = False

    def _notify_plane_changed(self, plane_id: int) -> None:
        state = self.planes.get(plane_id)
        if state is None:
            return
        self.on_plane_changed(
            plane_id,
            np.array(state.pose.position),
            np.array(state.pose.wxyz),
        )

    @staticmethod
    def _remove_gui(state: PlaneState):
        for handle in reversed(state.gui.values()):
            handle.remove()
        state.gui.clear()

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
    def _euler_degrees(wxyz):
        radians = tf.euler_from_quaternion(wxyz, axes="sxyz")
        return np.round(np.degrees(radians), 3)
