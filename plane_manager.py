from dataclasses import dataclass
from typing import Any

import numpy as np
from trimesh import transformations as tf
from viser import ViserServer

ORANGE = (255, 130, 0)
ARROW_BLUE = (47, 153, 238)

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


class PlaneManager:
    def __init__(self, server: ViserServer, gui_container: Any):
        self.server = server
        self.gui_container = gui_container
        self.next_id = 0
        self.planes: dict[int, PlaneState] = {}
        self.synching_gui = False

    def add_plane(self) -> int:
        plane_id = self.next_id
        self.next_id += 1

        pose = self.server.scene.add_frame(
            f"/planes/{plane_id}/pose",
        )
        anchor = self.server.scene.add_frame(
            f"/planes/{plane_id}/gizmo_anchor",
        )
        gizmo = self.server.scene.add_transform_controls(
            f"/planes/{plane_id}/gizmo_anchor/controls",
            scale=PLANE_HALF_SIZE * GIZMO_SCALE,
            line_width=GIZMO_LINE_WIDTH,
            depth_test=False,
        )

        half = PLANE_HALF_SIZE
        mesh = self.server.scene.add_mesh_simple(
            f"/planes/{plane_id}/pose/mesh",
            vertices=np.array(
                [
                    [-half, -half, 0.0],
                    [half, -half, 0.0],
                    [half, half, 0.0],
                    [-half, half, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            color=ORANGE,
            opacity=0.35,
            side="double",
        )
        normal = self.server.scene.add_arrows(
            f"/planes/{plane_id}/pose/normal",
            points=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, half * GIZMO_SCALE]]]),
            colors=ARROW_BLUE,
            shaft_radius=half * 0.012,
            head_radius=half * 0.04,
            head_length=half * 0.1,
        )

        rx, ry, rz = self._euler_degrees(pose.wxyz)

        with self.gui_container:
            folder = self.server.gui.add_folder(
                f"Plane {plane_id}",
                expand_by_default=True,
            )
        with folder:
            position = self.server.gui.add_vector3(
                "Position", pose.position, step=0.001
            )
            rotation_x = self.server.gui.add_number("Rotation X", rx, step=1.0)
            rotation_y = self.server.gui.add_number("Rotation Y", ry, step=1.0)
            rotation_z = self.server.gui.add_number("Rotation Z", rz, step=1.0)
            delete_button = self.server.gui.add_button("Delete Plane")

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

        def update_rotation() -> None:
            if self.synching_gui or plane_id not in self.planes:
                return
            rx = np.radians(float(rotation_x.value))
            ry = np.radians(float(rotation_y.value))
            rz = np.radians(float(rotation_z.value))
            self._set_plane_pose(
                plane_id,
                wxyz=tf.quaternion_from_euler(
                    rx,
                    ry,
                    rz,
                    axes="sxyz",
                ),
            )

        @rotation_x.on_update
        def _(_):
            update_rotation()

        @rotation_y.on_update
        def _(_):
            update_rotation()

        @rotation_z.on_update
        def _(_):
            update_rotation()

        @delete_button.on_click
        def _(_):
            self.remove_plane(plane_id)

        return plane_id

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

    def get_all_planes(self) -> list[Any]:
        return [state.pose for state in self.planes.values()]

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

    @staticmethod
    def _reset_gizmo(state: PlaneState) -> None:
        state.anchor.position = np.array(state.pose.position)
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
        radians = np.asarray(tf.euler_from_quaternion(wxyz, axes="sxyz"), dtype=float)
        return np.round(np.degrees(radians), 3)
