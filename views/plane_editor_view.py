from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from viser import ClientHandle

from models import PlaneSnapshot
from views.pose_editor_view import PoseEditorView
from views.theming import PENTOS_BLUE, PENTOS_ORANGE

PLANE_HALF_SIZE = 50.0


@dataclass
class PlaneState:
    mesh: Any
    normal: Any


class PlaneEditorView:
    def __init__(
        self,
        client: ClientHandle,
        on_plane_changed: Callable[[int, np.ndarray, np.ndarray], None],
        on_plane_deleted: Callable[[int], None],
        on_plane_snap_requested: Callable[[int], None],
        scene_prefix: str = "/planes",
    ) -> None:
        self.client = client
        self.pose_editor = PoseEditorView(
            client,
            on_plane_changed,
            on_plane_deleted,
            on_plane_snap_requested,
            item_label="Plane",
            delete_label="Delete Plane",
            scene_prefix=scene_prefix,
            gizmo_size=PLANE_HALF_SIZE,
        )
        self.scene_prefix = scene_prefix
        self.planes: dict[int, PlaneState] = {}

    def add_plane(self, plane: PlaneSnapshot, *, select: bool = True) -> None:
        plane_id = plane.plane_id
        self.pose_editor.add(plane_id, plane.position, plane.wxyz, select=select)
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
            points=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, half * 0.45]]]),
            colors=PENTOS_BLUE,
            shaft_radius=half * 0.012,
            head_radius=half * 0.04,
            head_length=half * 0.1,
        )
        self.planes[plane_id] = PlaneState(mesh, normal)

        @mesh.on_click
        def _(_) -> None:
            self._toggle_select(plane_id)

    def _toggle_select(self, plane_id: int) -> None:
        if plane_id not in self.planes:
            return
        if self.pose_editor.selected_id == plane_id:
            self.pose_editor.clear_selection()
        else:
            self.pose_editor.select(plane_id)

    def clear(self) -> None:
        for plane_id in list(self.planes):
            state = self.planes.pop(plane_id)
            state.normal.remove()
            state.mesh.remove()
        self.pose_editor.clear()

    def set_visible(self, visible: bool) -> None:
        self.pose_editor.set_visible(visible)

    def replace_planes(self, planes: list[PlaneSnapshot]) -> None:
        self.clear()
        for plane in planes:
            self.add_plane(plane, select=False)

    def set_plane_pose(
        self,
        plane_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        self.pose_editor.set_pose(plane_id, position, wxyz)

    def remove_plane(self, plane_id: int) -> None:
        state = self.planes.pop(plane_id, None)
        if state is None:
            return
        state.normal.remove()
        state.mesh.remove()
        self.pose_editor.remove(plane_id)
