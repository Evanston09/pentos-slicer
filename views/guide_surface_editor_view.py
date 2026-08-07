from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from viser import ClientHandle

from models import GuideSurfaceSnapshot, guide_surface_mesh
from views.pose_editor_view import PoseEditorView, PoseState
from views.theming import PENTOS_ORANGE


@dataclass
class GuideSurfaceState:
    pose: PoseState
    mesh: Any


class GuideSurfaceEditorView:
    def __init__(
        self,
        client: ClientHandle,
        on_guide_changed: Callable[[int, np.ndarray, np.ndarray, float, float], None],
        on_guide_deleted: Callable[[int], None],
        on_guide_snap_requested: Callable[[int], None],
        scene_prefix: str = "/guides",
    ) -> None:
        self.client = client
        self.on_guide_changed = on_guide_changed
        self.scene_prefix = scene_prefix
        self.pose_editor = PoseEditorView(
            client,
            self._handle_pose_changed,
            on_guide_deleted,
            on_guide_snap_requested,
            item_label="Guide",
            delete_label="Delete Guide",
            scene_prefix=scene_prefix,
            gizmo_size=50.0,
        )
        self.pose_editor.visible = False
        self.guides: dict[int, GuideSurfaceState] = {}

    def add_guide(self, guide: GuideSurfaceSnapshot) -> None:
        guide_id = guide.guide_id

        def add_bend_controls() -> dict[str, Any]:
            return {
                "bend_x": self.client.gui.add_number(
                    "Bend X", guide.bend_x, step=0.001
                ),
                "bend_y": self.client.gui.add_number(
                    "Bend Y", guide.bend_y, step=0.001
                ),
            }

        pose = self.pose_editor.add(
            guide_id,
            guide.position,
            guide.wxyz,
            add_bend_controls,
        )
        vertices, faces = guide_surface_mesh(guide.bend_x, guide.bend_y)
        mesh = self.client.scene.add_mesh_simple(
            f"{self.scene_prefix}/{guide_id}/pose/mesh",
            vertices=vertices,
            faces=faces,
            color=PENTOS_ORANGE,
            opacity=0.35,
            side="double",
        )
        self.guides[guide_id] = GuideSurfaceState(pose, mesh)

        def update_bend() -> None:
            state = self.guides.get(guide_id)
            if state is None:
                return
            self.on_guide_changed(
                guide_id,
                np.array(state.pose.pose.position),
                np.array(state.pose.pose.wxyz),
                state.pose.gui["bend_x"].value,
                state.pose.gui["bend_y"].value,
            )

        @pose.gui["bend_x"].on_update
        def _(_) -> None:
            update_bend()

        @pose.gui["bend_y"].on_update
        def _(_) -> None:
            update_bend()

    def clear(self) -> None:
        for guide_id in list(self.guides):
            self.remove_guide(guide_id)

    def replace_guides(self, guides: list[GuideSurfaceSnapshot]) -> None:
        self.clear()
        for guide in guides:
            self.add_guide(guide)

    def remove_guide(self, guide_id: int) -> None:
        state = self.guides.pop(guide_id, None)
        if state is None:
            return
        state.mesh.remove()
        self.pose_editor.remove(guide_id)

    def set_visible(self, visible: bool) -> None:
        self.pose_editor.set_visible(visible)

    def set_guide_pose(
        self,
        guide_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        self.pose_editor.set_pose(guide_id, position, wxyz)

    def set_guide_mesh(
        self,
        guide_id: int,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        state = self.guides.get(guide_id)
        if state is not None:
            state.mesh.vertices = vertices
            state.mesh.faces = faces

    def _handle_pose_changed(
        self,
        guide_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        state = self.guides.get(guide_id)
        if state is not None:
            self.on_guide_changed(
                guide_id,
                position,
                wxyz,
                state.pose.gui["bend_x"].value,
                state.pose.gui["bend_y"].value,
            )
