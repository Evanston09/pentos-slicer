from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh
import viser

from models import AppState, PlaneSnapshot
from views.plane_editor_view import PlaneEditorView
from views.theming import OVERHANG_RED, PENTOS_BLUE

if TYPE_CHECKING:
    from controllers.setup_controller import SetupController

MODEL_GIZMO_LINE_WIDTH = 5.0
MODEL_GIZMO_SCALE = 18.0


class SetupView:
    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        self.controller: SetupController | None = None
        self.model_frame_handle: Any | None = None
        self.model_mesh_handle: Any | None = None
        self.model_overhang_handle: Any | None = None
        self.model_gizmo_handle: Any | None = None
        self.model_faces: np.ndarray | None = None
        self.model_folder: Any | None = None
        self.model_x_position: Any | None = None
        self.model_y_position: Any | None = None
        self.model_z_rotation: Any | None = None
        self.model_reset_button: Any | None = None
        self.syncing_model_controls = False
        self.show_overhangs: Any | None = None
        self.show_overhangs_enabled = True
        self.upload: Any | None = None
        self.status: Any | None = None
        self.planes_folder: Any | None = None
        self.plane_editor = PlaneEditorView(
            self.server,
            self._handle_plane_changed,
            self._handle_plane_deleted,
            scene_prefix="/setup/planes",
        )
        self.add_plane_button: Any | None = None
        self.max_auto_planes: Any | None = None
        self.auto_planes_button: Any | None = None
        self.debug_mode: Any | None = None
        self.export_handle: Any | None = None
        self.slice_button: Any | None = None

    def bind_controller(self, controller: SetupController) -> None:
        self.controller = controller

    def mount(self, state: AppState) -> None:
        self.upload = self.server.gui.add_upload_button(
            "Upload Model/Scene",
            mime_type=".stl,.3mf,.obj,.ply,.pentos",
        )
        self.status = self.server.gui.add_text(
            "Status",
            "No model loaded",
            disabled=True,
        )

        self.model_folder = self.server.gui.add_folder(
            "Model",
            expand_by_default=True,
        )
        with self.model_folder:
            self.model_x_position = self.server.gui.add_number(
                "X Position",
                state.model_xy_position[0],
                step=1.0,
                disabled=state.current_model is None,
            )
            self.model_y_position = self.server.gui.add_number(
                "Y Position",
                state.model_xy_position[1],
                step=1.0,
                disabled=state.current_model is None,
            )
            self.model_z_rotation = self.server.gui.add_number(
                "Rotation Z",
                state.model_z_degrees,
                step=1.0,
                disabled=state.current_model is None,
            )
            self.model_reset_button = self.server.gui.add_button(
                "Reset Placement",
                disabled=state.current_model is None,
            )
            self.show_overhangs = self.server.gui.add_checkbox(
                "Show Overhangs",
                self.show_overhangs_enabled,
                disabled=state.current_model is None,
            )

        self.planes_folder = self.server.gui.add_folder(
            "Planes",
            expand_by_default=True,
        )
        self.plane_editor.gui_container = self.planes_folder
        with self.planes_folder:
            self.add_plane_button = self.server.gui.add_button(
                "Add Plane",
                icon=viser.Icon.SQUARES_DIAGONAL,
            )
            self.max_auto_planes = self.server.gui.add_number(
                "Max Auto Planes",
                2,
                min=0,
                max=6,
                step=1,
                disabled=state.current_model is None,
            )
            self.auto_planes_button = self.server.gui.add_button(
                "Auto Planes",
                disabled=state.current_model is None,
            )
        self.debug_mode = self.server.gui.add_checkbox(
            "Debug Mode",
            state.debug_mode,
        )
        self.export_handle = self.server.gui.add_button(
            "Export Scene",
            icon=viser.Icon.PACKAGE_EXPORT,
            disabled=state.current_model is None,
        )
        self.slice_button = self.server.gui.add_button(
            "Slice",
            icon=viser.Icon.CLOUD_COMPUTING,
        )

        @self.upload.on_upload
        def _(event) -> None:
            if self.controller is None:
                return
            uploaded = event.target.value
            self.controller.handle_upload(uploaded.name, uploaded.content)

        @self.model_x_position.on_update
        def _(_) -> None:
            self._handle_model_placement_input()

        @self.model_y_position.on_update
        def _(_) -> None:
            self._handle_model_placement_input()

        @self.model_z_rotation.on_update
        def _(_) -> None:
            self._handle_model_placement_input()

        @self.model_reset_button.on_click
        def _(_) -> None:
            if self.controller is not None:
                self.controller.reset_model_placement()

        @self.show_overhangs.on_update
        def _(_) -> None:
            self.show_overhangs_enabled = self.show_overhangs.value
            if self.show_overhangs_enabled and self.controller is not None:
                self.controller.refresh_overhang_preview()
            else:
                self.show_full_model()

        @self.add_plane_button.on_click
        def _(_) -> None:
            if self.controller is not None:
                self.controller.add_plane()

        @self.auto_planes_button.on_click
        def _(_) -> None:
            if self.controller is not None and self.max_auto_planes is not None:
                self.controller.select_auto_planes(
                    int(round(self.max_auto_planes.value))
                )

        @self.debug_mode.on_update
        def _(_) -> None:
            if self.controller is not None:
                self.controller.set_debug_mode(self.debug_mode.value)

        @self.export_handle.on_click
        def _(event) -> None:
            if self.controller is None:
                return
            download = self.controller.export_scene()
            if download is None:
                return
            filename, content = download
            assert event.client is not None
            event.client.send_file_download(
                filename,
                content,
                save_immediately=True,
            )

        @self.slice_button.on_click
        def _(_) -> None:
            if self.controller is not None:
                self.controller.slice_model()

    def unmount(self) -> None:
        if self.upload is None:
            return

        self.plane_editor.clear()
        self.plane_editor.gui_container = None
        self.clear_model_scene()

        for handle in (
            self.slice_button,
            self.export_handle,
            self.debug_mode,
            self.auto_planes_button,
            self.max_auto_planes,
            self.add_plane_button,
            self.planes_folder,
            self.model_reset_button,
            self.show_overhangs,
            self.model_z_rotation,
            self.model_y_position,
            self.model_x_position,
            self.model_folder,
            self.status,
            self.upload,
        ):
            if handle is not None:
                handle.remove()

        self.upload = None
        self.status = None
        self.planes_folder = None
        self.model_folder = None
        self.show_overhangs = None
        self.model_x_position = None
        self.model_y_position = None
        self.model_z_rotation = None
        self.model_reset_button = None
        self.add_plane_button = None
        self.max_auto_planes = None
        self.auto_planes_button = None
        self.debug_mode = None
        self.export_handle = None
        self.slice_button = None

    def set_status(self, message: str) -> None:
        if self.status is not None:
            self.status.value = message

    def show_mesh(
        self,
        mesh: trimesh.Trimesh,
        center: np.ndarray,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        self.clear_model_scene()
        self.model_faces = mesh.faces
        self.model_frame_handle = self.server.scene.add_frame(
            "/setup/model",
            show_axes=False,
            position=position,
            wxyz=wxyz,
        )
        self.model_mesh_handle = self.server.scene.add_mesh_simple(
            "/setup/model/mesh",
            vertices=mesh.vertices - center,
            faces=mesh.faces,
            color=PENTOS_BLUE,
            opacity=0.45,
            side="double",
        )
        self.model_overhang_handle = self.server.scene.add_mesh_simple(
            "/setup/model/overhangs",
            vertices=mesh.vertices - center,
            faces=np.empty((0, 3)),
            color=OVERHANG_RED,
            opacity=0.85,
            side="double",
        )
        self.model_gizmo_handle = self.server.scene.add_transform_controls(
            "/setup/model_controls",
            scale=MODEL_GIZMO_SCALE,
            line_width=MODEL_GIZMO_LINE_WIDTH,
            active_axes=(True, True, False),
            disable_rotations=True,
            depth_test=False,
            position=position,
        )

        @self.model_gizmo_handle.on_update
        def _(_) -> None:
            if self.controller is not None and self.model_gizmo_handle is not None:
                self.controller.set_model_placement(
                    [
                        self.model_gizmo_handle.position[0],
                        self.model_gizmo_handle.position[1],
                    ]
                )

        self.set_model_controls_enabled(True)

    def clear_model_scene(self) -> None:
        for handle in (
            self.model_gizmo_handle,
            self.model_overhang_handle,
            self.model_mesh_handle,
            self.model_frame_handle,
        ):
            if handle is not None:
                handle.remove()

        self.model_gizmo_handle = None
        self.model_overhang_handle = None
        self.model_mesh_handle = None
        self.model_frame_handle = None
        self.model_faces = None

    def set_model_controls_enabled(self, enabled: bool) -> None:
        for handle in (
            self.model_x_position,
            self.model_y_position,
            self.model_z_rotation,
            self.model_reset_button,
            self.export_handle,
            self.show_overhangs,
            self.max_auto_planes,
            self.auto_planes_button,
        ):
            if handle is not None:
                handle.disabled = not enabled

    def update_model_placement(
        self,
        xy_position: list[float],
        z_degrees: float,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        if self.model_frame_handle is not None:
            self.model_frame_handle.position = position
            self.model_frame_handle.wxyz = wxyz
        if self.model_gizmo_handle is not None:
            self.model_gizmo_handle.position = position
        self._sync_model_controls(xy_position, z_degrees)

    def show_overhang_faces(self, overhang_mask: np.ndarray) -> None:
        if (
            not self.show_overhangs_enabled
            or self.model_faces is None
            or self.model_mesh_handle is None
            or self.model_overhang_handle is None
        ):
            self.show_full_model()
            return

        self.model_mesh_handle.faces = self.model_faces[~overhang_mask]
        self.model_overhang_handle.faces = self.model_faces[overhang_mask]
        self.model_overhang_handle.visible = bool(np.any(overhang_mask))

    def show_full_model(self) -> None:
        if self.model_faces is None or self.model_mesh_handle is None:
            return

        self.model_mesh_handle.faces = self.model_faces
        if self.model_overhang_handle is not None:
            self.model_overhang_handle.visible = False

    def replace_planes(self, planes: list[PlaneSnapshot]) -> None:
        self.plane_editor.replace_planes(planes)

    def add_plane(self, plane: PlaneSnapshot) -> None:
        self.plane_editor.add_plane(plane)

    def remove_plane(self, plane_id: int) -> None:
        self.plane_editor.remove_plane(plane_id)

    def set_debug_mode_value(self, enabled: bool) -> None:
        if self.debug_mode is not None:
            self.debug_mode.value = enabled

    def _sync_model_controls(
        self,
        xy_position: list[float],
        z_degrees: float,
    ) -> None:
        if (
            self.model_x_position is None
            or self.model_y_position is None
            or self.model_z_rotation is None
        ):
            return

        self.syncing_model_controls = True
        try:
            self.model_x_position.value = xy_position[0]
            self.model_y_position.value = xy_position[1]
            self.model_z_rotation.value = z_degrees
        finally:
            self.syncing_model_controls = False

    def _handle_model_placement_input(self) -> None:
        if (
            self.syncing_model_controls
            or self.controller is None
            or self.model_x_position is None
            or self.model_y_position is None
            or self.model_z_rotation is None
        ):
            return

        self.controller.set_model_placement(
            [
                self.model_x_position.value,
                self.model_y_position.value,
            ],
            self.model_z_rotation.value,
        )

    def _handle_plane_changed(
        self,
        plane_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        if self.controller is not None:
            self.controller.update_plane(plane_id, position, wxyz)

    def _handle_plane_deleted(self, plane_id: int) -> None:
        if self.controller is not None:
            self.controller.remove_plane(plane_id)
