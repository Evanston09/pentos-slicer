from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import viser

from models import (
    AppState,
    GuideSurfaceSnapshot,
    PlaneSnapshot,
)
from views.guide_surface_editor_view import GuideSurfaceEditorView
from views.plane_editor_view import PlaneEditorView
from views.theming import OVERHANG_RED, PENTOS_BLUE

if TYPE_CHECKING:
    from controllers.setup_controller import SetupController

MODEL_GIZMO_LINE_WIDTH = 5.0
MODEL_GIZMO_SCALE = 18.0


@dataclass(frozen=True)
class SetupControls:
    upload: viser.GuiUploadButtonHandle
    status: viser.GuiTextHandle
    model_folder: viser.GuiFolderHandle
    model_x_position: viser.GuiNumberHandle[float]
    model_y_position: viser.GuiNumberHandle[float]
    model_z_rotation: viser.GuiNumberHandle[float]
    model_reset_button: viser.GuiButtonHandle
    show_overhangs: viser.GuiCheckboxHandle
    slicing_mode: viser.GuiButtonGroupHandle
    planes_folder: viser.GuiFolderHandle
    add_plane_button: viser.GuiButtonHandle
    max_auto_planes: viser.GuiNumberHandle[int]
    auto_planes_button: viser.GuiButtonHandle
    nonplanar_folder: viser.GuiFolderHandle
    add_guide_button: viser.GuiButtonHandle
    show_tweens: viser.GuiCheckboxHandle
    tween_surface_count: viser.GuiNumberHandle[int]
    debug_mode: viser.GuiCheckboxHandle
    export_button: viser.GuiButtonHandle
    slice_button: viser.GuiButtonHandle


@dataclass(frozen=True)
class ModelScene:
    frame: viser.FrameHandle
    mesh: viser.MeshHandle
    overhang: viser.MeshHandle
    gizmo: viser.TransformControlsHandle
    vertices: np.ndarray
    faces: np.ndarray


class SetupView:
    def __init__(self, client: viser.ClientHandle) -> None:
        self.client = client
        self.controller: SetupController
        self.controls: SetupControls | None = None
        self.model_scene: ModelScene | None = None
        self.syncing_model_controls = False
        self.show_overhangs_enabled = True
        self.plane_editor = PlaneEditorView(
            self.client,
            self._handle_plane_changed,
            self._handle_plane_deleted,
            self._arm_plane_snap,
            scene_prefix="/setup/planes",
        )
        self.guide_surface_editor = GuideSurfaceEditorView(
            self.client,
            self._handle_guide_surface_changed,
            self._handle_guide_surface_deleted,
            self._arm_guide_surface_snap,
            scene_prefix="/setup/guides",
        )
        self.armed_snap_target: tuple[str, int] | None = None

    def bind_controller(self, controller: SetupController) -> None:
        self.controller = controller

    def mount(self, state: AppState) -> None:
        self.plane_editor.set_visible(True)
        self.guide_surface_editor.set_visible(False)
        upload = self.client.gui.add_upload_button(
            "Upload Model/Scene",
            mime_type=".stl,.3mf,.obj,.ply,.pentos",
        )
        status = self.client.gui.add_text(
            "Status",
            "No model loaded",
            disabled=True,
        )

        model_folder = self.client.gui.add_folder(
            "Model",
            expand_by_default=True,
        )
        with model_folder:
            model_x_position = self.client.gui.add_number(
                "X Position",
                state.model_xy_position[0],
                step=1.0,
                disabled=state.current_model is None,
            )
            model_y_position = self.client.gui.add_number(
                "Y Position",
                state.model_xy_position[1],
                step=1.0,
                disabled=state.current_model is None,
            )
            model_z_rotation = self.client.gui.add_number(
                "Rotation Z",
                state.model_z_degrees,
                step=1.0,
                disabled=state.current_model is None,
            )
            model_reset_button = self.client.gui.add_button(
                "Reset Placement",
                disabled=state.current_model is None,
            )
            show_overhangs = self.client.gui.add_checkbox(
                "Show Overhangs",
                self.show_overhangs_enabled,
                disabled=state.current_model is None,
            )

        slicing_mode = self.client.gui.add_button_group(
            "Slicing Mode",
            ("Multiplanar", "Nonplanar"),
        )

        planes_folder = self.client.gui.add_folder(
            "Multiplanar",
            expand_by_default=True,
        )
        self.plane_editor.pose_editor.gui_container = planes_folder
        with planes_folder:
            add_plane_button = self.client.gui.add_button(
                "Add Plane",
                icon=viser.Icon.SQUARES_DIAGONAL,
            )
            max_auto_planes = self.client.gui.add_number(
                "Max Auto Planes",
                2,
                min=0,
                max=6,
                step=1,
                disabled=state.current_model is None,
            )
            auto_planes_button = self.client.gui.add_button(
                "Auto Planes",
                disabled=state.current_model is None,
            )
        nonplanar_folder = self.client.gui.add_folder(
            "Nonplanar",
            expand_by_default=True,
            visible=False,
        )
        self.guide_surface_editor.pose_editor.gui_container = nonplanar_folder
        with nonplanar_folder:
            add_guide_button = self.client.gui.add_button(
                "Add Guide Surface",
                icon=viser.Icon.SQUARES_DIAGONAL,
            )
            show_tweens = self.client.gui.add_checkbox(
                "Show Tweens", self.guide_surface_editor.tweens_visible
            )
            tween_surface_count = self.client.gui.add_number(
                "Tween Surfaces",
                self.controller.nonplanar.tween_surface_count,
                min=1,
                max=50,
                step=1,
            )
        debug_mode = self.client.gui.add_checkbox(
            "Debug Mode",
            state.debug_mode,
        )
        export_button = self.client.gui.add_button(
            "Export Scene",
            icon=viser.Icon.PACKAGE_EXPORT,
            disabled=state.current_model is None,
        )
        slice_button = self.client.gui.add_button(
            "Slice",
            icon=viser.Icon.CLOUD_COMPUTING,
        )
        controls = SetupControls(
            upload=upload,
            status=status,
            model_folder=model_folder,
            model_x_position=model_x_position,
            model_y_position=model_y_position,
            model_z_rotation=model_z_rotation,
            model_reset_button=model_reset_button,
            show_overhangs=show_overhangs,
            slicing_mode=slicing_mode,
            planes_folder=planes_folder,
            add_plane_button=add_plane_button,
            max_auto_planes=max_auto_planes,
            auto_planes_button=auto_planes_button,
            nonplanar_folder=nonplanar_folder,
            add_guide_button=add_guide_button,
            show_tweens=show_tweens,
            tween_surface_count=tween_surface_count,
            debug_mode=debug_mode,
            export_button=export_button,
            slice_button=slice_button,
        )
        self.controls = controls

        @controls.upload.on_upload
        def _(event) -> None:
            uploaded = event.target.value
            self.controller.handle_upload(uploaded.name, uploaded.content)

        @controls.model_x_position.on_update
        def _(_) -> None:
            self._handle_model_placement_input()

        @controls.model_y_position.on_update
        def _(_) -> None:
            self._handle_model_placement_input()

        @controls.model_z_rotation.on_update
        def _(_) -> None:
            self._handle_model_placement_input()

        @controls.model_reset_button.on_click
        def _(_) -> None:
            self.controller.reset_model_placement()

        @controls.show_overhangs.on_update
        def _(_) -> None:
            self.show_overhangs_enabled = controls.show_overhangs.value
            if self.show_overhangs_enabled:
                self.controller.refresh_overhang_preview()
            else:
                self.show_full_model()

        @controls.slicing_mode.on_click
        def _(_) -> None:
            self.set_slicing_mode(controls.slicing_mode.value.lower())

        @controls.add_plane_button.on_click
        def _(_) -> None:
            self.controller.add_plane()

        @controls.auto_planes_button.on_click
        def _(_) -> None:
            self.controller.select_auto_planes(
                int(round(controls.max_auto_planes.value))
            )

        @controls.add_guide_button.on_click
        def _(_) -> None:
            self.controller.nonplanar.add_guide()

        @controls.show_tweens.on_update
        def _(_) -> None:
            self.guide_surface_editor.set_tweens_visible(controls.show_tweens.value)

        @controls.tween_surface_count.on_update
        def _(_) -> None:
            self.controller.nonplanar.set_tween_surface_count(
                int(round(controls.tween_surface_count.value))
            )

        @controls.debug_mode.on_update
        def _(_) -> None:
            self.controller.set_debug_mode(controls.debug_mode.value)

        @controls.export_button.on_click
        def _(event) -> None:
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

        @controls.slice_button.on_click
        def _(_) -> None:
            self.controller.slice_model()

    def unmount(self) -> None:
        if self.controls is None:
            return

        controls = self.controls
        self.client.scene.remove_click_callback(self._handle_scene_click)
        self.armed_snap_target = None
        self.plane_editor.clear()
        self.plane_editor.pose_editor.gui_container = None
        self.guide_surface_editor.clear()
        self.guide_surface_editor.pose_editor.gui_container = None
        self.clear_model_scene()

        for handle in (
            controls.slice_button,
            controls.export_button,
            controls.debug_mode,
            controls.nonplanar_folder,
            controls.planes_folder,
            controls.model_folder,
            controls.slicing_mode,
            controls.status,
            controls.upload,
        ):
            handle.remove()

        self.controls = None

    def set_status(self, message: str) -> None:
        self._mounted().status.value = message

    def set_slice_enabled(self, enabled: bool) -> None:
        self._mounted().slice_button.disabled = not enabled

    def set_slicing_mode(self, mode: str) -> None:
        controls = self._mounted()
        multiplanar = mode == "multiplanar"
        controls.planes_folder.visible = multiplanar
        controls.debug_mode.visible = multiplanar
        controls.slice_button.visible = multiplanar
        controls.nonplanar_folder.visible = not multiplanar
        self.plane_editor.set_visible(multiplanar)
        self.guide_surface_editor.set_visible(not multiplanar)

    def show_mesh(
        self,
        mesh: trimesh.Trimesh,
        center: np.ndarray,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        self.clear_model_scene()
        vertices = mesh.vertices - center
        faces = mesh.faces
        frame = self.client.scene.add_frame(
            "/setup/model",
            show_axes=False,
            position=position,
            wxyz=wxyz,
        )
        model_mesh = self.client.scene.add_mesh_simple(
            "/setup/model/mesh",
            vertices=vertices,
            faces=mesh.faces,
            color=PENTOS_BLUE,
            opacity=0.45,
            side="double",
        )
        overhang = self.client.scene.add_mesh_simple(
            "/setup/model/overhangs",
            vertices=vertices,
            faces=np.empty((0, 3)),
            color=OVERHANG_RED,
            opacity=0.85,
            side="double",
        )
        gizmo = self.client.scene.add_transform_controls(
            "/setup/model_controls",
            scale=MODEL_GIZMO_SCALE,
            line_width=MODEL_GIZMO_LINE_WIDTH,
            active_axes=(True, True, False),
            disable_rotations=True,
            depth_test=False,
            position=position,
        )

        self.model_scene = ModelScene(
            frame=frame,
            mesh=model_mesh,
            overhang=overhang,
            gizmo=gizmo,
            vertices=vertices,
            faces=faces,
        )

        @gizmo.on_update
        def _(_) -> None:
            self.controller.set_model_placement([gizmo.position[0], gizmo.position[1]])

        self.set_model_controls_enabled(True)

    def clear_model_scene(self) -> None:
        if self.model_scene is None:
            return

        for handle in (
            self.model_scene.gizmo,
            self.model_scene.overhang,
            self.model_scene.mesh,
            self.model_scene.frame,
        ):
            handle.remove()
        self.model_scene = None

    def set_model_controls_enabled(self, enabled: bool) -> None:
        controls = self._mounted()
        for handle in (
            controls.model_x_position,
            controls.model_y_position,
            controls.model_z_rotation,
            controls.model_reset_button,
            controls.export_button,
            controls.show_overhangs,
            controls.max_auto_planes,
            controls.auto_planes_button,
        ):
            handle.disabled = not enabled

    def update_model_placement(
        self,
        xy_position: list[float],
        z_degrees: float,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        scene = self._model_scene()
        scene.frame.position = position
        scene.frame.wxyz = wxyz
        scene.gizmo.position = position
        self._sync_model_controls(xy_position, z_degrees)

    def show_overhang_faces(
        self,
        mesh: trimesh.Trimesh,
        overhang_mask: np.ndarray,
    ) -> None:
        if not self.show_overhangs_enabled:
            self.show_full_model()
            return

        scene = self._model_scene()
        rotation = trimesh.transformations.quaternion_matrix(scene.frame.wxyz)[:3, :3]
        vertices = np.asarray(
            (mesh.vertices - scene.frame.position) @ rotation,
            dtype=np.float32,
        )
        scene.mesh.vertices = vertices
        scene.mesh.faces = mesh.faces[~overhang_mask]
        scene.overhang.vertices = vertices
        scene.overhang.faces = mesh.faces[overhang_mask]
        scene.overhang.visible = bool(np.any(overhang_mask))

    def set_model_out_of_bounds(self, out_of_bounds: bool) -> None:
        self._model_scene().mesh.color = OVERHANG_RED if out_of_bounds else PENTOS_BLUE

    def show_full_model(self) -> None:
        if self.model_scene is None:
            return

        self.model_scene.mesh.vertices = self.model_scene.vertices
        self.model_scene.mesh.faces = self.model_scene.faces
        self.model_scene.overhang.visible = False

    def replace_planes(self, planes: list[PlaneSnapshot]) -> None:
        self.plane_editor.replace_planes(planes)

    def add_plane(self, plane: PlaneSnapshot) -> None:
        self.plane_editor.add_plane(plane)

    def remove_plane(self, plane_id: int) -> None:
        self._disarm_snap(("plane", plane_id))
        self.plane_editor.remove_plane(plane_id)

    def set_plane_pose(
        self,
        plane_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        self.plane_editor.set_plane_pose(plane_id, position, wxyz)

    def set_debug_mode_value(self, enabled: bool) -> None:
        self._mounted().debug_mode.value = enabled

    def replace_guide_surfaces(
        self,
        guides: list[GuideSurfaceSnapshot],
    ) -> None:
        self.guide_surface_editor.replace_guides(guides)

    def add_guide_surface(self, guide: GuideSurfaceSnapshot) -> None:
        self.guide_surface_editor.add_guide(guide)

    def remove_guide_surface(self, guide_id: int) -> None:
        self._disarm_snap(("guide", guide_id))
        self.guide_surface_editor.remove_guide(guide_id)

    def set_guide_surface_pose(
        self,
        guide_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        self.guide_surface_editor.set_guide_pose(guide_id, position, wxyz)

    def set_guide_surface_mesh(
        self,
        guide_id: int,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        self.guide_surface_editor.set_guide_mesh(guide_id, vertices, faces)

    def replace_tween_surfaces(
        self,
        meshes: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.guide_surface_editor.replace_tweens(meshes)

    def _sync_model_controls(
        self,
        xy_position: list[float],
        z_degrees: float,
    ) -> None:
        controls = self._mounted()
        self.syncing_model_controls = True
        try:
            controls.model_x_position.value = xy_position[0]
            controls.model_y_position.value = xy_position[1]
            controls.model_z_rotation.value = z_degrees
        finally:
            self.syncing_model_controls = False

    def _mounted(self) -> SetupControls:
        if self.controls is None:
            raise RuntimeError("Setup view is not mounted")
        return self.controls

    def _model_scene(self) -> ModelScene:
        if self.model_scene is None:
            raise RuntimeError("No model is displayed")
        return self.model_scene

    def _handle_model_placement_input(self) -> None:
        if self.syncing_model_controls:
            return

        controls = self._mounted()
        self.controller.set_model_placement(
            [
                controls.model_x_position.value,
                controls.model_y_position.value,
            ],
            controls.model_z_rotation.value,
        )

    def _handle_plane_changed(
        self,
        plane_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        self.controller.update_plane(plane_id, position, wxyz)

    def _handle_plane_deleted(self, plane_id: int) -> None:
        self.controller.remove_plane(plane_id)

    def _handle_guide_surface_changed(
        self,
        guide_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
        bend_x: float,
        bend_y: float,
    ) -> None:
        self.controller.nonplanar.update_guide(guide_id, position, wxyz, bend_x, bend_y)

    def _handle_guide_surface_deleted(self, guide_id: int) -> None:
        self.controller.nonplanar.remove_guide(guide_id)

    def _arm_plane_snap(self, plane_id: int) -> None:
        self._arm_snap("plane", plane_id)

    def _arm_guide_surface_snap(self, guide_id: int) -> None:
        self._arm_snap("guide", guide_id)

    def _arm_snap(self, kind: str, item_id: int) -> None:
        self.client.scene.remove_click_callback(self._handle_scene_click)
        self.client.scene.on_click()(self._handle_scene_click)
        self.armed_snap_target = (kind, item_id)
        self.set_status(f"{kind.title()} {item_id}: click a model face to snap")

    def _disarm_snap(self, target: tuple[str, int]) -> None:
        if self.armed_snap_target == target:
            self.armed_snap_target = None
            self.client.scene.remove_click_callback(self._handle_scene_click)

    def _handle_scene_click(self, event) -> None:
        target = self.armed_snap_target
        if target is None:
            return
        kind, item_id = target
        ray_origin = np.array(event.ray_origin)
        ray_direction = np.array(event.ray_direction)
        if kind == "plane":
            snapped = self.controller.snap_plane_to_face(
                item_id,
                ray_origin,
                ray_direction,
            )
        else:
            snapped = self.controller.nonplanar.snap_guide_to_face(
                item_id, ray_origin, ray_direction
            )
        if not snapped:
            return
        self.armed_snap_target = None
        self.client.scene.remove_click_callback(self._handle_scene_click)
        self.set_status(f"{kind.title()} {item_id} snapped to model face")
