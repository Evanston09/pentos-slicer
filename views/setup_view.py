from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh
import viser
from trimesh import transformations as tf

from app_state import AppState
from machine import BUILD_PLATE_CENTER
from plane_manager import PlaneManager
from slice_tools import Slicer
from theming import PENTOS_BLUE

MODEL_GIZMO_LINE_WIDTH = 5.0
MODEL_GIZMO_SCALE = 18.0


def normalize_mesh_units(mesh: trimesh.Trimesh) -> None:
    if mesh.units is None:
        mesh.units = "m" if float(mesh.extents.max()) < 1.0 else "mm"

    if mesh.units != "mm":
        mesh.convert_units("mm")


def load_model(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path)
    normalize_mesh_units(mesh)

    lower, upper = mesh.bounds
    mesh_center_xy = (lower[:2] + upper[:2]) / 2.0
    mesh.apply_translation(
        [
            BUILD_PLATE_CENTER[0] - mesh_center_xy[0],
            BUILD_PLATE_CENTER[1] - mesh_center_xy[1],
            -lower[2],
        ]
    )
    return mesh


class SetupView:
    def __init__(
        self,
        server: viser.ViserServer,
        state: AppState,
        slicer: Slicer,
        show_preview: Callable[[], None],
    ) -> None:
        self.server = server
        self.state = state
        self.slicer = slicer
        self.show_preview = show_preview
        self.model_frame_handle: Any | None = None
        self.model_mesh_handle: Any | None = None
        self.model_gizmo_handle: Any | None = None
        self.model_folder: Any | None = None
        self.model_x_offset: Any | None = None
        self.model_y_offset: Any | None = None
        self.model_z_rotation: Any | None = None
        self.model_reset_button: Any | None = None
        self.syncing_model_controls = False
        self.upload: Any | None = None
        self.status: Any | None = None
        self.planes_folder: Any | None = None
        self.plane_manager = PlaneManager(
            self.server,
            scene_prefix="/setup/planes",
        )
        self.add_plane_button: Any | None = None
        self.debug_mode: Any | None = None
        self.slice_button: Any | None = None

    def mount(self) -> None:
        self.upload = self.server.gui.add_upload_button(
            "Upload Model",
            mime_type=".stl,.3mf,.obj,.ply",
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
            self.model_x_offset = self.server.gui.add_number(
                "X Offset",
                float(self.state.model_xy_offset[0]),
                step=1.0,
                disabled=self.state.current_model is None,
            )
            self.model_y_offset = self.server.gui.add_number(
                "Y Offset",
                float(self.state.model_xy_offset[1]),
                step=1.0,
                disabled=self.state.current_model is None,
            )
            self.model_z_rotation = self.server.gui.add_number(
                "Rotation Z",
                float(self.state.model_z_degrees),
                step=1.0,
                disabled=self.state.current_model is None,
            )
            self.model_reset_button = self.server.gui.add_button(
                "Reset Placement",
                disabled=self.state.current_model is None,
            )

        self.planes_folder = self.server.gui.add_folder(
            "Planes",
            expand_by_default=True,
        )
        self.plane_manager.gui_container = self.planes_folder
        with self.planes_folder:
            self.add_plane_button = self.server.gui.add_button(
                "Add Plane",
                icon=viser.Icon.SQUARES_DIAGONAL,
            )
        self.debug_mode = self.server.gui.add_checkbox(
            "Debug Mode",
            self.state.debug_mode,
        )
        self.slice_button = self.server.gui.add_button(
            "Slice",
            icon=viser.Icon.CLOUD_COMPUTING,
        )

        for snapshot in self.state.plane_snapshots:
            self.plane_manager.add_plane(snapshot.position, snapshot.wxyz)

        if self.state.current_model is not None:
            mesh, _ = self.state.current_model
            self.show_mesh(mesh)

        @self.upload.on_upload
        def _(event) -> None:
            self.handle_upload(event)

        @self.model_x_offset.on_update
        def _(_) -> None:
            self.handle_model_placement_input()

        @self.model_y_offset.on_update
        def _(_) -> None:
            self.handle_model_placement_input()

        @self.model_z_rotation.on_update
        def _(_) -> None:
            self.handle_model_placement_input()

        @self.model_reset_button.on_click
        def _(_) -> None:
            self.set_model_placement([0.0, 0.0], 0.0)

        @self.add_plane_button.on_click
        def _(_) -> None:
            self.plane_manager.add_plane()

        @self.debug_mode.on_update
        def _(_) -> None:
            self.state.debug_mode = self.debug_mode.value

        @self.slice_button.on_click
        def _(_) -> None:
            self.handle_slice()

    def unmount(self) -> None:
        if self.upload is None:
            return

        self.state.plane_snapshots = self.plane_manager.snapshot_planes()
        self.plane_manager.clear()
        self.plane_manager.gui_container = None
        self.clear_model_scene()

        for handle in (
            self.slice_button,
            self.debug_mode,
            self.add_plane_button,
            self.planes_folder,
            self.model_reset_button,
            self.model_z_rotation,
            self.model_y_offset,
            self.model_x_offset,
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
        self.model_x_offset = None
        self.model_y_offset = None
        self.model_z_rotation = None
        self.model_reset_button = None
        self.add_plane_button = None
        self.debug_mode = None
        self.slice_button = None

    def show_mesh(self, mesh: trimesh.Trimesh) -> None:
        self.clear_model_scene()

        center = self.model_center(mesh)
        position = self.model_frame_position(mesh)
        wxyz = self.model_wxyz()

        self.model_frame_handle = self.server.scene.add_frame(
            "/setup/model",
            show_axes=False,
            position=position,
            wxyz=wxyz,
        )
        self.model_mesh_handle = self.server.scene.add_mesh_simple(
            "/setup/model/mesh",
            vertices=np.asarray(mesh.vertices - center),
            faces=np.asarray(mesh.faces),
            color=PENTOS_BLUE,
            opacity=0.45,
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
            self.handle_model_gizmo_update()

        self.set_model_controls_enabled(True)
        self.sync_model_controls()
        if self.status is not None and self.state.current_model is not None:
            self.status.value = f"Loaded {self.state.current_model[1]}"

    def clear_model_scene(self) -> None:
        for handle in (
            self.model_gizmo_handle,
            self.model_mesh_handle,
            self.model_frame_handle,
        ):
            if handle is not None:
                handle.remove()

        self.model_gizmo_handle = None
        self.model_mesh_handle = None
        self.model_frame_handle = None

    def set_model_controls_enabled(self, enabled: bool) -> None:
        for handle in (
            self.model_x_offset,
            self.model_y_offset,
            self.model_z_rotation,
            self.model_reset_button,
        ):
            if handle is not None:
                handle.disabled = not enabled

    def model_center(self, mesh: trimesh.Trimesh) -> np.ndarray:
        return mesh.bounds.mean(axis=0)

    def model_wxyz(self) -> np.ndarray:
        return np.asarray(
            tf.quaternion_from_euler(
                0.0,
                0.0,
                np.radians(float(self.state.model_z_degrees)),
                axes="sxyz",
            ),
            dtype=float,
        )

    def model_frame_position(self, mesh: trimesh.Trimesh) -> np.ndarray:
        center = self.model_center(mesh)
        offset = np.asarray(self.state.model_xy_offset, dtype=float)
        return np.array([center[0] + offset[0], center[1] + offset[1], center[2]])

    def set_model_placement(
        self,
        offset: list[float] | np.ndarray | None = None,
        z_degrees: float | None = None,
    ) -> None:
        if self.state.current_model is None:
            return

        mesh, _ = self.state.current_model
        if offset is not None:
            self.state.model_xy_offset = list(offset)
        if z_degrees is not None:
            self.state.model_z_degrees = float(z_degrees)

        position = self.model_frame_position(mesh)
        if self.model_frame_handle is not None:
            self.model_frame_handle.position = position
            self.model_frame_handle.wxyz = self.model_wxyz()
        if self.model_gizmo_handle is not None:
            self.model_gizmo_handle.position = position

        self.sync_model_controls()

    def sync_model_controls(self) -> None:
        if (
            self.model_x_offset is None
            or self.model_y_offset is None
            or self.model_z_rotation is None
        ):
            return

        self.syncing_model_controls = True
        try:
            self.model_x_offset.value = float(self.state.model_xy_offset[0])
            self.model_y_offset.value = float(self.state.model_xy_offset[1])
            self.model_z_rotation.value = float(self.state.model_z_degrees)
        finally:
            self.syncing_model_controls = False

    def handle_model_placement_input(self) -> None:
        if (
            self.syncing_model_controls
            or self.model_x_offset is None
            or self.model_y_offset is None
            or self.model_z_rotation is None
        ):
            return

        self.set_model_placement(
            [
                float(self.model_x_offset.value),
                float(self.model_y_offset.value),
            ],
            float(self.model_z_rotation.value),
        )

    def handle_model_gizmo_update(self) -> None:
        if self.state.current_model is None or self.model_gizmo_handle is None:
            return

        mesh, _ = self.state.current_model
        self.set_model_placement(self.model_gizmo_handle.position[:2] - self.model_center(mesh)[:2])

    def handle_upload(self, event) -> None:
        uploaded = event.target.value
        upload_dir = Path("uploaded_models")
        upload_dir.mkdir(exist_ok=True)

        path = upload_dir / uploaded.name
        path.write_bytes(uploaded.content)

        try:
            mesh = load_model(path)
            self.state.current_model = (mesh, path.stem)
            self.state.model_xy_offset = [0.0, 0.0]
            self.state.model_z_degrees = 0.0
            self.state.gcode_path = None
            self.show_mesh(mesh)
            if self.status is not None:
                print(self.status.value)
        except Exception as exc:
            if self.status is not None:
                self.status.value = f"Failed to load {uploaded.name}: {exc}"
                print(self.status.value)

    def handle_slice(self) -> None:
        if self.state.current_model is None:
            if self.status is not None:
                self.status.value = "Load a model before slicing"
            return

        if not self.plane_manager.planes:
            if self.status is not None:
                self.status.value = "Add a plane before slicing"
            return

        planes = self.plane_manager.get_all_planes()
        base_mesh, source_name = self.state.current_model
        mesh = base_mesh.copy()
        if not np.isclose(self.state.model_z_degrees, 0.0):
            mesh.apply_transform(
                tf.rotation_matrix(
                    np.radians(float(self.state.model_z_degrees)),
                    [0.0, 0.0, 1.0],
                    point=self.model_center(base_mesh),
                ),
            )
        offset = np.asarray(self.state.model_xy_offset, dtype=float)
        mesh.apply_translation([offset[0], offset[1], 0.0])

        try:
            debug_mode = bool(self.debug_mode.value) if self.debug_mode else False
            self.state.debug_mode = debug_mode
            if self.status is not None:
                self.status.value = (
                    "Generating debug transition check..."
                    if debug_mode
                    else "Slicing..."
                )
            if debug_mode:
                output_path = self.slicer.debug_transition_check(
                    mesh,
                    planes,
                    source_name,
                )
            else:
                output_path = self.slicer.slice(mesh, planes, source_name)
        except Exception as exc:
            if self.status is not None:
                self.status.value = f"Failed to slice: {exc}"
                print(self.status.value)
            return

        self.state.gcode_path = output_path
        self.show_preview()
