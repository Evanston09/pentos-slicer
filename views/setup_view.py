
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh
import viser

from app_state import AppState
from machine import BUILD_PLATE_CENTER
from plane_manager import PlaneManager
from slice_tools import Slicer
from theming import PENTOS_BLUE


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
        self.model_handle: Any | None = None
        self.upload: Any | None = None
        self.status: Any | None = None
        self.planes_folder: Any | None = None
        self.plane_manager = PlaneManager(
            self.server,
            scene_prefix="/setup/planes",
        )
        self.add_plane_button: Any | None = None
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
        self.slice_button = self.server.gui.add_button(
            "Slice",
            icon=viser.Icon.CLOUD_COMPUTING,
        )

        for snapshot in self.state.plane_snapshots:
            self.plane_manager.add_plane(snapshot.position, snapshot.wxyz)

        if self.state.current_model is not None:
            mesh, source_name = self.state.current_model
            self.show_mesh(mesh)
            if self.status is not None:
                self.status.value = f"Loaded {source_name}"

        @self.upload.on_upload
        def _(event) -> None:
            self.handle_upload(event)

        @self.add_plane_button.on_click
        def _(_) -> None:
            self.plane_manager.add_plane()

        @self.slice_button.on_click
        def _(_) -> None:
            self.handle_slice()

    def unmount(self) -> None:
        if self.upload is None:
            return

        self.state.plane_snapshots = self.plane_manager.snapshot_planes()
        self.plane_manager.clear()
        self.plane_manager.gui_container = None

        for handle in (
            self.slice_button,
            self.add_plane_button,
            self.planes_folder,
            self.status,
            self.upload,
            self.model_handle
        ):
            if handle is not None:
                handle.remove()

        self.upload = None
        self.status = None
        self.planes_folder = None
        self.add_plane_button = None
        self.slice_button = None
        self.model_handle = None

    def show_mesh(self, mesh: trimesh.Trimesh) -> None:
        if self.model_handle is not None:
            self.model_handle.remove()
            self.model_handle = None

        self.model_handle = self.server.scene.add_mesh_simple(
            "/setup/model",
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            color=PENTOS_BLUE,
            opacity=0.45,
            side="double",
        )

    def handle_upload(self, event) -> None:
        uploaded = event.target.value
        upload_dir = Path("uploaded_models")
        upload_dir.mkdir(exist_ok=True)

        path = upload_dir / uploaded.name
        path.write_bytes(uploaded.content)

        try:
            mesh = load_model(path)
            self.show_mesh(mesh)
            self.state.current_model = (mesh, path.stem)
            self.state.gcode_path = None
            if self.status is not None:
                self.status.value = f"Loaded {uploaded.name}"
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
        mesh, source_name = self.state.current_model

        try:
            if self.status is not None:
                self.status.value = "Slicing..."
            output_path = self.slicer.slice(mesh, planes, source_name)
        except Exception as exc:
            if self.status is not None:
                self.status.value = f"Failed to slice: {exc}"
                print(self.status.value)
            return

        self.state.gcode_path = output_path
        self.show_preview()
