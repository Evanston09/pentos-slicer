from collections.abc import Callable
from typing import Protocol

import numpy as np
import trimesh

from machine import BUILD_PLATE_CENTER
from models import AppState, PlaneSnapshot
from services.auto_planes import (
    AutoPlaneConfig,
    AutoPlaneSelector,
    overhang_preview_mesh,
)
from services.model_tools import (
    load_uploaded_model,
    model_center,
    model_frame_position,
    model_within_build_volume,
    model_wxyz,
    transformed_model,
)
from services.project_io import load_scene, save_scene
from services.session_workspace import SessionWorkspace
from services.slice_jobs import SlicingBusyError, SlicingCoordinator
from services.slicing import Slicer


class SetupViewPort(Protocol):
    def mount(self, state: AppState) -> None: ...

    def unmount(self) -> None: ...

    def set_status(self, message: str) -> None: ...

    def set_slice_enabled(self, enabled: bool) -> None: ...

    def show_mesh(
        self,
        mesh: trimesh.Trimesh,
        center: np.ndarray,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None: ...

    def clear_model_scene(self) -> None: ...

    def show_overhang_faces(
        self,
        mesh: trimesh.Trimesh,
        overhang_mask: np.ndarray,
    ) -> None: ...

    def set_model_out_of_bounds(self, out_of_bounds: bool) -> None: ...

    def set_model_controls_enabled(self, enabled: bool) -> None: ...

    def update_model_placement(
        self,
        xy_position: list[float],
        z_degrees: float,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None: ...

    def replace_planes(self, planes: list[PlaneSnapshot]) -> None: ...

    def add_plane(self, plane: PlaneSnapshot) -> None: ...

    def remove_plane(self, plane_id: int) -> None: ...

    def set_debug_mode_value(self, enabled: bool) -> None: ...


class SetupController:
    def __init__(
        self,
        state: AppState,
        slicer: Slicer,
        view: SetupViewPort,
        show_preview: Callable[[], None],
        workspace: SessionWorkspace,
        slicing_coordinator: SlicingCoordinator,
    ) -> None:
        self.state = state
        self.slicer = slicer
        self.view = view
        self.show_preview = show_preview
        self.workspace = workspace
        self.upload_dir = workspace.path / "uploads"
        self.slicing_coordinator = slicing_coordinator
        self.overhang_threshold_degrees = AutoPlaneConfig().overhang_threshold_degrees
        self.next_plane_id = 0
        self._assign_missing_plane_ids()

    def mount(self) -> None:
        self.view.mount(self.state)
        self.view.replace_planes(self.state.plane_snapshots)
        if self.state.current_model is not None:
            self._show_current_model()
            self.view.set_status(f"Loaded {self.state.current_model[1]}")

    def unmount(self) -> None:
        self.view.unmount()

    def handle_upload(self, name: str, content: bytes) -> None:
        if name.lower().endswith(".pentos"):
            try:
                self._load_scene_state(load_scene(content))
            except Exception as exc:
                self.view.set_status(f"Failed to load {name}: {exc}")
                print(f"Failed to load {name}: {exc}")
            return

        try:
            mesh, source_name = load_uploaded_model(name, content, self.upload_dir)
            self.state.current_model = (mesh, source_name)
            self.state.model_xy_position = BUILD_PLATE_CENTER[:2]
            self.state.model_z_degrees = 0.0
            self.state.gcode_path = None
            self._show_current_model()
            self.view.set_status(f"Loaded {source_name}")
        except Exception as exc:
            self.view.set_status(f"Failed to load {name}: {exc}")
            print(f"Failed to load {name}: {exc}")

    def set_model_placement(
        self,
        xy_position: list[float] | None = None,
        z_degrees: float | None = None,
    ) -> None:
        if self.state.current_model is None:
            return

        mesh, _ = self.state.current_model
        if xy_position is not None:
            self.state.model_xy_position = [xy_position[0], xy_position[1]]
        if z_degrees is not None:
            self.state.model_z_degrees = z_degrees

        self.view.update_model_placement(
            self.state.model_xy_position,
            self.state.model_z_degrees,
            model_frame_position(self.state, mesh),
            model_wxyz(self.state),
        )
        self._refresh_model_bounds()
        self.refresh_overhang_preview()

    def reset_model_placement(self) -> None:
        self.set_model_placement(BUILD_PLATE_CENTER[:2], 0.0)

    def add_plane(
        self,
        position: np.ndarray | None = None,
        wxyz: np.ndarray | None = None,
    ) -> None:
        plane = PlaneSnapshot(
            position=np.zeros(3) if position is None else np.array(position),
            wxyz=(
                np.array([1.0, 0.0, 0.0, 0.0])
                if wxyz is None
                else self._normalize_quaternion(wxyz)
            ),
            plane_id=self._allocate_plane_id(),
        )
        self.state.plane_snapshots.append(plane)
        self.view.add_plane(plane)
        self.refresh_overhang_preview()

    def update_plane(
        self,
        plane_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        plane = self._find_plane(plane_id)
        if plane is None:
            return
        plane.position = np.array(position)
        plane.wxyz = self._normalize_quaternion(wxyz)
        self.refresh_overhang_preview()

    def remove_plane(self, plane_id: int) -> None:
        self.state.plane_snapshots = [
            plane for plane in self.state.plane_snapshots if plane.plane_id != plane_id
        ]
        self.view.remove_plane(plane_id)
        self.refresh_overhang_preview()

    def set_debug_mode(self, enabled: bool) -> None:
        self.state.debug_mode = enabled

    def refresh_overhang_preview(self) -> None:
        model = transformed_model(self.state)
        if model is None:
            return

        mesh, _ = model
        preview_mesh, overhang_mask = overhang_preview_mesh(
            mesh,
            self.state.plane_snapshots,
            self.overhang_threshold_degrees,
        )
        self.view.show_overhang_faces(preview_mesh, overhang_mask)

    def select_auto_planes(self, max_planes: int) -> None:
        model = transformed_model(self.state)
        if model is None:
            self.view.set_status("Load a model before selecting planes")
            return

        mesh, _ = model
        self.view.set_status("Selecting auto planes...")
        try:
            selector = AutoPlaneSelector(AutoPlaneConfig(max_planes=max_planes))
            candidates = selector.select(mesh)
        except Exception as exc:
            self.view.set_status(f"Auto plane selection failed: {exc}")
            print(f"Auto plane selection failed: {exc}")
            return

        self.state.plane_snapshots = [
            PlaneSnapshot(
                position=np.array(candidate.position),
                wxyz=np.array(candidate.wxyz),
                plane_id=self._allocate_plane_id(),
            )
            for candidate in candidates
        ]
        self.view.replace_planes(self.state.plane_snapshots)
        self.refresh_overhang_preview()

        if not candidates:
            self.view.set_status(
                "Auto Planes: no split beat baseline; flat slicing is available"
            )
        else:
            self.view.set_status(f"Auto Planes: selected {len(candidates)} plane(s)")

    def export_scene(self) -> tuple[str, bytes] | None:
        self.view.set_status("Exporting scene")
        try:
            scene_bytes = save_scene(self.state)
        except Exception as exc:
            self.view.set_status(f"Scene export failed: {exc}")
            return None

        filename = "pentos_scene.pentos"
        if self.state.current_model is not None:
            filename = self.state.current_model[1] + ".pentos"

        self.view.set_status(f"Exported {filename}")
        return filename, scene_bytes

    def slice_model(self) -> None:
        model = transformed_model(self.state)
        if model is None:
            self.view.set_status("Load a model before slicing")
            return

        mesh, source_name = model
        self.view.set_slice_enabled(False)
        try:
            with self.workspace.active_job():
                with self.slicing_coordinator.slot():
                    self.view.set_status(
                        "Generating debug transition check..."
                        if self.state.debug_mode
                        else "Slicing..."
                    )
                    if self.state.debug_mode:
                        output_path = self.slicer.debug_transition_check(
                            mesh,
                            self.state.plane_snapshots,
                            source_name,
                        )
                    else:
                        output_path = self.slicer.slice(
                            mesh,
                            self.state.plane_snapshots,
                            source_name,
                        )
        except SlicingBusyError as exc:
            self.view.set_status(str(exc))
            return
        except Exception as exc:
            self.view.set_status(f"Failed to slice: {exc}")
            print(f"Failed to slice: {exc}")
            return
        finally:
            self.view.set_slice_enabled(True)

        self.state.gcode_path = output_path
        self.show_preview()

    def _load_scene_state(self, loaded_state: AppState) -> None:
        self.state.current_model = loaded_state.current_model
        self.state.model_xy_position = loaded_state.model_xy_position
        self.state.model_z_degrees = loaded_state.model_z_degrees
        self.state.plane_snapshots = loaded_state.plane_snapshots
        self.state.gcode_path = None
        self.state.debug_mode = loaded_state.debug_mode
        self._assign_missing_plane_ids()

        self.view.clear_model_scene()
        self.view.replace_planes(self.state.plane_snapshots)
        self.view.set_debug_mode_value(self.state.debug_mode)

        if self.state.current_model is not None:
            self._show_current_model()
            self.view.set_status(f"Loaded scene {self.state.current_model[1]}")

    def _show_current_model(self) -> None:
        if self.state.current_model is None:
            return
        mesh, _ = self.state.current_model
        self.view.show_mesh(
            mesh,
            model_center(mesh),
            model_frame_position(self.state, mesh),
            model_wxyz(self.state),
        )
        self.view.set_model_controls_enabled(True)
        self.view.update_model_placement(
            self.state.model_xy_position,
            self.state.model_z_degrees,
            model_frame_position(self.state, mesh),
            model_wxyz(self.state),
        )
        self._refresh_model_bounds()

    def _refresh_model_bounds(self) -> None:
        model = transformed_model(self.state)
        if model is not None:
            self.view.set_model_out_of_bounds(not model_within_build_volume(model[0]))

    def _assign_missing_plane_ids(self) -> None:
        used_ids = {
            plane.plane_id
            for plane in self.state.plane_snapshots
            if plane.plane_id is not None
        }
        self.next_plane_id = max(used_ids, default=-1) + 1
        for plane in self.state.plane_snapshots:
            if plane.plane_id is None:
                plane.plane_id = self._allocate_plane_id()

    def _allocate_plane_id(self) -> int:
        plane_id = self.next_plane_id
        self.next_plane_id += 1
        return plane_id

    def _find_plane(self, plane_id: int) -> PlaneSnapshot | None:
        return next(
            (
                plane
                for plane in self.state.plane_snapshots
                if plane.plane_id == plane_id
            ),
            None,
        )

    @staticmethod
    def _normalize_quaternion(wxyz: np.ndarray) -> np.ndarray:
        wxyz = np.array(wxyz, dtype=float)
        norm = np.linalg.norm(wxyz)
        if np.isclose(norm, 0.0):
            return np.array([1.0, 0.0, 0.0, 0.0])
        return wxyz / norm
