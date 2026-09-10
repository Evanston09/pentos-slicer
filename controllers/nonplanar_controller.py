from typing import Protocol

import numpy as np
import trimesh

from models import AppState, GuideSurfaceSnapshot
from models.guide_surface import GUIDE_HALF_SIZE
from services.auto_planes import quaternion_from_z_to
from services.guide_fitting import align_guide_to_surface
from services.model_tools import transformed_model
from services.volumetric_deformation import (
    TetrahedralVolume,
    solve_guide_deformation,
    solve_guide_scalar_field,
    tetrahedralize,
)


class NonplanarViewPort(Protocol):
    def replace_guide_surfaces(
        self,
        guides: list[GuideSurfaceSnapshot],
    ) -> None: ...

    def add_guide_surface(self, guide: GuideSurfaceSnapshot) -> None: ...

    def remove_guide_surface(self, guide_id: int) -> None: ...

    def update_guide_surface(self, guide: GuideSurfaceSnapshot) -> None: ...

    def set_status(self, message: str) -> None: ...


class NonplanarController:
    def __init__(self, state: AppState, view: NonplanarViewPort) -> None:
        self.state = state
        self.view = view
        self.next_guide_id = (
            max(
                (guide.guide_id for guide in self.state.guide_surfaces),
                default=-1,
            )
            + 1
        )

    def mount(self) -> None:
        self.view.replace_guide_surfaces(self.state.guide_surfaces)

    def add_guide(self) -> None:
        model = transformed_model(self.state)
        position = np.zeros(3) if model is None else model[0].bounds.mean(axis=0)
        wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        size_mm = np.repeat(2.0 * GUIDE_HALF_SIZE, 2)
        heights_mm = np.zeros((4, 4))

        if model is not None and not self.state.guide_surfaces:
            position[2] = model[0].bounds[0, 2]

        if self.state.guide_surfaces:
            previous = self.state.guide_surfaces[-1]
            wxyz = previous.wxyz.copy()
            size_mm = previous.size_mm.copy()
            heights_mm = previous.heights_mm.copy()
            normal = previous.world_normal(np.zeros((1, 2)))[0]
            distance = 10.0
            if model is not None:
                projections = model[0].vertices @ normal
                remaining = float(projections.max() - previous.position @ normal)
                distance = max(remaining, float(np.ptp(projections)) / 10.0)
            position = previous.position + normal * distance

        guide = GuideSurfaceSnapshot(
            position=position,
            wxyz=wxyz,
            guide_id=self._allocate_guide_id(),
            size_mm=size_mm,
            heights_mm=heights_mm,
        )
        self.state.guide_surfaces.append(guide)
        self.view.add_guide_surface(guide)

    def update_guide(
        self,
        guide_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None:
        guide = self._find_guide(guide_id)
        guide.position = np.array(position)
        guide.wxyz = np.array(wxyz)
        self.view.update_guide_surface(guide)

    def set_control_height(
        self,
        guide_id: int,
        row: int,
        column: int,
        height: float,
        *,
        finished: bool = True,
    ) -> None:
        guide = self._find_guide(guide_id)
        guide.heights_mm[row, column] = height
        if finished:
            guide.normalize_center()
        self.view.update_guide_surface(guide)

    def apply_bend(self, guide_id: int, bend_x: float, bend_y: float) -> None:
        guide = self._find_guide(guide_id)
        guide.apply_bend_preset(bend_x, bend_y)
        self.view.update_guide_surface(guide)

    def remove_guide(self, guide_id: int) -> None:
        self.state.guide_surfaces = [
            guide for guide in self.state.guide_surfaces if guide.guide_id != guide_id
        ]
        self.view.remove_guide_surface(guide_id)

    def snap_guide_to_face(
        self,
        guide_id: int,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> bool:
        model = transformed_model(self.state)
        if model is None:
            return False

        guide = self._find_guide(guide_id)
        mesh, _ = model
        locations, _, face_indices = mesh.ray.intersects_location(
            [ray_origin], [ray_direction]
        )
        if len(locations) == 0:
            return False

        hit_index = int(np.linalg.norm(locations - ray_origin, axis=1).argmin())
        guide.position = locations[hit_index]
        guide.wxyz = quaternion_from_z_to(mesh.face_normals[face_indices[hit_index]])
        self.view.update_guide_surface(guide)
        return True

    def align_guide_to_face(
        self,
        guide_id: int,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> bool:
        model = transformed_model(self.state)
        if model is None:
            return False
        guide = self._find_guide(guide_id)
        try:
            candidate = align_guide_to_surface(
                model[0], guide, ray_origin, ray_direction
            )
        except ValueError as exc:
            self.view.set_status(f"Guide {guide_id}: alignment failed: {exc}")
            return False
        if candidate is None:
            return False
        guide.position = candidate.position
        guide.wxyz = candidate.wxyz
        guide.heights_mm = candidate.heights_mm
        self.view.update_guide_surface(guide)
        return True

    def scalar_field_surface(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = transformed_model(self.state)
        if model is None:
            raise ValueError("Load a model before visualizing the scalar field")
        if len(self.state.guide_surfaces) < 2:
            raise ValueError(
                "Add at least two guide surfaces to visualize the scalar field"
            )

        mesh, _ = model
        volume = tetrahedralize(mesh)
        values = solve_guide_scalar_field(volume, self.state.guide_surfaces)
        return volume.original_vertices, volume.boundary_faces, values

    def deformed_mesh(self) -> tuple[trimesh.Trimesh, TetrahedralVolume, str]:
        model = transformed_model(self.state)
        if model is None:
            raise ValueError("Load a model before deforming")
        if len(self.state.guide_surfaces) < 2:
            raise ValueError("Add at least two guide surfaces before deforming")

        mesh, source_name = model
        volume = tetrahedralize(mesh)
        solve_guide_deformation(volume, self.state.guide_surfaces)
        deformed = trimesh.Trimesh(
            vertices=volume.deformed_vertices.copy(),
            faces=volume.boundary_faces,
            process=False,
        )
        if not deformed.is_volume:
            raise ValueError("Deformation produced an invalid outer surface")
        return deformed, volume, source_name

    def _allocate_guide_id(self) -> int:
        guide_id = self.next_guide_id
        self.next_guide_id += 1
        return guide_id

    def _find_guide(self, guide_id: int) -> GuideSurfaceSnapshot:
        return next(
            guide for guide in self.state.guide_surfaces if guide.guide_id == guide_id
        )
