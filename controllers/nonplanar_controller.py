from typing import Protocol

import numpy as np
import trimesh
from trimesh import transformations as tf

from models import AppState, GuideSurfaceSnapshot, guide_surface_mesh
from services.auto_planes import quaternion_from_z_to
from services.model_tools import transformed_model
from services.volumetric_deformation import (
    TetrahedralVolume,
    solve_guide_deformation,
    tetrahedralize,
)


class NonplanarViewPort(Protocol):
    def replace_guide_surfaces(
        self,
        guides: list[GuideSurfaceSnapshot],
    ) -> None: ...

    def add_guide_surface(self, guide: GuideSurfaceSnapshot) -> None: ...

    def remove_guide_surface(self, guide_id: int) -> None: ...

    def set_guide_surface_pose(
        self,
        guide_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
    ) -> None: ...

    def set_guide_surface_mesh(
        self,
        guide_id: int,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None: ...

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
        bend_x = 0.0
        bend_y = 0.0

        if model is not None and not self.state.guide_surfaces:
            position[2] = model[0].bounds[0, 2]

        if self.state.guide_surfaces:
            previous = max(
                self.state.guide_surfaces,
                key=lambda guide: guide.guide_id,
            )
            wxyz = previous.wxyz.copy()
            bend_x = previous.bend_x
            bend_y = previous.bend_y
            normal = tf.quaternion_matrix(wxyz)[:3, 2]
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
            bend_x=bend_x,
            bend_y=bend_y,
        )
        self.state.guide_surfaces.append(guide)
        self.view.add_guide_surface(guide)

    def update_guide(
        self,
        guide_id: int,
        position: np.ndarray,
        wxyz: np.ndarray,
        bend_x: float,
        bend_y: float,
    ) -> None:
        guide = self._find_guide(guide_id)
        guide.position = np.array(position)
        guide.wxyz = np.array(wxyz)
        guide.bend_x = bend_x
        guide.bend_y = bend_y
        self.view.set_guide_surface_mesh(
            guide_id,
            *guide_surface_mesh(bend_x, bend_y),
        )

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
        self.view.set_guide_surface_pose(guide_id, guide.position, guide.wxyz)
        return True

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
