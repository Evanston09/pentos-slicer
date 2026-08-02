from typing import Protocol

import numpy as np

from models import AppState, GuideSurfaceSnapshot, guide_surface_mesh
from services.auto_planes import quaternion_from_z_to
from services.model_tools import transformed_model


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


class NonplanarController:
    def __init__(self, state: AppState, view: NonplanarViewPort) -> None:
        self.state = state
        self.view = view
        self.next_guide_id = 0

    def mount(self) -> None:
        self.view.replace_guide_surfaces(self.state.guide_surfaces)

    def add_guide(self) -> None:
        guide = GuideSurfaceSnapshot(
            position=np.zeros(3),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            guide_id=self._allocate_guide_id(),
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
        if guide is None:
            return
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
        guide = self._find_guide(guide_id)
        if model is None or guide is None:
            return False

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

    def _allocate_guide_id(self) -> int:
        guide_id = self.next_guide_id
        self.next_guide_id += 1
        return guide_id

    def _find_guide(self, guide_id: int) -> GuideSurfaceSnapshot | None:
        return next(
            (
                guide
                for guide in self.state.guide_surfaces
                if guide.guide_id == guide_id
            ),
            None,
        )
