from typing import Protocol

import numpy as np

from models import (
    AppState,
    GuideSurfaceSnapshot,
    DEFAULT_TWEEN_SURFACES_PER_PAIR,
    guide_surface_mesh,
    tween_surface_meshes,
)
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

    def replace_tween_surfaces(
        self,
        meshes: list[tuple[np.ndarray, np.ndarray]],
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
        self.tween_surface_count = DEFAULT_TWEEN_SURFACES_PER_PAIR

    def mount(self) -> None:
        self.view.replace_guide_surfaces(self.state.guide_surfaces)
        self._refresh_tweens()

    def add_guide(self) -> None:
        guide = GuideSurfaceSnapshot(
            position=np.zeros(3),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            guide_id=self._allocate_guide_id(),
        )
        self.state.guide_surfaces.append(guide)
        self.view.add_guide_surface(guide)
        self._refresh_tweens()

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
        self._refresh_tweens()

    def remove_guide(self, guide_id: int) -> None:
        self.state.guide_surfaces = [
            guide for guide in self.state.guide_surfaces if guide.guide_id != guide_id
        ]
        self.view.remove_guide_surface(guide_id)
        self._refresh_tweens()

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
        self._refresh_tweens()
        return True

    def set_tween_surface_count(self, count: int) -> None:
        self.tween_surface_count = max(1, count)
        self._refresh_tweens()

    def _refresh_tweens(self) -> None:
        try:
            meshes = tween_surface_meshes(
                self.state.guide_surfaces,
                self.tween_surface_count,
            )
        except ValueError as error:
            self.view.replace_tween_surfaces([])
            self.view.set_status(str(error))
            return

        self.view.replace_tween_surfaces(meshes)

    def _allocate_guide_id(self) -> int:
        guide_id = self.next_guide_id
        self.next_guide_id += 1
        return guide_id

    def _find_guide(self, guide_id: int) -> GuideSurfaceSnapshot:
        return next(
            guide for guide in self.state.guide_surfaces if guide.guide_id == guide_id
        )
