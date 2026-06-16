import numpy as np
from viser import ViserServer

ORANGE = (255, 130, 0)


class PlaneManager:
    def __init__(self, server: ViserServer):
        self.server = server
        self.next_id = 0
        self.planes = {}

    def add_plane(self, mesh=None):
        plane_id = self.next_id
        self.next_id += 1

        half = 0.05
        vertices = np.array(
            [
                [-half, -half, 0],
                [half, -half, 0],
                [half, half, 0],
                [-half, half, 0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        plane = self.server.scene.add_mesh_simple(
            f"/planes/{plane_id}/mesh",
            vertices,
            faces,
            color=ORANGE,
            opacity=0.35,
            side="double",
        )
        gizmo = self.server.scene.add_transform_controls(
            f"/planes/{plane_id}/controls",
            scale=half * 0.2,
        )

        self.planes[plane_id] = {
            "plane": plane,
            "gizmo": gizmo,
        }

        @gizmo.on_update
        def _(_event):
            plane.position = gizmo.position
            plane.wxyz = gizmo.wxyz

        return plane_id

    def remove_plane(self, plane_id):
        handles = self.planes.pop(plane_id)
        handles["plane"].remove()
        handles["gizmo"].remove()

    def get_all_planes(self):
        return [handles["plane"] for handles in self.planes.values()]
