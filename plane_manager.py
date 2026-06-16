import numpy as np
from trimesh import transformations as tf
from viser import ViserServer

ORANGE = (255, 130, 0)
ARROW_BLUE = (47, 153, 238)


class PlaneManager:
    def __init__(self, server: ViserServer):
        self.server = server
        self.next_id = 0
        self.planes = {}
        self.selected_plane_id = None
        self.updating_gui_from_scene = False
        self.plane_half_size = 0.05
        self.arrow_length = self.plane_half_size

    def add_plane(self, mesh=None):
        plane_id = self.next_id
        self.next_id += 1

        half = self.plane_half_size
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
        arrow = self.server.scene.add_arrows(
            f"/planes/{plane_id}/normal",
            points=np.array(
                [[[0.0, 0.0, 0.0], [0.0, 0.0, self.arrow_length]]],
                dtype=float,
            ),
            colors=ARROW_BLUE,
            shaft_radius=half * 0.035,
            head_radius=half * 0.09,
            head_length=half * 0.22,
        )

        self.planes[plane_id] = {
            "plane": plane,
            "gizmo": gizmo,
            "arrow": arrow,
        }

        @gizmo.on_update
        def _(_event):
            self._sync_plane_from_gizmo(plane_id)

        @plane.on_click
        def _(_event):
            self.select_plane(plane_id)

        self.select_plane(plane_id)
        return plane_id

    def remove_plane(self, plane_id):
        if plane_id not in self.planes:
            return

        handles = self.planes.pop(plane_id)
        handles["plane"].remove()
        handles["gizmo"].remove()
        handles["arrow"].remove()

        if self.selected_plane_id == plane_id:
            self.selected_plane_id = None
            self._remove_selected_plane_menu()

    def get_all_planes(self):
        return [handles["plane"] for handles in self.planes.values()]

    def select_plane(self, plane_id):
        if plane_id not in self.planes:
            return

        self.selected_plane_id = plane_id
        self._build_selected_plane_menu(plane_id)

    def _build_selected_plane_menu(self, plane_id):
        self._remove_selected_plane_menu()

        handles = self.planes[plane_id]
        plane = handles["plane"]
        euler_degrees = self._euler_degrees_for_plane(plane)

        folder = self.server.gui.add_folder(
            f"Plane {plane_id}",
            expand_by_default=True,
        )

        with folder:
            position = self.server.gui.add_vector3(
                "Position",
                plane.position,
                step=0.001,
            )
            rotation_x = self.server.gui.add_number(
                "Rotation X",
                euler_degrees[0],
                step=1.0,
            )
            rotation_y = self.server.gui.add_number(
                "Rotation Y",
                euler_degrees[1],
                step=1.0,
            )
            rotation_z = self.server.gui.add_number(
                "Rotation Z",
                euler_degrees[2],
                step=1.0,
            )
            delete_button = self.server.gui.add_button("Delete Plane")

        handles["gui"] = {
            "folder": folder,
            "position": position,
            "rotation_x": rotation_x,
            "rotation_y": rotation_y,
            "rotation_z": rotation_z,
            "delete_button": delete_button,
        }

        @position.on_update
        def _(_event):
            if self.updating_gui_from_scene or plane_id not in self.planes:
                return

            self.planes[plane_id]["gizmo"].position = position.value
            self._sync_plane_from_gizmo(plane_id)

        def update_rotation_from_gui():
            if self.updating_gui_from_scene or plane_id not in self.planes:
                return

            self.planes[plane_id]["gizmo"].wxyz = self._wxyz_from_euler_degrees(
                (
                    rotation_x.value,
                    rotation_y.value,
                    rotation_z.value,
                )
            )
            self._sync_plane_from_gizmo(plane_id)

        @rotation_x.on_update
        def _(_event):
            update_rotation_from_gui()

        @rotation_y.on_update
        def _(_event):
            update_rotation_from_gui()

        @rotation_z.on_update
        def _(_event):
            update_rotation_from_gui()

        @delete_button.on_click
        def _(_event):
            self.remove_plane(plane_id)

    def _remove_selected_plane_menu(self):
        for handles in self.planes.values():
            gui = handles.pop("gui", None)
            if gui is not None:
                for handle in reversed(gui.values()):
                    handle.remove()

    def _sync_plane_from_gizmo(self, plane_id):
        if plane_id not in self.planes:
            return

        handles = self.planes[plane_id]
        plane = handles["plane"]
        gizmo = handles["gizmo"]
        arrow = handles["arrow"]

        plane.position = gizmo.position
        plane.wxyz = gizmo.wxyz
        arrow.position = gizmo.position
        arrow.wxyz = gizmo.wxyz

        if self.selected_plane_id == plane_id:
            self._sync_selected_gui_from_plane(plane_id)

    def _sync_selected_gui_from_plane(self, plane_id):
        handles = self.planes.get(plane_id)
        if handles is None or "gui" not in handles:
            return

        plane = handles["plane"]
        gui = handles["gui"]

        self.updating_gui_from_scene = True
        gui["position"].value = plane.position
        euler_degrees = self._euler_degrees_for_plane(plane)
        gui["rotation_x"].value = euler_degrees[0]
        gui["rotation_y"].value = euler_degrees[1]
        gui["rotation_z"].value = euler_degrees[2]
        self.updating_gui_from_scene = False

    @staticmethod
    def _euler_degrees_for_plane(plane):
        radians = tf.euler_from_quaternion(plane.wxyz, axes="sxyz")
        degrees = np.degrees(np.asarray(radians, dtype=float))
        return tuple(np.round(degrees, 3))

    @staticmethod
    def _wxyz_from_euler_degrees(degrees):
        radians = np.radians(degrees)
        return tf.quaternion_from_euler(*radians, axes="sxyz")
