from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf
from viser.transforms import SO3

from machine import rotation_matrix
from models import MachineConfig, MachinePose

PENTOS_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "Pentos_URDF"
    / "urdf"
    / "Pentos_URDF.urdf"
)

# CAD anchors in the URDF base frame at zero configuration. The nozzle tip is
# the x-carriage origin; the plate anchor is the center of its top surface.
URDF_NOZZLE_ZERO_M = np.array([-0.1152, -0.0637, 0.169036679])
URDF_PLATE_ZERO_M = np.array([0.000205120059, -0.0257794817, 0.168542899673])
CENTERED_XYZ_JOINTS_M = URDF_PLATE_ZERO_M - URDF_NOZZLE_ZERO_M


class PrinterSimulation:
    def __init__(
        self,
        client: viser.ClientHandle,
        machine_config: MachineConfig,
    ) -> None:
        self.machine_config = machine_config
        centered_plate_m = np.array(
            [
                URDF_PLATE_ZERO_M[0],
                URDF_NOZZLE_ZERO_M[1],
                URDF_PLATE_ZERO_M[2],
            ]
        )
        self.root = client.scene.add_frame(
            "/preview/printer",
            show_axes=False,
            position=np.asarray(machine_config.build_plate_center)
            - centered_plate_m * 1000.0,
        )
        self.bed_frame = client.scene.add_frame("/shared/build_plate", show_axes=False)
        self.print_frame = client.scene.add_frame("/preview/toolpath", show_axes=False)
        self.urdf = ViserUrdf(
            client,
            PENTOS_URDF_PATH,
            scale=1000.0,
            root_node_name="/preview/printer",
        )
        self.joint_names = self.urdf.get_actuated_joint_names()

    def set_pose(self, pose: MachinePose) -> None:
        machine_center = np.asarray(self.machine_config.machine_plate_center_mm)
        xyz = CENTERED_XYZ_JOINTS_M + (pose.xyz_mm - machine_center) / 1000.0
        values = {
            "x_joint": xyz[0],
            "y_joint": xyz[1],
            "z_joint": xyz[2],
            # Positive machine A is about +Y; the CAD joint axis is -Y.
            "a_joint": -np.radians(pose.ab_degrees[0]),
            # Positive machine B is clockwise from above, matching the -Z axis.
            "b_joint": np.radians(pose.ab_degrees[1]),
        }
        self.urdf.update_cfg(np.asarray([values[name] for name in self.joint_names]))
        rotation = rotation_matrix(*pose.ab_degrees)
        center = np.asarray(self.machine_config.rotation_center_local_mm)
        position = center - rotation @ center
        position[1] -= pose.xyz_mm[1] - self.machine_config.machine_plate_center_mm[1]
        wxyz = SO3.from_matrix(rotation).wxyz
        for frame in (self.bed_frame, self.print_frame):
            frame.position = position
            frame.wxyz = wxyz

    def reset_bed_pose(self) -> None:
        for frame in (self.bed_frame, self.print_frame):
            frame.position = np.zeros(3)
            frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    def remove(self) -> None:
        self.reset_bed_pose()
        self.print_frame.remove()
        self.root.remove()
