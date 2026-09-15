import numpy as np
import viser
from viser.extras import ViserUrdf
from viser.transforms import SO3

from models import MachineConfig, MachinePose
from models.printer import PrinterModel


class PrinterSimulation:
    def __init__(
        self,
        client: viser.ClientHandle,
        machine_config: MachineConfig,
        model: PrinterModel,
    ) -> None:
        self.machine_config = machine_config
        self.model = model.urdf
        self.bed_zero_inverse = np.linalg.inv(self.model.get_transform("b_rotary_link"))
        self.root = client.scene.add_frame(
            "/preview/printer",
            show_axes=False,
            position=model.position_mm,
        )
        # Borrow the shared plate: reset its pose on removal, but keep its geometry.
        self.bed_frame = client.scene.add_frame("/shared/build_plate", show_axes=False)
        self.print_frame = client.scene.add_frame("/preview/toolpath", show_axes=False)
        self.urdf = ViserUrdf(
            client,
            model.urdf,
            scale=1000.0,
            root_node_name="/preview/printer",
        )
        self.joint_names = self.urdf.get_actuated_joint_names()

    def set_pose(self, pose: MachinePose) -> None:
        machine_center = np.asarray(self.machine_config.machine_plate_center_mm)
        xyz = (pose.xyz_mm - machine_center) / 1000.0
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
        # Carry the plate and printed path with the URDF bed, relative to its flat pose.
        transform = self.model.get_transform("b_rotary_link") @ self.bed_zero_inverse
        rotation = transform[:3, :3]
        position = (
            self.root.position
            - rotation @ self.root.position
            + transform[:3, 3] * 1000.0
        )
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
