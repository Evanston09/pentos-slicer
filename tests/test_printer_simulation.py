from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
from viser.transforms import SO3

from machine import rotation_matrix
from models import DEFAULT_MACHINE_CONFIG, MachinePose
from views.printer_simulation import CENTERED_XYZ_JOINTS_M, PrinterSimulation


def test_pose_maps_joints_and_rotates_bed_and_print() -> None:
    printer = PrinterSimulation.__new__(PrinterSimulation)
    printer.machine_config = DEFAULT_MACHINE_CONFIG
    printer.joint_names = (
        "b_joint",
        "x_joint",
        "a_joint",
        "z_joint",
        "y_joint",
    )
    configurations = []
    printer.urdf = SimpleNamespace(update_cfg=configurations.append)
    printer.bed_frame = SimpleNamespace(position=None, wxyz=None)
    printer.print_frame = SimpleNamespace(position=None, wxyz=None)
    pose = MachinePose(
        np.asarray(DEFAULT_MACHINE_CONFIG.machine_plate_center_mm)
        + np.array([0.0, 10.0, 0.0]),
        np.array([30.0, 45.0]),
    )

    printer.set_pose(pose)

    assert_allclose(
        configurations[0],
        [
            np.radians(45.0),
            CENTERED_XYZ_JOINTS_M[0],
            -np.radians(30.0),
            CENTERED_XYZ_JOINTS_M[2],
            CENTERED_XYZ_JOINTS_M[1] + 0.01,
        ],
    )
    rotation = rotation_matrix(30.0, 45.0)
    center = np.asarray(DEFAULT_MACHINE_CONFIG.rotation_center_local_mm)
    expected_position = center - rotation @ center + [0.0, -10.0, 0.0]
    for frame in (printer.bed_frame, printer.print_frame):
        assert_allclose(frame.position, expected_position)
        assert_allclose(frame.wxyz, SO3.from_matrix(rotation).wxyz)

    printer.reset_bed_pose()
    for frame in (printer.bed_frame, printer.print_frame):
        assert_allclose(frame.position, np.zeros(3))
        assert_allclose(frame.wxyz, [1.0, 0.0, 0.0, 0.0])
