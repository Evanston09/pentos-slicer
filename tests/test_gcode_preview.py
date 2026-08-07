import numpy as np
from numpy.testing import assert_allclose
from viser.transforms import SO3

from machine import BUILD_PLATE_CENTER, MACHINE_OFFSET, ROTATION_CENTER, rotation_matrix
from models import MachinePose
from services.gcode_preview import parse_gcode_preview
from views.preview_view import (
    URDF_BED_SURFACE_OFFSET,
    URDF_BED_SURFACE_ROTATION,
    URDF_SCALE,
    bed_surface_pose,
    build_plate_pose,
    machine_configuration,
    machine_root_pose,
)


def test_preview_uses_each_endpoint_ab_pose() -> None:
    start = np.array([45.0, 45.0, 5.0])
    end = np.array([46.0, 45.0, 5.0])
    center = np.asarray(ROTATION_CENTER)
    machine_start = np.asarray(MACHINE_OFFSET) + start
    machine_end = (
        np.asarray(MACHINE_OFFSET)
        + center
        + rotation_matrix(10.0, 0.0) @ (end - center)
    )
    text = (
        "G90\n"
        "M83\n"
        f"G1 X{machine_start[0]} Y{machine_start[1]} Z{machine_start[2]}\n"
        ";LAYER_CHANGE\n"
        f"G1 X{machine_end[0]} Y{machine_end[1]} Z{machine_end[2]} "
        "A10 B0 E0.1\n"
    )

    preview = parse_gcode_preview(text)

    assert_allclose(preview.parts[0].extrusion[0], [start, end])


def test_preview_tracks_machine_xyzab_poses() -> None:
    preview = parse_gcode_preview(
        "G90\nG1 X68 Y7 Z10\nG1 A90 B45\n;LAYER_CHANGE\nG1 X69 Y8\n"
    )

    assert len(preview.machine_poses) == 3
    assert_allclose(preview.machine_poses[0].xyz, [68.0, 7.0, 10.0])
    assert_allclose(preview.machine_poses[0].ab, [0.0, 0.0])
    assert_allclose(preview.machine_poses[1].xyz, [68.0, 7.0, 10.0])
    assert_allclose(preview.machine_poses[1].ab, [90.0, 45.0])
    assert_allclose(preview.machine_poses[2].xyz, [69.0, 8.0, 10.0])
    assert_allclose(preview.machine_poses[2].ab, [90.0, 45.0])


def test_machine_bed_surface_stays_on_virtual_build_plate() -> None:
    angle = np.radians(35.0)
    bed_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    bed_transform = np.eye(4)
    bed_transform[:3, :3] = bed_rotation
    bed_transform[:3, 3] = [0.2, -0.1, 0.3]

    root_wxyz, root_position = machine_root_pose(bed_transform)
    root_rotation = SO3(root_wxyz).as_matrix()
    surface_rotation = bed_rotation @ URDF_BED_SURFACE_ROTATION
    surface_position = bed_transform[:3, 3] + bed_rotation @ URDF_BED_SURFACE_OFFSET

    assert_allclose(root_rotation @ surface_rotation, np.eye(3), atol=1e-7)
    assert_allclose(
        root_position + root_rotation @ surface_position * URDF_SCALE,
        BUILD_PLATE_CENTER,
    )


def test_virtual_build_plate_follows_bed_surface() -> None:
    zero_transform = np.eye(4)
    zero_transform[:3, 3] = [0.0, -0.03, 0.17]
    root_wxyz, root_position = machine_root_pose(zero_transform)

    moved_transform = np.eye(4)
    moved_transform[:3, :3] = SO3.from_y_radians(np.radians(40.0)).as_matrix()
    moved_transform[:3, 3] = [0.0, -0.08, 0.14]
    plate_wxyz, plate_position = build_plate_pose(
        root_wxyz,
        root_position,
        moved_transform,
    )

    root_rotation = SO3(root_wxyz).as_matrix()
    surface_rotation, surface_position = bed_surface_pose(moved_transform)
    assert_allclose(
        SO3(plate_wxyz).as_matrix(),
        root_rotation @ surface_rotation,
        atol=1e-7,
    )
    assert_allclose(
        plate_position + SO3(plate_wxyz).as_matrix() @ np.asarray(BUILD_PLATE_CENTER),
        root_position + root_rotation @ surface_position * URDF_SCALE,
    )


def test_machine_configuration_uses_urdf_units_and_joint_order() -> None:
    pose = MachinePose(
        xyz=np.array([100.0, 50.0, 25.0]),
        ab=np.array([90.0, -180.0]),
    )

    configuration = machine_configuration(
        ("z_joint", "x_joint", "y_joint", "a_joint", "b_joint"),
        pose,
    )

    assert_allclose(configuration, [0.025, 0.1023, 0.0362, -np.pi / 2, -np.pi])
