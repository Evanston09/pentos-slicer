from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from viser.transforms import SO3
from yourdfpy import URDF

from machine import rotation_matrix
from models import DEFAULT_MACHINE_CONFIG, MachineConfig, MachinePose
from services.printer_model import (
    PENTOS_URDF_PATH,
    load_printer_model,
)
from views.printer_simulation import PrinterSimulation


@pytest.mark.parametrize(
    "config",
    [
        DEFAULT_MACHINE_CONFIG,
        replace(DEFAULT_MACHINE_CONFIG, rotation_center_machine_mm=(150.0, 80.0, 20.0)),
        MachineConfig(
            build_volume_mm=(120.0, 100.0, 90.0),
            machine_plate_center_mm=(110.0, 55.0, 3.0),
            rotation_center_machine_mm=(114.0, 53.0, 10.0),
        ),
    ],
)
def test_plate_and_print_follow_ideal_urdf(monkeypatch, config) -> None:
    configurations = []

    def make_urdf(client, model, **kwargs):
        assert len(model.scene.geometry) == 6
        # Exercise mapping by name rather than relying on URDF joint ordering.
        names = ("b_joint", "x_joint", "a_joint", "z_joint", "y_joint")

        def update(values):
            configuration = dict(zip(names, values))
            configurations.append(configuration)
            model.update_cfg(configuration)

        return SimpleNamespace(
            get_actuated_joint_names=lambda: names, update_cfg=update
        )

    monkeypatch.setattr("views.printer_simulation.ViserUrdf", make_urdf)
    client = SimpleNamespace(
        scene=SimpleNamespace(
            add_frame=lambda *args, **kwargs: SimpleNamespace(**kwargs)
        )
    )
    printer = PrinterSimulation(client, config, load_printer_model(config))
    original = URDF.load(PENTOS_URDF_PATH, load_meshes=False)
    plate_m = original.get_transform("plate_center")[:3, 3]
    offsets_mm = np.array([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 5]])
    cad_points = np.column_stack([plate_m + offsets_mm / 1000.0, np.ones(4)])
    bed_points = np.linalg.inv(original.get_transform("b_rotary_link")) @ cad_points.T

    for a, b, y in [(0, 0, 0), (90, 0, 0), (0, 90, 0), (30, 45, 10), (-60, -120, -15)]:
        printer.set_pose(
            MachinePose(
                np.asarray(config.machine_plate_center_mm) + [5, y, 8],
                np.asarray([a, b]),
            )
        )
        expected_joints = {
            "x_joint": 0.005,
            "y_joint": y / 1000.0,
            "z_joint": 0.008,
            "a_joint": -np.radians(a),
            "b_joint": np.radians(b),
        }
        for name, value in expected_joints.items():
            assert_allclose(configurations[-1][name], value)
        original.update_cfg(expected_joints)
        cad_world = (original.get_transform("b_rotary_link") @ bed_points)[
            :3
        ].T * 1000.0 + printer.root.position
        for frame in (printer.bed_frame, printer.print_frame):
            rotation = SO3(frame.wxyz).as_matrix()
            assert_allclose(rotation, rotation_matrix(a, b), atol=1e-12)
            preview_world = (
                rotation @ (np.asarray(config.build_plate_center) + offsets_mm).T
            ).T + frame.position
            assert_allclose(cad_world, preview_world, atol=1e-6)

    printer.reset_bed_pose()
    for frame in (printer.bed_frame, printer.print_frame):
        assert_allclose(frame.position, np.zeros(3))
        assert_allclose(frame.wxyz, [1.0, 0.0, 0.0, 0.0])


def test_loader_uses_one_urdf_without_rewriting_geometry(monkeypatch) -> None:
    source = URDF.load(PENTOS_URDF_PATH, load_meshes=False)
    assert_allclose(source.cfg, 0.0)
    assert_allclose(
        source.get_transform("nozzle_tip")[:3, 3],
        source.get_transform("plate_center")[:3, 3],
        atol=1e-12,
    )
    origins = [joint.origin.copy() for joint in source.robot.joints]
    axes = [joint.axis.copy() for joint in source.robot.joints]
    visuals = [
        visual.origin.copy() for link in source.robot.links for visual in link.visuals
    ]
    loads = []

    def load(path, **kwargs):
        loads.append(path)
        return source

    monkeypatch.setattr("services.printer_model.URDF.load", load)
    model = load_printer_model(DEFAULT_MACHINE_CONFIG)
    assert loads == [PENTOS_URDF_PATH]
    assert model.urdf is source
    assert_allclose(model.urdf.cfg, 0.0)
    nozzle_m = source.get_transform("nozzle_tip")[:3, 3]
    plate_m = source.get_transform("plate_center")[:3, 3]
    assert_allclose(nozzle_m, plate_m, atol=1e-9)
    assert_allclose(
        plate_m * 1000 + model.position_mm, DEFAULT_MACHINE_CONFIG.build_plate_center
    )
    for joint, origin, axis in zip(model.urdf.robot.joints, origins, axes):
        assert_allclose(joint.origin, origin)
        assert_allclose(joint.axis, axis)
    for visual, origin in zip(
        [visual for link in source.robot.links for visual in link.visuals], visuals
    ):
        assert_allclose(visual.origin, origin)
