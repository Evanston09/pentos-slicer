import json

import pytest

from models import MachineConfig
from services.machine_config_io import load_machine_config, save_machine_config


def test_machine_config_round_trip_and_derives_offset() -> None:
    config = MachineConfig(
        name="Workshop Pentos",
        build_volume_mm=(100.0, 80.0, 120.0),
        machine_plate_center_mm=(120.0, 60.0, 1.0),
        rotation_center_machine_mm=(119.0, 59.0, 3.5),
        a_max_velocity_deg_s=8.0,
        a_max_acceleration_deg_s2=40.0,
        b_max_velocity_deg_s=16.0,
        b_max_acceleration_deg_s2=80.0,
        max_normal_error_degrees=0.5,
        b_degrees_min=-170.0,
        b_degrees_max=170.0,
    )

    loaded = load_machine_config(save_machine_config(config))

    assert loaded == config
    assert loaded.build_plate_center == (50.0, 40.0, 0.0)
    assert loaded.machine_offset == (70.0, 20.0, 1.0)
    assert loaded.rotation_center_local_mm == (49.0, 39.0, 2.5)


def test_machine_config_rejects_unknown_and_invalid_values() -> None:
    data = json.loads(save_machine_config(MachineConfig()))
    data["unexpected"] = True
    with pytest.raises(ValueError, match="Unknown machine configuration field"):
        load_machine_config(json.dumps(data).encode())

    data.pop("unexpected")
    data["build_volume_mm"] = [90, -1, 90]
    with pytest.raises(ValueError, match="dimensions must be positive"):
        load_machine_config(json.dumps(data).encode())

    data["build_volume_mm"] = [90, 90, 90]
    data["a_max_velocity_deg_s"] = 0
    with pytest.raises(ValueError, match="must be a positive number"):
        load_machine_config(json.dumps(data).encode())

    data["a_max_velocity_deg_s"] = 10
    data["b_degrees_min"] = 180
    data["b_degrees_max"] = -180
    with pytest.raises(ValueError, match="must be less than"):
        load_machine_config(json.dumps(data).encode())
