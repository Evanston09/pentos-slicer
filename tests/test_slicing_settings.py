from dataclasses import FrozenInstanceError, replace
import json

import pytest

from models import FilamentSettings, PrintSettings, SlicingSettings
from models.slicing_settings import filament_preset
from services.slicing_config import (
    DEFAULT_CONFIG_PATH,
    load_slicing_settings,
    save_slicing_settings,
    write_slicing_config,
)


def config_values(text: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for line in text.splitlines()
        if "=" in line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }


def test_defaults_preserve_baseline_and_protected_lines(tmp_path) -> None:
    original = DEFAULT_CONFIG_PATH.read_text()
    generated = write_slicing_config(
        SlicingSettings(), tmp_path / "config.ini"
    ).read_text()
    before, after = config_values(original), config_values(generated)
    assert before.keys() == after.keys()
    edited_keys = set(FilamentSettings.__dataclass_fields__) | set(
        PrintSettings.__dataclass_fields__
    )
    for key in before:
        if key in edited_keys and key != "filament_type":
            assert float(before[key].rstrip("%")) == float(after[key].rstrip("%"))
        else:
            assert before[key] == after[key]
    for line in original.splitlines():
        if line.startswith(
            ("#", "start_gcode", "end_gcode", "use_relative_e_distances")
        ):
            assert line in generated
    assert DEFAULT_CONFIG_PATH.read_text() == original


def test_petg_and_custom_values_are_isolated_per_session(tmp_path) -> None:
    pla = SlicingSettings()
    petg = SlicingSettings(
        filament=replace(filament_preset("PETG"), temperature=245),
        print=PrintSettings(fill_density=25),
    )
    first = write_slicing_config(pla, tmp_path / "client1" / "temp" / "config.ini")
    second = write_slicing_config(petg, tmp_path / "client2" / "temp" / "config.ini")
    values = config_values(second.read_text())
    assert values["filament_type"] == "PETG"
    assert values["filament_settings_id"] == '"Generic PETG @CREALITY (modified)"'
    assert values["temperature"] == "245"
    assert values["first_layer_temperature"] == "240"
    assert values["idle_temperature"] == "160"
    assert values["min_fan_speed"] == "20"
    assert values["max_fan_speed"] == "50"
    assert values["fan_below_layer_time"] == "20"
    assert values["disable_fan_first_layers"] == "3"
    assert values["filament_max_volumetric_speed"] == "8.0"
    baseline = config_values(DEFAULT_CONFIG_PATH.read_text())
    for key in baseline:
        if key.endswith("_speed") and key not in (
            "min_fan_speed",
            "max_fan_speed",
            "filament_max_volumetric_speed",
        ):
            assert values[key] == baseline[key]
    assert values["filament_density"] == "1.27"
    assert values["filament_cost"] == "30.0"
    assert values["fill_density"] == "25%"
    assert config_values(first.read_text())["temperature"] == "205"
    with pytest.raises(FrozenInstanceError):
        petg.filament.temperature = 250


def test_solid_infill_uses_supported_pattern(tmp_path) -> None:
    settings = SlicingSettings(print=PrintSettings(fill_density=100))
    generated = write_slicing_config(settings, tmp_path / "config.ini").read_text()
    assert config_values(generated)["fill_pattern"] == "rectilinear"


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 0},
        {"temperature": True},
        {"first_layer_temperature": 0},
        {"temperature": 205.5},
        {"bed_temperature": -1},
        {"filament_type": "ABS"},
    ],
)
def test_invalid_filament_settings(changes) -> None:
    with pytest.raises(ValueError):
        FilamentSettings(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"layer_height": 0.5},
        {"first_layer_height": 0},
        {"perimeters": True},
        {"fill_density": 101},
        {"fill_density": float("nan")},
        {"layer_height": float("inf")},
    ],
)
def test_invalid_print_settings(changes) -> None:
    with pytest.raises(ValueError):
        PrintSettings(**changes)


def test_settings_json_round_trip() -> None:
    settings = SlicingSettings(filament=filament_preset("PETG"))
    assert settings.as_dict() == {
        "filament": {
            "filament_type": "PETG",
            "temperature": 240,
            "first_layer_temperature": 240,
            "bed_temperature": 70,
            "first_layer_bed_temperature": 70,
        },
        "print": {
            "layer_height": 0.2,
            "first_layer_height": 0.2,
            "perimeters": 3,
            "fill_density": 15.0,
        },
    }
    assert load_slicing_settings(save_slicing_settings(settings)) == settings
    for content in (
        "null",
        "[]",
        "invalid",
        json.dumps({"filament": None, "print": {}}),
    ):
        with pytest.raises(ValueError):
            load_slicing_settings(content)
