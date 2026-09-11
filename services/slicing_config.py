import json
from pathlib import Path

from models.slicing_settings import SlicingSettings
from models.slicing_settings import filament_preset

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "pentos_config.ini"
SLICING_SETTINGS_STORAGE_KEY = "pentos-slicing-settings"

# PLA uses the baseline; PETG overrides follow PrusaSlicer's Creality profile as of 9/10/26.
MATERIAL_OVERRIDES = {
    "PLA": {},
    "PETG": {
        "min_fan_speed": 20,
        "max_fan_speed": 50,
        "disable_fan_first_layers": 3,
        "filament_max_volumetric_speed": 8.0,
        "idle_temperature": 160,
        "fan_below_layer_time": 20,
        "filament_density": 1.27,
        "filament_cost": 30.0,
    },
}


def load_slicing_settings(content: str) -> SlicingSettings:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Slicing settings must be valid JSON") from exc
    return SlicingSettings.from_dict(data)


def save_slicing_settings(settings: SlicingSettings) -> str:
    return json.dumps(settings.as_dict())


def write_slicing_config(
    settings: SlicingSettings,
    output_path: Path,
    baseline_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    groups = settings.as_dict()
    material = settings.filament.filament_type
    values = {
        **MATERIAL_OVERRIDES[material],
        **groups["filament"],
        **groups["print"],
    }
    replacements = {key: str(value) for key, value in values.items()}
    replacements["fill_density"] += "%"
    profile = f"Generic {material} @CREALITY"
    replacements["default_filament_profile"] = f'"{profile}"'
    if settings.filament != filament_preset(material):
        profile += " (modified)"
    replacements["filament_settings_id"] = f'"{profile}"'
    # The baseline grid pattern cannot be sliced at 100% density.
    if settings.print.fill_density == 100:
        replacements["fill_pattern"] = "rectilinear"
    lines = []
    for line in baseline_path.read_text().splitlines(keepends=True):
        key, separator, _ = line.partition("=")
        key = key.strip()
        if separator and key in replacements:
            line = f"{key} = {replacements.pop(key)}\n"
        lines.append(line)
    if replacements:
        raise ValueError(f"Baseline is missing settings: {', '.join(replacements)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines))
    return output_path
