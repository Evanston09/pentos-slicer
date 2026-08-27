import json
import math
from dataclasses import asdict, fields

from models import MachineConfig

FORMAT = "pentos-machine"
VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
_FIELDS = {"format", "version"} | {field.name for field in fields(MachineConfig)}


def load_machine_config(content: bytes) -> MachineConfig:
    if not content:
        raise ValueError("Machine configuration is empty")
    if len(content) > MAX_CONFIG_BYTES:
        raise ValueError("Machine configuration exceeds 64 KB")

    try:
        data = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Machine configuration is not valid UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Machine configuration must be a JSON object")
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError(f"Unknown machine configuration field: {sorted(unknown)[0]}")
    if data.get("format") != FORMAT:
        raise ValueError(f'Machine configuration format must be "{FORMAT}"')
    if data.get("version") != VERSION:
        raise ValueError(
            f"Unsupported machine configuration version: {data.get('version')}"
        )

    name = data.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ValueError("Machine name must be between 1 and 100 characters")

    max_normal_error_degrees = _positive_number(data, "max_normal_error_degrees", 1.0)
    if max_normal_error_degrees > 90.0:
        raise ValueError("max_normal_error_degrees must not exceed 90")
    b_degrees_min = _number(data, "b_degrees_min", -180.0)
    b_degrees_max = _number(data, "b_degrees_max", 180.0)
    if b_degrees_min >= b_degrees_max:
        raise ValueError("b_degrees_min must be less than b_degrees_max")

    return MachineConfig(
        name=name.strip(),
        build_volume_mm=_vector(data, "build_volume_mm", positive=True),
        machine_plate_center_mm=_vector(data, "machine_plate_center_mm"),
        rotation_center_machine_mm=_vector(data, "rotation_center_machine_mm"),
        a_max_velocity_deg_s=_positive_number(data, "a_max_velocity_deg_s", 200.0),
        a_max_acceleration_deg_s2=_positive_number(
            data, "a_max_acceleration_deg_s2", 1000.0
        ),
        b_max_velocity_deg_s=_positive_number(data, "b_max_velocity_deg_s", 200.0),
        b_max_acceleration_deg_s2=_positive_number(
            data, "b_max_acceleration_deg_s2", 1000.0
        ),
        max_normal_error_degrees=max_normal_error_degrees,
        b_degrees_min=b_degrees_min,
        b_degrees_max=b_degrees_max,
    )


def save_machine_config(config: MachineConfig) -> bytes:
    data = {"format": FORMAT, "version": VERSION, **asdict(config)}
    return (json.dumps(data, indent=2) + "\n").encode()


def _number(data: dict, field: str, default: float) -> float:
    value = data.get(field, default)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _positive_number(data: dict, field: str, default: float) -> float:
    value = _number(data, field, default)
    if not 0 < value <= 10_000:
        raise ValueError(f"{field} must be a positive number")
    return value


def _vector(
    data: dict,
    field: str,
    *,
    positive: bool = False,
) -> tuple[float, float, float]:
    value = data.get(field)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three numbers")
    if any(type(number) not in (int, float) for number in value):
        raise ValueError(f"{field} must contain exactly three numbers")

    vector = tuple(float(number) for number in value)
    if not all(math.isfinite(number) and abs(number) <= 10_000 for number in vector):
        raise ValueError(f"{field} contains an invalid coordinate")
    if positive and not all(0 < number <= 10_000 for number in vector):
        raise ValueError(f"{field} dimensions must be positive")
    return vector
