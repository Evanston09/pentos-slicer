import math
from dataclasses import asdict, dataclass, field, fields


# Based on default PLA settings for PrusaSlicer Creality Ender 3 V2 as of 9/10/2026
@dataclass(frozen=True)
class FilamentSettings:
    filament_type: str = "PLA"
    temperature: int = 205
    first_layer_temperature: int = 210
    bed_temperature: int = 60
    first_layer_bed_temperature: int = 60

    def __post_init__(self) -> None:
        if self.filament_type not in ("PLA", "PETG"):
            raise ValueError("Filament must be PLA or PETG")
        _validate_numbers(self, skip=("filament_type",))
        for name in ("temperature", "first_layer_temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


# Based on default settings for PrusaSlicer Creality Ender 3 V2 as of 9/10/2026
@dataclass(frozen=True)
class PrintSettings:
    layer_height: float = 0.2
    first_layer_height: float = 0.2
    perimeters: int = 3
    # Named fill_density to match PrusaSlicer's setting name. (it's just infill)
    fill_density: float = 15.0

    def __post_init__(self) -> None:
        _validate_numbers(self)
        # Supported layer-height range of the baseline 0.4 mm nozzle profile.
        for name in ("layer_height", "first_layer_height"):
            if not 0.06 <= getattr(self, name) <= 0.32:
                raise ValueError(f"{name} must be between 0.06 and 0.32 mm")
        if self.fill_density > 100:
            raise ValueError("Infill must be between 0 and 100 percent")


def _validate_numbers(settings, skip: tuple[str, ...] = ()) -> None:
    for item in fields(settings):
        if item.name in skip:
            continue
        value = getattr(settings, item.name)
        if item.type is int:
            valid = type(value) is int and value >= 0
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        if not valid:
            raise ValueError(f"Invalid value for {item.name}")


def filament_preset(material: str) -> FilamentSettings:
    if material == "PLA":
        return FilamentSettings()
    if material == "PETG":
        # PrusaSlicer Creality.ini: Generic PETG -> *PET* -> *common*.
        return FilamentSettings(
            filament_type="PETG",
            temperature=240,
            first_layer_temperature=240,
            bed_temperature=70,
            first_layer_bed_temperature=70,
        )
    raise ValueError("Filament must be PLA or PETG")


@dataclass(frozen=True)
class SlicingSettings:
    filament: FilamentSettings = field(default_factory=FilamentSettings)
    print: PrintSettings = field(default_factory=PrintSettings)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SlicingSettings":
        try:
            if not isinstance(data, dict) or set(data) != {"filament", "print"}:
                raise ValueError("Slicing settings must contain filament and print")
            return cls(
                filament=FilamentSettings(**data["filament"]),
                print=PrintSettings(**data["print"]),
            )
        except TypeError as exc:
            raise ValueError("Invalid slicing settings fields") from exc
