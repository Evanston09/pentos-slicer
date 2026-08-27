from dataclasses import dataclass


@dataclass(frozen=True)
class MachineConfig:
    name: str = "Default Pentos"
    build_volume_mm: tuple[float, float, float] = (90.0, 90.0, 90.0)
    machine_plate_center_mm: tuple[float, float, float] = (113.0, 52.0, 0.0)
    rotation_center_machine_mm: tuple[float, float, float] = (112.0, 51.0, 2.0)
    a_max_velocity_deg_s: float = 200.0
    a_max_acceleration_deg_s2: float = 1000.0
    b_max_velocity_deg_s: float = 200.0
    b_max_acceleration_deg_s2: float = 1000.0
    max_normal_error_degrees: float = 1.0
    b_degrees_min: float = -180.0
    b_degrees_max: float = 180.0

    @property
    def build_plate_center(self) -> tuple[float, float, float]:
        return (
            self.build_volume_mm[0] / 2.0,
            self.build_volume_mm[1] / 2.0,
            0.0,
        )

    @property
    def machine_offset(self) -> tuple[float, float, float]:
        return tuple(
            machine - local
            for machine, local in zip(
                self.machine_plate_center_mm, self.build_plate_center
            )
        )

    @property
    def rotation_center_local_mm(self) -> tuple[float, float, float]:
        return tuple(
            machine - offset
            for machine, offset in zip(
                self.rotation_center_machine_mm, self.machine_offset
            )
        )


DEFAULT_MACHINE_CONFIG = MachineConfig()
