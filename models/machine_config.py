from dataclasses import dataclass


@dataclass(frozen=True)
class MachineConfig:
    name: str = "Default Pentos"
    build_volume_mm: tuple[float, float, float] = (90.0, 90.0, 90.0)
    machine_plate_center_mm: tuple[float, float, float] = (113.0, 52.0, 0.0)
    rotation_center_machine_mm: tuple[float, float, float] = (112.0, 51.0, 2.0)

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
