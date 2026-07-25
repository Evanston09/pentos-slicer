from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class PlaneSnapshot:
    position: np.ndarray
    wxyz: np.ndarray
    plane_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            position=np.array(data["position"]),
            wxyz=np.array(data["wxyz"]),
        )

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "position": self.position.tolist(),
            "wxyz": self.wxyz.tolist(),
        }
