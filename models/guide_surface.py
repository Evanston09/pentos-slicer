from dataclasses import dataclass
from typing import Self

import numpy as np

GUIDE_HALF_SIZE = 50.0
GUIDE_GRID_SIZE = 11


@dataclass
class GuideSurfaceSnapshot:
    position: np.ndarray
    wxyz: np.ndarray
    guide_id: int
    bend_x: float = 0.0
    bend_y: float = 0.0

    @classmethod
    def from_dict(cls, data: dict, guide_id: int) -> Self:
        return cls(
            position=np.array(data["position"]),
            wxyz=np.array(data["wxyz"]),
            guide_id=guide_id,
            bend_x=float(data.get("bend_x", 0.0)),
            bend_y=float(data.get("bend_y", 0.0)),
        )

    def as_dict(self) -> dict[str, list[float] | float]:
        return {
            "position": self.position.tolist(),
            "wxyz": self.wxyz.tolist(),
            "bend_x": self.bend_x,
            "bend_y": self.bend_y,
        }


def guide_surface_mesh(
    bend_x: float,
    bend_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(-GUIDE_HALF_SIZE, GUIDE_HALF_SIZE, GUIDE_GRID_SIZE)
    x, y = np.meshgrid(coordinates, coordinates)
    # In the future maybe add avanced options to add own functions to represent bending
    vertices = np.column_stack(
        (x.ravel(), y.ravel(), bend_x * x.ravel() ** 2 + bend_y * y.ravel() ** 2)
    )

    faces = []
    for row in range(GUIDE_GRID_SIZE - 1):
        for column in range(GUIDE_GRID_SIZE - 1):
            lower_left = row * GUIDE_GRID_SIZE + column
            faces.extend(
                (
                    [lower_left, lower_left + 1, lower_left + GUIDE_GRID_SIZE + 1],
                    [
                        lower_left,
                        lower_left + GUIDE_GRID_SIZE + 1,
                        lower_left + GUIDE_GRID_SIZE,
                    ],
                )
            )
    return vertices, np.array(faces)
