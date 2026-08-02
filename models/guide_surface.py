from dataclasses import dataclass

import numpy as np

GUIDE_HALF_SIZE = 50.0
GUIDE_GRID_SIZE = 11


@dataclass
class GuideSurfaceSnapshot:
    position: np.ndarray
    wxyz: np.ndarray
    bend_x: float = 0.0
    bend_y: float = 0.0
    guide_id: int | None = None


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
