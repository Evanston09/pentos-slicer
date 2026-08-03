from dataclasses import dataclass

import numpy as np
from trimesh import transformations as tf

GUIDE_HALF_SIZE = 50.0
GUIDE_GRID_SIZE = 11
DEFAULT_TWEEN_SURFACES_PER_PAIR = 4


@dataclass
class GuideSurfaceSnapshot:
    position: np.ndarray
    wxyz: np.ndarray
    guide_id: int
    bend_x: float = 0.0
    bend_y: float = 0.0


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


def tween_surface_meshes(
    guides: list[GuideSurfaceSnapshot],
    count: int = DEFAULT_TWEEN_SURFACES_PER_PAIR,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tween adjacent guides by ID using their shortest whole-grid alignment."""
    ordered = sorted(guides, key=lambda guide: guide.guide_id)
    if len(ordered) < 2:
        return []

    index_grid = np.arange(GUIDE_GRID_SIZE**2).reshape(GUIDE_GRID_SIZE, GUIDE_GRID_SIZE)
    rotations = [np.rot90(index_grid, turns).ravel() for turns in range(4)]

    previous, previous_normal = _world_geometry(ordered[0])
    _, faces = guide_surface_mesh(0.0, 0.0)
    tweens = []

    for guide in ordered[1:]:
        vertices, normal = _world_geometry(guide)

        if previous_normal @ normal <= 0.0:
            raise ValueError(
                f"Guide {guide.guide_id} faces away from the previous guide"
            )

        # Find the shortest whole-grid rotation.
        current = min(
            (vertices[rotation] for rotation in rotations),
            key=lambda candidate: np.sum((candidate - previous) ** 2),
        )

        for step in range(1, count + 1):
            amount = step / (count + 1)
            tweens.append(((1.0 - amount) * previous + amount * current, faces))

        previous = current
        previous_normal = normal

    return tweens


# TODO: Perhaps add arrow later to visualize to better understand the normal
def _world_geometry(guide: GuideSurfaceSnapshot) -> tuple[np.ndarray, np.ndarray]:
    vertices, _ = guide_surface_mesh(guide.bend_x, guide.bend_y)
    rotation = tf.quaternion_matrix(guide.wxyz)[:3, :3]
    return vertices @ rotation.T + guide.position, rotation[:, 2]
