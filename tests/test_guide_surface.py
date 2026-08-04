import numpy as np
from numpy.testing import assert_allclose

from models.guide_surface import GUIDE_GRID_SIZE, guide_surface_mesh


def test_guide_surface_mesh_follows_bend_equation_and_is_triangulated() -> None:
    vertices, faces = guide_surface_mesh(0.01, -0.02)

    assert_allclose(
        vertices[:, 2],
        0.01 * vertices[:, 0] ** 2 - 0.02 * vertices[:, 1] ** 2,
    )
    assert vertices.shape == (GUIDE_GRID_SIZE**2, 3)
    assert faces.shape == (2 * (GUIDE_GRID_SIZE - 1) ** 2, 3)
    assert np.all(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )[:, 2]
        > 0
    )
