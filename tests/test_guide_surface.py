import numpy as np
from numpy.testing import assert_allclose
import pytest
from trimesh import transformations as tf

from models.guide_surface import (
    GUIDE_GRID_SIZE,
    GuideSurfaceSnapshot,
    guide_surface_mesh,
    tween_surface_meshes,
)


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


def guide(guide_id: int, z: float, wxyz: np.ndarray | None = None):
    return GuideSurfaceSnapshot(
        position=np.array([0.0, 0.0, z]),
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]) if wxyz is None else wxyz,
        guide_id=guide_id,
    )


def test_tweens_follow_guide_id_order_across_multiple_guides() -> None:
    meshes = tween_surface_meshes(
        [guide(2, 20.0), guide(0, 0.0), guide(1, 10.0)],
        count=1,
    )

    assert len(meshes) == 2
    assert_allclose(meshes[0][0][:, 2], 5.0)
    assert_allclose(meshes[1][0][:, 2], 15.0)


def test_tween_chooses_shortest_whole_grid_rotation() -> None:
    rotated = guide(1, 0.0, tf.quaternion_about_axis(np.pi / 2.0, [0, 0, 1]))
    vertices, _ = guide_surface_mesh(0.0, 0.0)

    meshes = tween_surface_meshes([guide(0, 0.0), rotated], count=1)

    assert_allclose(meshes[0][0], vertices, atol=1e-12)


def test_tween_rejects_opposing_guide_normals() -> None:
    upside_down = guide(1, 10.0, tf.quaternion_about_axis(np.pi, [1, 0, 0]))

    with pytest.raises(ValueError, match="faces away"):
        tween_surface_meshes([guide(0, 0.0), upside_down])
