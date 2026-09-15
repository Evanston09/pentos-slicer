import numpy as np
from numpy.testing import assert_allclose

from models import GUIDE_PREVIEW_SIZE, GuideSurfaceSnapshot
from services.auto_planes import quaternion_from_z_to


def make_guide() -> GuideSurfaceSnapshot:
    guide = GuideSurfaceSnapshot(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        7,
        np.array([12.0, 18.0]),
    )
    x, y = guide.control_xy
    grid_x, grid_y = np.meshgrid(x, y)
    guide.heights_mm = 0.02 * grid_x**2 - 0.01 * grid_y**2 + 0.2 * grid_x
    return guide


def test_grid_interpolates_every_control_and_derivatives() -> None:
    guide = make_guide()
    x, y = guide.control_xy
    grid_x, grid_y = np.meshgrid(x, y)
    xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    height, derivative_x, derivative_y = guide.evaluate_local(xy)

    assert_allclose(height, guide.heights_mm.ravel(), atol=1e-12)
    assert_allclose(derivative_x, 0.04 * xy[:, 0] + 0.2, atol=1e-12)
    assert_allclose(derivative_y, -0.02 * xy[:, 1], atol=1e-12)


def test_center_normalization_preserves_world_surface() -> None:
    guide = make_guide()
    points = np.array([[-3.0, -4.0], [0.0, 0.0], [5.0, 7.0]])
    old_height = guide.evaluate_local(points)[0]
    old_position = guide.position.copy()

    guide.normalize_center()

    new_height = guide.evaluate_local(points)[0]
    assert_allclose(guide.evaluate_local([[0.0, 0.0]])[0], 0.0, atol=1e-12)
    assert_allclose(new_height + guide.position[2], old_height + old_position[2])


def test_preview_mesh_uses_preview_resolution_and_consistent_triangulation() -> None:
    vertices, faces = make_guide().preview_mesh()

    assert vertices.shape == (GUIDE_PREVIEW_SIZE**2, 3)
    assert faces.shape == (2 * (GUIDE_PREVIEW_SIZE - 1) ** 2, 3)
    assert np.all(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )[:, 2]
        > 0
    )


def test_bend_preset_and_transformed_world_normal() -> None:
    guide = make_guide()
    guide.apply_bend_preset(0.01, -0.02)
    point = np.array([[2.0, -3.0]])
    height, dx, dy = guide.evaluate_local(point)
    assert_allclose(height, [0.01 * 2.0**2 - 0.02 * 3.0**2])
    assert_allclose(dx, [0.04])
    assert_allclose(dy, [0.12])

    guide.wxyz = quaternion_from_z_to(np.array([1.0, 0.0, 0.0]))
    expected = np.array([[1.0, -0.12, 0.04]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert_allclose(guide.world_normal(point), expected)
