import numpy as np
from numpy.testing import assert_allclose

from machine import rotation_matrix


def test_rotation_matrix_identity() -> None:
    assert_allclose(rotation_matrix(0.0, 0.0), np.eye(3))


def test_positive_a_lifts_negative_x_side() -> None:
    rotation = rotation_matrix(90.0, 0.0)

    assert_allclose(
        rotation @ np.array([-1.0, 0.0, 0.0]),
        [0.0, 0.0, 1.0],
        atol=1e-7,
    )
    assert_allclose(
        rotation @ np.array([1.0, 0.0, 0.0]),
        [0.0, 0.0, -1.0],
        atol=1e-7,
    )


def test_positive_b_moves_positive_x_toward_front() -> None:
    rotation = rotation_matrix(0.0, 90.0)

    assert_allclose(
        rotation @ np.array([1.0, 0.0, 0.0]),
        [0.0, -1.0, 0.0],
        atol=1e-7,
    )
