import numpy as np
import pytest

from slice_tools import Slicer


@pytest.mark.parametrize(
    ("normal", "expected"),
    [
        (None, (0.0, 0.0)),
        (np.array([0.0, 0.0, 1.0]), (0.0, 0.0)),
        (np.array([0.0, 0.0, -1.0]), (180.0, 0.0)),
        (np.array([-1.0, 0.0, 0.0]), (90.0, 0.0)),
        (np.array([0.0, -1.0, 0.0]), (90.0, 90.0)),
    ],
)
def test_ab_angles_for_known_normals(
    normal: np.ndarray | None,
    expected: tuple[float, float],
) -> None:
    assert Slicer.ab_angles(normal) == pytest.approx(expected)
