import numpy as np
import pytest

from machine import rotation_matrix
from services.slicing import DEFAULT_CONFIG_PATH, Slicer


def test_default_config_path_points_to_repository_profile() -> None:
    assert DEFAULT_CONFIG_PATH.name == "pentos_config.ini"
    assert DEFAULT_CONFIG_PATH.is_file()


@pytest.mark.parametrize(
    ("normal", "expected"),
    [
        (None, (0.0, 0.0)),
        (np.array([0.0, 0.0, 1.0]), (0.0, 0.0)),
        (np.array([0.0, 0.0, -1.0]), (180.0, 0.0)),
        (np.array([-1.0, 0.0, 0.0]), (90.0, 0.0)),
        (np.array([1.0, 0.0, 0.0]), (90.0, -180.0)),
        (np.array([0.0, -1.0, 0.0]), (90.0, 90.0)),
        (np.array([0.0, 1.0, 0.0]), (90.0, -90.0)),
        (np.array([1.0, 0.0, 1.0]), (45.0, -180.0)),
        (np.array([-1.0, 0.0, 1.0]), (45.0, 0.0)),
        (np.array([1.0, 1.0, 0.0]), (90.0, -135.0)),
    ],
)
def test_ab_angles_for_known_normals(
    normal: np.ndarray | None,
    expected: tuple[float, float],
) -> None:
    a, b = Slicer.ab_angles(normal)
    assert (a, b) == pytest.approx(expected)
    if normal is not None:
        unit_normal = normal / np.linalg.norm(normal)
        np.testing.assert_allclose(
            rotation_matrix(a, b) @ unit_normal, [0.0, 0.0, 1.0], atol=1e-12
        )


@pytest.mark.parametrize(
    ("index", "total", "expected"),
    [
        (0, 1, []),
        (
            0,
            3,
            ["--top-solid-layers", "0", "--top-solid-min-thickness", "0"],
        ),
        (
            1,
            3,
            [
                "--top-solid-layers",
                "0",
                "--top-solid-min-thickness",
                "0",
                "--bottom-solid-layers",
                "0",
                "--bottom-solid-min-thickness",
                "0",
            ],
        ),
        (
            2,
            3,
            ["--bottom-solid-layers", "0", "--bottom-solid-min-thickness", "0"],
        ),
    ],
)
def test_chunk_shell_overrides(
    index: int,
    total: int,
    expected: list[str],
) -> None:
    assert Slicer._chunk_shell_overrides(index, total) == expected
