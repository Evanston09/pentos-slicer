import numpy as np

BUILD_PLATE_SIZE = 90.0
BUILD_PLATE_CENTER = np.array([BUILD_PLATE_SIZE / 2.0, BUILD_PLATE_SIZE / 2.0, 0.0])


def rotation_matrix(a_degrees: float, b_degrees: float) -> np.ndarray:
    a = np.radians(a_degrees)
    b = np.radians(b_degrees)

    tilt = np.array(
        [
            [np.cos(a), 0.0, np.sin(a)],
            [0.0, 1.0, 0.0],
            [-np.sin(a), 0.0, np.cos(a)],
        ],
    )
    twist = np.array(
        [
            [np.cos(b), -np.sin(b), 0.0],
            [np.sin(b), np.cos(b), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    return twist @ tilt
