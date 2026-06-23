import numpy as np

MACHINE_XY_SIZE = 235.0
BUILD_PLATE_SIZE = 90.0
BUILD_PLATE_CENTER = np.array([BUILD_PLATE_SIZE / 2.0, BUILD_PLATE_SIZE / 2.0, 0.0])
MACHINE_BUILD_PLATE_CENTER = np.array([113.0, 52.0, 0.0])
MACHINE_OFFSET = MACHINE_BUILD_PLATE_CENTER - BUILD_PLATE_CENTER
ROTATION_CENTER = BUILD_PLATE_CENTER.copy()


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
            [np.cos(b), np.sin(b), 0.0],
            [-np.sin(b), np.cos(b), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    return twist @ tilt
