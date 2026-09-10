from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.polynomial import Polynomial
from trimesh import transformations as tf

GUIDE_CONTROL_SIZE = 4
GUIDE_PREVIEW_SIZE = 30
GUIDE_HALF_SIZE = 50.0


def _validate_size(size_mm: np.ndarray) -> np.ndarray:
    size = np.asarray(size_mm)
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("Guide size_mm must contain two finite positive values")
    return size


def _validate_heights(heights_mm: np.ndarray) -> np.ndarray:
    heights = np.asarray(heights_mm)
    if heights.shape != (GUIDE_CONTROL_SIZE, GUIDE_CONTROL_SIZE):
        raise ValueError("Guide heights_mm must be a 4x4 grid")
    if not np.all(np.isfinite(heights)):
        raise ValueError("Guide heights_mm must contain only finite values")
    return heights


@dataclass
class GuideSurfaceSnapshot:
    position: np.ndarray
    wxyz: np.ndarray
    guide_id: int
    size_mm: np.ndarray = field(
        default_factory=lambda: np.repeat(2.0 * GUIDE_HALF_SIZE, 2)
    )
    heights_mm: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position)
        self.wxyz = np.asarray(self.wxyz)
        self.size_mm = _validate_size(self.size_mm)
        self.heights_mm = _validate_heights(self.heights_mm)

    @classmethod
    def from_dict(cls, data: dict, guide_id: int) -> Self:
        return cls(
            position=np.array(data["position"]),
            wxyz=np.array(data["wxyz"]),
            guide_id=guide_id,
            size_mm=np.array(data["size_mm"]),
            heights_mm=np.array(data["heights_mm"]),
        )

    def as_dict(self) -> dict[str, list]:
        return {
            "position": self.position.tolist(),
            "wxyz": self.wxyz.tolist(),
            "size_mm": self.size_mm.tolist(),
            "heights_mm": self.heights_mm.tolist(),
        }

    @property
    def rotation(self) -> np.ndarray:
        return tf.quaternion_matrix(self.wxyz)[:3, :3]

    @property
    def control_xy(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.linspace(-self.size_mm[0] / 2.0, self.size_mm[0] / 2.0, 4),
            np.linspace(-self.size_mm[1] / 2.0, self.size_mm[1] / 2.0, 4),
        )

    # TOOD: Learn what is going on past this point

    def evaluate_local(
        self, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate height and its local X/Y derivatives at arbitrary XY points."""
        points = np.asarray(xy, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError("Guide evaluation points must end in an XY pair")
        flat = points.reshape(-1, 2)
        x_coordinates, y_coordinates = self.control_xy
        basis_x, derivative_basis_x = _cubic_basis(x_coordinates, flat[:, 0])
        basis_y, derivative_basis_y = _cubic_basis(y_coordinates, flat[:, 1])
        heights = np.einsum("ni,ij,nj->n", basis_y, self.heights_mm, basis_x)
        derivatives_x = np.einsum(
            "ni,ij,nj->n", basis_y, self.heights_mm, derivative_basis_x
        )
        derivatives_y = np.einsum(
            "ni,ij,nj->n", derivative_basis_y, self.heights_mm, basis_x
        )
        shape = points.shape[:-1]
        return (
            heights.reshape(shape),
            derivatives_x.reshape(shape),
            derivatives_y.reshape(shape),
        )

    def world_normal(self, xy: np.ndarray) -> np.ndarray:
        _, dx, dy = self.evaluate_local(xy)
        normal = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1) @ self.rotation.T
        return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    def signed_height(self, world_points: np.ndarray) -> np.ndarray:
        points = np.asarray(world_points, dtype=np.float64)
        local = (points - self.position) @ self.rotation
        height, _, _ = self.evaluate_local(local[..., :2])
        return local[..., 2] - height

    def normalize_center(self) -> None:
        """Move the pose origin onto the sheet without changing visible geometry."""
        center_height = float(self.evaluate_local(np.zeros((1, 2)))[0][0])
        self.heights_mm = self.heights_mm - center_height
        self.position = self.position + self.rotation[:, 2] * center_height

    def fit_heights(self, xy: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Least-squares fit the 4x4 control grid to sampled local XY heights."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        values = np.asarray(heights, dtype=np.float64).reshape(-1)
        if points.shape[0] != values.shape[0]:
            raise ValueError("Guide fit samples and heights must have equal length")
        if len(points) < GUIDE_CONTROL_SIZE**2:
            raise ValueError("Surface samples do not determine a full 4x4 guide fit")
        # Scale to the samples so small faces remain well-conditioned on a large guide.
        center = points.mean(axis=0)
        scale = np.ptp(points, axis=0)
        scale[scale == 0.0] = 1.0
        normalized = (points - center) / scale
        design = np.polynomial.polynomial.polyvander2d(
            normalized[:, 0], normalized[:, 1], [3, 3]
        )
        fitted, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
        if rank < GUIDE_CONTROL_SIZE**2:
            raise ValueError("Surface samples do not determine a full 4x4 guide fit")
        x, y = np.meshgrid(*self.control_xy)
        return np.polynomial.polynomial.polyval2d(
            (x - center[0]) / scale[0],
            (y - center[1]) / scale[1],
            fitted.reshape(GUIDE_CONTROL_SIZE, GUIDE_CONTROL_SIZE),
        )

    def apply_bend_preset(self, bend_x: float, bend_y: float) -> None:
        x, y = self.control_xy
        grid_x, grid_y = np.meshgrid(x, y)
        self.heights_mm = bend_x * grid_x**2 + bend_y * grid_y**2
        self.normalize_center()

    def preview_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        resolution = GUIDE_PREVIEW_SIZE
        x = np.linspace(-self.size_mm[0] / 2.0, self.size_mm[0] / 2.0, resolution)
        y = np.linspace(-self.size_mm[1] / 2.0, self.size_mm[1] / 2.0, resolution)
        grid_x, grid_y = np.meshgrid(x, y)
        height, _, _ = self.evaluate_local(np.stack((grid_x, grid_y), axis=-1))
        vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), height.ravel()))
        cells = np.arange(resolution**2).reshape(resolution, resolution)
        faces = np.stack(
            (
                np.stack((cells[:-1, :-1], cells[:-1, 1:], cells[1:, 1:]), axis=-1),
                np.stack((cells[:-1, :-1], cells[1:, 1:], cells[1:, :-1]), axis=-1),
            ),
            axis=-2,
        )
        return vertices, faces.reshape(-1, 3)


def _cubic_basis(
    coordinates: np.ndarray, samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return cubic Lagrange basis values and derivatives for four nodes."""
    values = np.empty((len(samples), 4))
    derivatives = np.empty_like(values)
    for control in range(4):
        others = np.delete(coordinates, control)
        polynomial = Polynomial.fromroots(others) / np.prod(
            coordinates[control] - others
        )
        values[:, control] = polynomial(samples)
        derivatives[:, control] = polynomial.deriv()(samples)
    return values, derivatives
