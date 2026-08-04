from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
import trimesh

from models import GuideSurfaceSnapshot
from services.auto_planes import quaternion_from_z_to
from services.model_tools import load_model
from services.volumetric_deformation import (
    guide_target_rotations,
    solve_deformation,
    solve_guide_deformation,
    tetrahedralize,
)


def test_tetrahedralize_builds_identity_volume_with_matching_boundary() -> None:
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 30.0])

    volume = tetrahedralize(mesh)

    assert volume.tetrahedra.shape[1] == 4
    assert len(volume.tetrahedra) > 0
    assert_allclose(volume.deformed_vertices, volume.original_vertices)
    assert not np.shares_memory(volume.deformed_vertices, volume.original_vertices)

    points = volume.original_vertices[volume.tetrahedra]
    signed_volumes = (
        np.linalg.det(
            np.stack(
                (
                    points[:, 1] - points[:, 0],
                    points[:, 2] - points[:, 0],
                    points[:, 3] - points[:, 0],
                ),
                axis=2,
            )
        )
        / 6.0
    )
    assert np.all(signed_volumes > 0.0)

    boundary = trimesh.Trimesh(
        vertices=volume.original_vertices,
        faces=volume.boundary_faces,
        process=False,
    )
    assert boundary.is_volume
    assert_allclose(boundary.bounds, mesh.bounds)
    assert_allclose(boundary.volume, mesh.volume)


def guide(
    guide_id: int,
    position: Sequence[float],
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    bend_x: float = 0.0,
    bend_y: float = 0.0,
) -> GuideSurfaceSnapshot:
    return GuideSurfaceSnapshot(
        position=np.asarray(position, dtype=float),
        wxyz=quaternion_from_z_to(np.asarray(normal, dtype=float)),
        guide_id=guide_id,
        bend_x=bend_x,
        bend_y=bend_y,
    )


def test_flat_guide_preserves_identity() -> None:
    volume = tetrahedralize(trimesh.creation.box())

    rotations = guide_target_rotations(volume, [guide(0, [0.0, 0.0, 0.0])])
    solve_deformation(volume, rotations)

    assert_allclose(volume.deformed_vertices, volume.original_vertices, atol=1e-6)


def test_guide_deformation_maps_guides_to_flat_heights() -> None:
    volume = tetrahedralize(trimesh.creation.box())

    solve_guide_deformation(
        volume,
        [guide(0, [0.0, 0.0, -0.5]), guide(1, [0.0, 0.0, 0.5])],
    )

    assert_allclose(
        volume.deformed_vertices[:, 2],
        volume.original_vertices[:, 2] + 0.5,
    )


def test_uniform_tilt_rotates_full_xyz_volume() -> None:
    volume = tetrahedralize(trimesh.creation.box())
    normal = np.array([1.0, 0.0, 2.0]) / np.sqrt(5.0)
    rotations = guide_target_rotations(
        volume,
        [guide(0, [0.0, 0.0, 0.0], normal.tolist())],
    )
    anchor = volume.original_vertices[0]

    solve_deformation(volume, rotations)

    expected = (volume.original_vertices - anchor) @ rotations[0].T + anchor
    assert_allclose(volume.deformed_vertices, expected, atol=1e-4)


def test_full_xyz_deformation_handles_large_guide_changes_on_cad_mesh() -> None:
    mesh = load_model(Path("samples/Part Studio 1.stl"))
    volume = tetrahedralize(mesh)
    guides = [
        guide(0, [45.0, 45.0, 0.0]),
        guide(1, [45.0, 45.0, 18.86], [-0.766044, 0.0, 0.642788], bend_y=0.002),
        guide(2, [32.739, 43.125, 36.764], [-0.996195, 0.0, 0.087156]),
    ]

    rotations = guide_target_rotations(volume, guides)
    solve_deformation(volume, rotations)

    shell = trimesh.Trimesh(
        vertices=volume.deformed_vertices,
        faces=volume.boundary_faces,
        process=False,
    )
    displacement = np.linalg.norm(
        volume.deformed_vertices - volume.original_vertices,
        axis=1,
    )
    assert shell.is_volume
    assert np.median(displacement) > 5.0


def test_duplicate_guide_positions_are_rejected() -> None:
    volume = tetrahedralize(trimesh.creation.box())

    with pytest.raises(ValueError, match="different positions"):
        guide_target_rotations(
            volume,
            [guide(0, [0.0, 0.0, 0.0]), guide(1, [0.0, 0.0, 0.0])],
        )


def test_barycentric_mapping_round_trips_affine_deformation() -> None:
    volume = tetrahedralize(trimesh.creation.box())
    source_points = np.array(
        [
            volume.original_vertices[volume.tetrahedra[0]].mean(axis=0),
            volume.original_vertices[0],
            [0.0, 0.0, 0.0],
        ]
    )
    transform = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.5],
        ]
    )
    offset = np.array([2.0, -1.0, 3.0])
    volume.deformed_vertices = volume.original_vertices @ transform.T + offset

    deformed_points = volume.map_to_deformed(source_points)

    assert_allclose(deformed_points, source_points @ transform.T + offset)
    assert_allclose(volume.map_to_original(deformed_points), source_points, atol=1e-12)
    locations = volume.locate_original(source_points)
    assert_allclose(locations.weights.sum(axis=1), 1.0)
    assert np.all(locations.weights >= 0.0)


def test_barycentric_mapping_rejects_outside_points() -> None:
    volume = tetrahedralize(trimesh.creation.box())

    with pytest.raises(ValueError, match="Point 0 is outside"):
        volume.map_to_deformed(np.array([[2.0, 2.0, 2.0]]))


def test_barycentric_mapping_allows_s4_style_inverted_cells() -> None:
    volume = tetrahedralize(trimesh.creation.box())
    point = np.array([[0.1, 0.2, 0.3]])
    volume.deformed_vertices[:, 0] *= -1.0

    deformed = volume.map_to_deformed(point)

    assert_allclose(volume.map_to_original(deformed), point)


def test_barycentric_mapping_requires_point_rows() -> None:
    volume = tetrahedralize(trimesh.creation.box())

    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        volume.map_to_deformed(np.zeros(3))


def test_tetrahedralize_preserves_dense_cad_boundary() -> None:
    mesh = load_model(Path("samples/Part Studio 1.stl"))

    volume = tetrahedralize(mesh)

    assert len(volume.tetrahedra) > 0
    assert len(volume.boundary_faces) == len(mesh.faces)


def test_tetrahedralize_rejects_open_mesh() -> None:
    box = trimesh.creation.box()
    open_mesh = trimesh.Trimesh(
        vertices=box.vertices,
        faces=box.faces[:-1],
        process=False,
    )

    with pytest.raises(ValueError, match="watertight"):
        tetrahedralize(open_mesh)
