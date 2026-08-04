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
    _tetrahedron_determinants,
    solve_guide_deformation,
    solve_guide_scalar_field,
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


def test_guide_deformation_maps_guides_to_flat_heights() -> None:
    volume = tetrahedralize(trimesh.creation.box())

    solve_guide_deformation(
        volume,
        [guide(0, [0.0, 0.0, -0.5]), guide(1, [0.0, 0.0, 0.5])],
    )

    assert_allclose(
        volume.deformed_vertices[:, 2],
        volume.original_vertices[:, 2] + 0.5,
        atol=1e-8,
    )


def test_curved_guides_produce_an_injective_scalar_flattening() -> None:
    volume = tetrahedralize(trimesh.creation.box(extents=[2.0, 2.0, 1.0]))
    guides = [
        guide(0, [0.0, 0.0, -0.4], bend_x=0.05),
        guide(1, [0.0, 0.0, 0.4], bend_x=0.05),
    ]

    solve_guide_deformation(volume, guides)

    assert np.all(
        _tetrahedron_determinants(volume.deformed_vertices, volume.tetrahedra) > 0.0
    )
    normals = volume.layer_normals([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert normals[0, 0] > 0.0 > normals[1, 0]


def test_harmonic_field_derives_heights_and_normals_from_one_solution() -> None:
    volume = tetrahedralize(trimesh.creation.box())
    guides = [guide(0, [0.0, 0.0, -0.5]), guide(1, [0.0, 0.0, 0.5])]

    heights = solve_guide_scalar_field(volume, guides)

    assert_allclose(heights, volume.original_vertices[:, 2] + 0.5, atol=1e-5)
    assert_allclose(
        volume.layer_normals([[0.0, 0.0, 0.0]]),
        [[0.0, 0.0, 1.0]],
        atol=1e-5,
    )


def test_duplicate_guide_positions_are_rejected() -> None:
    volume = tetrahedralize(trimesh.creation.box())

    with pytest.raises(ValueError, match="different positions"):
        solve_guide_scalar_field(
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


def test_barycentric_mapping_tolerates_rounded_boundary_points() -> None:
    volume = tetrahedralize(trimesh.creation.box())
    point = volume.original_vertices[[0]].copy()
    point[0, 2] -= 5e-7

    mapped = volume.map_to_deformed(point)

    assert_allclose(mapped, volume.original_vertices[[0]], atol=1e-6)


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
