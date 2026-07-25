import numpy as np
import trimesh
from numpy.testing import assert_allclose

from services.auto_planes import (
    AutoPlaneCandidate,
    face_print_directions,
    overhang_face_mask,
)


def translated_box(min_z: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation([0.0, 0.0, min_z + 1.0])
    return mesh


def test_overhang_face_mask() -> None:
    mesh = translated_box(min_z=1.0)

    risky = overhang_face_mask(mesh, None, 40.0)

    assert np.array_equal(risky, mesh.face_normals[:, 2] < -0.9)


def test_overhang_face_mask_uses_each_faces_print_direction() -> None:
    mesh = translated_box(min_z=1.0)

    risky = overhang_face_mask(mesh, -mesh.face_normals, 40.0)

    assert np.all(risky)


def test_overhang_face_mask_excludes_bed_contact() -> None:
    mesh = translated_box(min_z=0.0)

    assert not np.any(overhang_face_mask(mesh, None, 40.0))


def test_face_print_directions_follow_plane_normal_side() -> None:
    mesh = translated_box(min_z=1.0)
    plane = AutoPlaneCandidate(
        position=np.zeros(3),
        normal=np.array([1.0, 0.0, 0.0]),
    )

    directions = face_print_directions(mesh, [plane])
    positive_x = mesh.triangles_center[:, 0] >= 0.0

    assert_allclose(directions[positive_x], [[1.0, 0.0, 0.0]] * positive_x.sum())
    assert_allclose(directions[~positive_x], [[0.0, 0.0, 1.0]] * (~positive_x).sum())
