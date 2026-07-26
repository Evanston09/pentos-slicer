import numpy as np
import trimesh
from numpy.testing import assert_allclose

from services.auto_planes import (
    AutoPlaneCandidate,
    _plane_intersection_has_bed_clearance,
    overhang_preview_mesh,
)


def translated_box(min_z: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation([0.0, 0.0, min_z + 1.0])
    return mesh


def test_overhang_preview_uses_decomposed_mesh() -> None:
    mesh = translated_box(min_z=0.0)
    plane = AutoPlaneCandidate(np.zeros(3), np.array([1.0, 0.0, 0.0]))

    preview, overhangs = overhang_preview_mesh(mesh, [plane], 40.0)

    assert len(preview.faces) > len(mesh.faces)
    assert not overhangs.any()


def test_overhang_preview_keeps_unsupported_faces() -> None:
    floor = translated_box(min_z=0.0)
    floor.apply_translation([4.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([floor, translated_box(min_z=10.0)])

    preview, overhangs = overhang_preview_mesh(mesh, [], 40.0)

    assert_allclose(preview.area_faces[overhangs].sum(), 4.0)


def test_plane_intersection_requires_nozzle_bed_clearance() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 10.0))
    mesh.apply_translation([0.0, 0.0, 5.0])
    position = np.array([0.0, 0.0, 6.0])
    unsafe_plane = AutoPlaneCandidate(position, np.array([1.0, 0.0, 0.0]))
    safe_plane = AutoPlaneCandidate(position, np.array([0.0, 0.0, 1.0]))

    assert not _plane_intersection_has_bed_clearance(mesh, unsafe_plane)
    assert _plane_intersection_has_bed_clearance(mesh, safe_plane)
