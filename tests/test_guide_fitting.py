import numpy as np
from numpy.testing import assert_allclose
import pytest
import trimesh

from dev.generate_flat_base_curved_top import make_mesh
from models import GuideSurfaceSnapshot
import models.guide_surface as guide_surface_module
from services.guide_fitting import align_guide_to_surface


def test_alignment_fits_concave_surface_above_guide_plane() -> None:
    mesh = make_mesh(top_height=10.0, curvature=-0.0025, segments=80)
    guide = GuideSurfaceSnapshot(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 4, np.array([40.0, 20.0])
    )
    original = guide.as_dict()
    aligned = align_guide_to_surface(
        mesh, guide, np.array([45.0, 45.0, 20.0]), np.array([0.0, 0.0, -1.0])
    )

    assert aligned is not None
    assert guide.as_dict() == original
    assert_allclose(aligned.position, [45.0, 45.0, 10.0], atol=1e-8)
    xy = np.array([[-18.0, -8.0], [0.0, 0.0], [18.0, 8.0]])
    assert_allclose(aligned.evaluate_local(xy)[0], 0.0025 * xy[:, 0] ** 2, atol=0.001)


def test_alignment_ignores_other_components_and_preview_resolution(monkeypatch) -> None:
    mesh = make_mesh(top_height=10.0, curvature=0.0025, segments=80)
    # Beside the selected patch, this slab is closer to its plane than its edges.
    slab = trimesh.creation.box(extents=[70.0, 50.0, 1.0])
    slab.apply_translation([45.0, 45.0, 9.0])
    combined = trimesh.util.concatenate([mesh, slab])
    guide = GuideSurfaceSnapshot(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0, np.array([60.0, 40.0])
    )
    origin, direction = np.array([45.0, 45.0, 20.0]), np.array([0.0, 0.0, -1.0])
    expected = align_guide_to_surface(mesh, guide, origin, direction)
    monkeypatch.setattr(guide_surface_module, "GUIDE_PREVIEW_SIZE", 7)
    actual = align_guide_to_surface(combined, guide, origin, direction)

    assert actual is not None and expected is not None
    assert_allclose(actual.position, expected.position)
    assert_allclose(actual.heights_mm, expected.heights_mm, atol=1e-8)


@pytest.mark.parametrize("sample_count", [0, 4, 30])
def test_fit_rejects_insufficient_or_collinear_samples(sample_count) -> None:
    guide = GuideSurfaceSnapshot(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0)
    xy = np.column_stack(
        (np.linspace(-10.0, 10.0, sample_count), np.zeros(sample_count))
    )
    with pytest.raises(ValueError, match="full 4x4 guide fit"):
        guide.fit_heights(xy, np.zeros(sample_count))
