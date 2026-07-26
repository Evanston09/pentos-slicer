from numpy.testing import assert_allclose
import trimesh

from models import AppState
from services.model_tools import (
    load_model,
    normalize_mesh_units,
    model_within_build_volume,
    placed_model,
)


def test_normalize_mesh_units_preserves_millimeters() -> None:
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    mesh.units = "mm"

    normalize_mesh_units(mesh)

    assert mesh.units == "mm"
    assert_allclose(mesh.extents, [10.0, 20.0, 30.0])


def test_normalize_mesh_units_converts_small_unknown_mesh_to_mm() -> None:
    mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    mesh.units = None

    normalize_mesh_units(mesh)

    assert mesh.units == "mm"
    assert_allclose(mesh.extents, [100.0, 200.0, 300.0])


def test_load_model_centers_on_plate_and_places_base_at_zero(tmp_path) -> None:
    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=[10.0, 20.0, 30.0]).export(path)

    mesh = load_model(path)

    assert_allclose(mesh.bounds.mean(axis=0)[:2], [45.0, 45.0])
    assert mesh.bounds[0][2] == 0.0


def test_placed_model_does_not_mutate_stored_mesh() -> None:
    base = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    base.apply_translation([45.0, 45.0, 15.0])
    original_vertices = base.vertices.copy()
    state = AppState(
        current_model=(base, "box"),
        model_xy_position=[20.0, 30.0],
        model_z_degrees=90.0,
    )

    result = placed_model(state)

    assert result is not None
    placed, name = result
    assert name == "box"
    assert_allclose(placed.bounds.mean(axis=0)[:2], [20.0, 30.0])
    assert_allclose(base.vertices, original_vertices)


def test_model_within_build_volume_checks_all_three_axes() -> None:
    inside = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    inside.apply_translation([45.0, 45.0, 5.0])
    assert model_within_build_volume(inside)

    outside_xy = inside.copy()
    outside_xy.apply_translation([41.0, 0.0, 0.0])
    assert not model_within_build_volume(outside_xy)

    outside_z = inside.copy()
    outside_z.apply_translation([0.0, 0.0, 86.0])
    assert not model_within_build_volume(outside_z)
