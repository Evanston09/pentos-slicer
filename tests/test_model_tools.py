import pytest
from numpy.testing import assert_allclose
import trimesh

import services.model_tools as model_tools_module
from models import AppState
from services.model_tools import (
    load_model,
    load_uploaded_model,
    max_upload_bytes,
    normalize_mesh_units,
    model_within_build_volume,
    source_name_from_filename,
    transformed_model,
    validate_mesh,
    validate_upload,
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


def test_uploaded_model_stays_in_its_workspace(monkeypatch, tmp_path) -> None:
    loaded_paths = []
    mesh = trimesh.creation.box()

    def fake_load_model(path):
        loaded_paths.append(path)
        return mesh

    monkeypatch.setattr(model_tools_module, "load_model", fake_load_model)
    first_uploads = tmp_path / "first" / "uploads"
    second_uploads = tmp_path / "second" / "uploads"

    first_mesh, first_name = load_uploaded_model(
        "../shared.stl",
        b"first",
        first_uploads,
    )
    second_mesh, second_name = load_uploaded_model(
        "/shared.stl",
        b"second",
        second_uploads,
    )

    assert first_mesh is mesh
    assert second_mesh is mesh
    assert first_name == second_name == "shared"
    assert loaded_paths == [
        first_uploads / "shared.stl",
        second_uploads / "shared.stl",
    ]
    assert loaded_paths[0].read_bytes() == b"first"
    assert loaded_paths[1].read_bytes() == b"second"


def test_upload_validation_rejects_unsupported_empty_and_oversized_files(
    monkeypatch,
) -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_upload("model.exe", b"content")

    with pytest.raises(ValueError, match="Uploaded file is empty"):
        validate_upload("model.stl", b"")

    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    assert max_upload_bytes() == 1024 * 1024
    with pytest.raises(ValueError, match="exceeds the 1 MB limit"):
        validate_upload("model.stl", b"x" * (1024 * 1024 + 1))


def test_mesh_validation_rejects_empty_and_complex_meshes(monkeypatch) -> None:
    with pytest.raises(ValueError, match="Uploaded mesh is empty"):
        validate_mesh(trimesh.Trimesh())

    monkeypatch.setattr(model_tools_module, "MAX_MESH_FACES", 1)
    with pytest.raises(ValueError, match="face limit"):
        validate_mesh(trimesh.creation.box())


def test_failed_model_load_removes_temporary_upload(monkeypatch, tmp_path) -> None:
    def fail_load(path):
        raise ValueError("invalid mesh")

    monkeypatch.setattr(model_tools_module, "load_model", fail_load)
    upload_dir = tmp_path / "uploads"

    with pytest.raises(ValueError, match="invalid mesh"):
        load_uploaded_model("model.stl", b"invalid", upload_dir)

    assert list(upload_dir.iterdir()) == []


def test_source_name_is_safe_for_generated_output_paths() -> None:
    assert source_name_from_filename("../../bad model.stl") == "bad model"
    assert source_name_from_filename(r"..\bad model.stl") == "bad model"
    assert source_name_from_filename("...") == "model"


def test_transformed_model_does_not_mutate_stored_mesh() -> None:
    base = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    base.apply_translation([45.0, 45.0, 15.0])
    original_vertices = base.vertices.copy()
    state = AppState(
        current_model=(base, "box"),
        model_xy_position=[20.0, 30.0],
        model_z_degrees=90.0,
    )

    result = transformed_model(state)

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
