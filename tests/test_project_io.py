import io
import json
from pathlib import Path
import zipfile
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest
import trimesh

from models import AppState, GuideSurfaceSnapshot, PlaneSnapshot
from services.project_io import load_scene, save_scene
from models import PrintSettings, SlicingSettings
from models.slicing_settings import filament_preset


def test_scene_preserves_custom_slicing_settings() -> None:
    state = AppState(
        current_model=(trimesh.creation.box(), "box"),
        slicing_settings=SlicingSettings(
            filament=replace(filament_preset("PETG"), temperature=245),
            print=PrintSettings(fill_density=30),
        ),
    )
    assert load_scene(save_scene(state)).slicing_settings == state.slicing_settings


def test_scene_round_trip_preserves_nonplanar_project() -> None:
    state = AppState(
        current_model=(trimesh.creation.box(), "box"),
        model_xy_position=(12.0, 34.0),
        model_z_degrees=15.0,
        plane_snapshots=[
            PlaneSnapshot(
                position=np.array([1.0, 2.0, 3.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                plane_id=42,
            )
        ],
        guide_surfaces=[
            GuideSurfaceSnapshot(
                position=np.array([4.0, 5.0, 6.0]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                guide_id=9,
                size_mm=np.array([80.0, 90.0]),
                heights_mm=np.arange(16, dtype=float).reshape(4, 4),
            )
        ],
        slicing_mode="nonplanar",
        debug_mode=True,
    )

    content = save_scene(state)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    assert manifest["version"] == 3
    assert "plane_id" not in manifest["plane_snapshots"][0]
    assert "guide_id" not in manifest["guide_surfaces"][0]

    loaded = load_scene(content)
    assert loaded.current_model is not None
    assert loaded.current_model[1] == "box"
    assert loaded.model_xy_position == (12.0, 34.0)
    assert loaded.model_z_degrees == 15.0
    assert loaded.debug_mode
    assert loaded.slicing_mode == "nonplanar"
    assert loaded.plane_snapshots[0].plane_id == 0
    assert loaded.guide_surfaces[0].guide_id == 0
    assert_allclose(loaded.guide_surfaces[0].size_mm, [80.0, 90.0])
    assert_allclose(loaded.guide_surfaces[0].heights_mm, np.arange(16).reshape(4, 4))


@pytest.mark.parametrize("version", [1, 2])
def test_older_project_versions_are_rejected(version: int) -> None:
    content = save_scene(AppState(current_model=(trimesh.creation.box(), "box")))
    source = zipfile.ZipFile(io.BytesIO(content))
    files = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(files["manifest.json"])
    manifest["version"] = version
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(
                name,
                json.dumps(manifest) if name == "manifest.json" else data,
            )

    with pytest.raises(ValueError, match="version 3 is required"):
        load_scene(output.getvalue())


def test_malformed_guide_grid_is_rejected() -> None:
    state = AppState(current_model=(trimesh.creation.box(), "box"))
    state.guide_surfaces = [
        GuideSurfaceSnapshot(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0)
    ]
    content = save_scene(state)
    with zipfile.ZipFile(io.BytesIO(content)) as source:
        files = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(files["manifest.json"])
    manifest["guide_surfaces"][0]["heights_mm"] = [[0.0]]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(
                name,
                json.dumps(manifest) if name == "manifest.json" else data,
            )

    with pytest.raises(ValueError, match="4x4"):
        load_scene(output.getvalue())


def test_every_sample_scene_loads() -> None:
    for path in Path("samples").glob("*.pentos"):
        loaded = load_scene(path.read_bytes())
        assert loaded.current_model is not None, path
        assert loaded.slicing_settings == SlicingSettings(), path
