import io
import json
from pathlib import Path
import zipfile

import numpy as np
import trimesh

from models import AppState, GuideSurfaceSnapshot, PlaneSnapshot
from services.project_io import load_scene, save_scene


def test_scene_round_trip_preserves_nonplanar_project() -> None:
    state = AppState(
        current_model=(trimesh.creation.box(), "box"),
        model_xy_position=[12.0, 34.0],
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
                bend_x=0.01,
                bend_y=-0.02,
            )
        ],
        slicing_mode="nonplanar",
        debug_mode=True,
    )

    content = save_scene(state)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    assert manifest["version"] == 2
    assert "plane_id" not in manifest["plane_snapshots"][0]
    assert "guide_id" not in manifest["guide_surfaces"][0]

    loaded = load_scene(content)
    assert loaded.current_model is not None
    assert loaded.current_model[1] == "box"
    assert loaded.model_xy_position == [12.0, 34.0]
    assert loaded.model_z_degrees == 15.0
    assert loaded.debug_mode
    assert loaded.slicing_mode == "nonplanar"
    assert loaded.plane_snapshots[0].plane_id == 0
    assert loaded.guide_surfaces[0].guide_id == 0
    assert loaded.guide_surfaces[0].bend_x == 0.01
    assert loaded.guide_surfaces[0].bend_y == -0.02


def test_sample_scenes_load() -> None:
    loaded = load_scene(Path("samples/Tube.pentos").read_bytes())

    assert loaded.current_model is not None
    assert loaded.current_model[1] == "Tube"
    assert len(loaded.plane_snapshots) == 1
    assert loaded.guide_surfaces == []
    assert loaded.slicing_mode == "multiplanar"

    loaded = load_scene(Path("samples/arched_bridge.pentos").read_bytes())

    assert loaded.current_model is not None
    assert loaded.current_model[1] == "arched_bridge"
    assert len(loaded.guide_surfaces) == 3
    assert loaded.slicing_mode == "nonplanar"
