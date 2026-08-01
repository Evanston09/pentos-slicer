import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import trimesh

from controllers.setup_controller import SetupController
from models import AppState, PlaneSnapshot
from services.project_io import load_scene, save_scene


class FakeSetupView:
    def mount(self, state: AppState) -> None:
        pass

    def unmount(self) -> None:
        pass

    def set_status(self, message: str) -> None:
        pass

    def show_mesh(self, mesh, center, position, wxyz) -> None:
        pass

    def clear_model_scene(self) -> None:
        pass

    def set_model_controls_enabled(self, enabled: bool) -> None:
        pass

    def update_model_placement(self, xy_position, z_degrees, position, wxyz) -> None:
        pass

    def replace_planes(self, planes) -> None:
        pass

    def add_plane(self, plane) -> None:
        pass

    def remove_plane(self, plane_id: int) -> None:
        pass

    def set_debug_mode_value(self, enabled: bool) -> None:
        pass


class FakeSlicer:
    pass


def test_scene_round_trip_preserves_version_one_and_omits_plane_ids() -> None:
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
        debug_mode=True,
    )

    content = save_scene(state)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))

    assert manifest["version"] == 1
    assert "plane_id" not in manifest["plane_snapshots"][0]

    loaded = load_scene(content)
    assert loaded.current_model is not None
    assert loaded.current_model[1] == "box"
    assert loaded.model_xy_position == [12.0, 34.0]
    assert loaded.model_z_degrees == 15.0
    assert loaded.debug_mode
    assert loaded.plane_snapshots[0].plane_id is None

    SetupController(
        loaded,
        FakeSlicer(),
        FakeSetupView(),
        lambda: None,
        SimpleNamespace(path=Path(".")),
        object(),
    )
    assert loaded.plane_snapshots[0].plane_id == 0


def test_existing_sample_scene_still_loads() -> None:
    content = Path("samples/Tube.pentos").read_bytes()
    loaded = load_scene(content)

    assert loaded.current_model is not None
    assert loaded.current_model[1] == "Tube"
    assert len(loaded.plane_snapshots) == 1
