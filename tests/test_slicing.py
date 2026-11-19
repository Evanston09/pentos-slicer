from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import trimesh
import pytest

import services.slicing as slicing_module
from machine import rotation_matrix
from models import DEFAULT_MACHINE_CONFIG
from services.slicing import MeshPiece, Slicer


@pytest.mark.parametrize(
    ("target", "previous", "settings", "expected"),
    [
        ((45.0, -180.0), (0.0, 0.0), {}, (-45.0, 0.0)),
        ((45.0, 100.0), (45.0, 80.0), {}, (45.0, 100.0)),
        ((45.0, 100.0), (45.0, 0.0), {}, (45.0, 100.0)),
        (
            (45.0, 100.0),
            (45.0, 0.0),
            {"a_max_velocity_deg_s": 100.0, "b_max_velocity_deg_s": 10.0},
            (-45.0, -80.0),
        ),
        (
            (45.0, -170.0),
            (45.0, 170.0),
            {"b_degrees_min": 0.0, "b_degrees_max": 360.0},
            (45.0, 190.0),
        ),
        (
            (45.0, 100.0),
            (45.0, 80.0),
            {"b_degrees_min": -90.0, "b_degrees_max": 90.0},
            (-45.0, -80.0),
        ),
        ((0.0, 0.0), (45.0, 80.0), {}, (0.0, 80.0)),
        ((180.0, 0.0), (-170.0, 80.0), {}, (-180.0, 80.0)),
    ],
)
def test_pose_selection_uses_previous_pose_speeds_and_b_limits(
    tmp_path, target, previous, settings, expected
) -> None:
    config = replace(DEFAULT_MACHINE_CONFIG, **settings)
    slicer = Slicer(tmp_path / "output", tmp_path / "temp", config)
    normal = rotation_matrix(*target).T @ [0.0, 0.0, 1.0]

    pose = slicer._select_ab_angles(normal, previous)

    assert pose == pytest.approx(expected)
    assert config.b_degrees_min <= pose[1] <= config.b_degrees_max
    np.testing.assert_allclose(
        rotation_matrix(*pose) @ normal, [0.0, 0.0, 1.0], atol=1e-12
    )


def test_pose_selection_rejects_unreachable_b(tmp_path) -> None:
    config = replace(DEFAULT_MACHINE_CONFIG, b_degrees_min=-10.0, b_degrees_max=10.0)
    slicer = Slicer(tmp_path / "output", tmp_path / "temp", config)
    normal = rotation_matrix(45.0, 80.0).T @ [0.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="B limits"):
        slicer._select_ab_angles(normal, (0.0, 0.0))


def test_chunk_export_preserves_pose_continuity_and_compensation(
    monkeypatch, tmp_path
) -> None:
    mesh = trimesh.creation.box(extents=[10.0, 12.0, 14.0])
    mesh.apply_translation([45.0, 45.0, 7.0])
    normals = [
        None,
        rotation_matrix(45.0, 80.0).T @ [0.0, 0.0, 1.0],
        rotation_matrix(45.0, 100.0).T @ [0.0, 0.0, 1.0],
    ]
    monkeypatch.setattr(
        slicing_module,
        "decompose_mesh",
        lambda *args, **kwargs: [MeshPiece(mesh.copy(), normal) for normal in normals],
    )
    slicer = Slicer(tmp_path / "output", tmp_path / "temp", DEFAULT_MACHINE_CONFIG)

    chunks = slicer.export_stl_chunks(mesh, [], "test")

    assert [(c.a_degrees, c.b_degrees) for c in chunks] == pytest.approx(
        [(0.0, 0.0), (45.0, 80.0), (45.0, 100.0)]
    )
    center = np.asarray(DEFAULT_MACHINE_CONFIG.rotation_center_local_mm)
    for chunk in chunks:
        oriented = trimesh.load_mesh(chunk.path)
        oriented.apply_translation(
            [-chunk.flat_xy_offset[0], -chunk.flat_xy_offset[1], chunk.z_offset]
        )
        restored = center + (oriented.vertices - center) @ rotation_matrix(
            chunk.a_degrees, chunk.b_degrees
        )
        # Every exported vertex returns to its source after undoing compensation.
        distances = np.linalg.norm(
            restored[:, None, :] - mesh.vertices[None, :, :], axis=2
        )
        assert np.max(np.min(distances, axis=1)) < 1e-4


def test_prusa_slicer_reports_chunk_progress(monkeypatch, tmp_path) -> None:
    chunks = [
        SimpleNamespace(path=tmp_path / "first.stl"),
        SimpleNamespace(path=tmp_path / "second.stl"),
    ]
    events = []
    commands = []
    monkeypatch.setattr(
        slicing_module.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    slicer = Slicer(tmp_path / "output", tmp_path / "temp", DEFAULT_MACHINE_CONFIG)

    paths = slicer.run_prusa_slicer(
        chunks,
        config_path=tmp_path / "custom.ini",
        progress=lambda value, message: events.append((value, message)),
    )

    assert paths == [tmp_path / "first.gcode", tmp_path / "second.gcode"]
    assert events == [
        (0.15, "Slicing chunk 1 of 2..."),
        (0.5, "Slicing chunk 2 of 2..."),
    ]
    assert all(
        command[command.index("--load") + 1] == str(tmp_path / "custom.ini")
        for command in commands
    )
    assert commands[0][commands[0].index("--top-solid-layers") + 1] == "0"
    assert commands[1][commands[1].index("--bottom-solid-layers") + 1] == "0"
    assert commands[1][commands[1].index("--brim-type") + 1] == "no_brim"


@pytest.mark.parametrize("method", ["slice", "debug_transition_check"])
def test_slice_entrypoints_forward_custom_config(method, monkeypatch, tmp_path) -> None:
    slicer = Slicer(tmp_path / "output", tmp_path / "temp", DEFAULT_MACHINE_CONFIG)
    raw_path = tmp_path / "temp" / "chunk.gcode"
    raw_path.write_text("G90\nG1 X1 Y2 Z3\n")
    configs = []
    monkeypatch.setattr(slicer, "export_stl_chunks", lambda *args: [SimpleNamespace()])

    def run(chunks, *, progress, config_path):
        configs.append(config_path)
        return [raw_path]

    monkeypatch.setattr(slicer, "run_prusa_slicer", run)
    monkeypatch.setattr(
        slicing_module,
        "generate_debug_transition_check",
        lambda paths, chunks, output, offset: output,
    )
    getattr(slicer, method)(
        trimesh.creation.box(),
        [],
        progress=lambda *args: None,
        config_path=tmp_path / "custom.ini",
    )
    assert configs == [tmp_path / "custom.ini"]


def test_single_slice_skips_merge_and_applies_machine_offset(
    monkeypatch,
    tmp_path,
) -> None:
    raw_path = tmp_path / "temp" / "chunk.gcode"
    raw_path.parent.mkdir()
    raw_path.write_text("G90\nG1 X1 Y2 Z3\n")
    slicer = Slicer(tmp_path / "output", tmp_path / "temp", DEFAULT_MACHINE_CONFIG)
    chunk = SimpleNamespace()
    monkeypatch.setattr(slicer, "export_stl_chunks", lambda *args: [chunk])
    monkeypatch.setattr(
        slicer, "run_prusa_slicer", lambda chunks, progress, config_path: [raw_path]
    )
    monkeypatch.setattr(slicing_module, "merge_gcode_files", lambda *args: None)

    output = slicer.slice(
        trimesh.creation.box(),
        [],
        "model",
        progress=lambda _value, _message: None,
    )

    assert "G1 X69.0 Y9.0 Z3.0" in output.read_text()
