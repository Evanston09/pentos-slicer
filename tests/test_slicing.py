from types import SimpleNamespace

import trimesh
import pytest

import services.slicing as slicing_module
from models import DEFAULT_MACHINE_CONFIG
from services.slicing import Slicer


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
