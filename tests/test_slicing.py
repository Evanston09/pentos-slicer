from types import SimpleNamespace

import trimesh

import services.slicing as slicing_module
from models import DEFAULT_MACHINE_CONFIG
from services.slicing import Slicer


def test_prusa_slicer_reports_chunk_progress(monkeypatch, tmp_path) -> None:
    chunks = [
        SimpleNamespace(path=tmp_path / "first.stl"),
        SimpleNamespace(path=tmp_path / "second.stl"),
    ]
    events = []
    monkeypatch.setattr(slicing_module.subprocess, "run", lambda *args, **kwargs: None)
    slicer = Slicer(tmp_path / "output", tmp_path / "temp", DEFAULT_MACHINE_CONFIG)

    paths = slicer.run_prusa_slicer(
        chunks,
        progress=lambda value, message: events.append((value, message)),
    )

    assert paths == [tmp_path / "first.gcode", tmp_path / "second.gcode"]
    assert events == [
        (0.15, "Slicing chunk 1 of 2..."),
        (0.5, "Slicing chunk 2 of 2..."),
    ]


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
    monkeypatch.setattr(slicer, "run_prusa_slicer", lambda chunks, progress: [raw_path])
    monkeypatch.setattr(slicing_module, "merge_gcode_files", lambda *args: None)

    output = slicer.slice(
        trimesh.creation.box(),
        [],
        "model",
        progress=lambda _value, _message: None,
    )

    assert "G1 X69.0 Y9.0 Z3.0" in output.read_text()
