from types import SimpleNamespace

import trimesh

import services.slicing as slicing_module
from models import DEFAULT_MACHINE_CONFIG
from services.slicing import Slicer


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
    monkeypatch.setattr(slicer, "run_prusa_slicer", lambda chunks: [raw_path])
    monkeypatch.setattr(slicing_module, "merge_gcode_files", lambda *args: None)

    output = slicer.slice(trimesh.creation.box(), [], "model")

    assert "G1 X69.0 Y9.0 Z3.0" in output.read_text()
