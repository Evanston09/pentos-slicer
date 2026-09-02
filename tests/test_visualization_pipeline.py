"""Tests for the visualization pipeline: schema, extraction, orchestration,
and low-resolution Blender smoke renders (skipped when blender is missing)."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = REPO_ROOT / "visualizations"
for path in (str(REPO_ROOT), str(VIZ_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import encode  # noqa: E402
import prepare_assets  # noqa: E402
import storyboards  # noqa: E402
from render_all import (  # noqa: E402
    PipelineError,
    build_render_config,
    check_tools,
    existing_frames,
    resolve_cache_dirs,
)

BLENDER = shutil.which("blender")


# ---------------------------------------------------------------------------
# Storyboards
# ---------------------------------------------------------------------------


def test_storyboards_validate() -> None:
    videos = storyboards.videos()
    storyboards.validate_storyboards(videos)
    assert set(videos) == set(storyboards.VIDEO_IDS)
    for video in videos.values():
        assert video.frames == sum(shot.frames for shot in video.shots)
        for shot in video.shots:
            assert shot.scene == video.scene


def test_shot_frame_ranges_are_contiguous() -> None:
    for video in storyboards.videos().values():
        next_expected = 1
        for _shot, first, last in storyboards.shot_frame_ranges(video):
            assert first == next_expected
            assert last >= first
            next_expected = last + 1


def _broken_video(mutate) -> None:
    videos = storyboards.videos()
    mutate(videos)
    with pytest.raises(storyboards.StoryboardError):
        storyboards.validate_storyboards(videos)


def test_validation_rejects_bad_caption() -> None:
    def mutate(videos):
        videos["meet_pentos"].shots[0].caption.strip()
        object.__setattr__(videos["meet_pentos"].shots[0], "caption", "  ")

    _broken_video(mutate)


def test_validation_rejects_duplicate_shot_ids() -> None:
    def mutate(videos):
        video = videos["meet_pentos"]
        from dataclasses import replace

        duplicate = replace(video.shots[0], id=video.shots[1].id)
        object.__setattr__(video, "shots", video.shots + (duplicate,))

    _broken_video(mutate)


def test_validation_rejects_unknown_action_type() -> None:
    def mutate(videos):
        video = videos["tube_divide"]
        from dataclasses import replace

        shot = replace(video.shots[0], action={"type": "teleport"})
        object.__setattr__(video, "shots", (shot,))

    _broken_video(mutate)


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


def test_cache_keys_stable_and_content_sensitive(tmp_path, monkeypatch) -> None:
    source = tmp_path / "sample.pentos"
    source.write_bytes(b"first")
    config = tmp_path / "config.ini"
    config.write_bytes(b"profile")
    monkeypatch.setattr(prepare_assets, "SAMPLES", {"tube": source})
    monkeypatch.setattr(prepare_assets, "CONFIG_INI", config)
    monkeypatch.setattr(prepare_assets, "REPO_ROOT", tmp_path)

    first = prepare_assets.cache_key_tube()
    second = prepare_assets.cache_key_tube()
    assert first == second

    source.write_bytes(b"second")
    assert prepare_assets.cache_key_tube() != first


# ---------------------------------------------------------------------------
# Toolpath extraction
# ---------------------------------------------------------------------------

GCODE_SAMPLE = """; header
M107
G90
M83
G1 E2 F240
G1 X5 Y5 Z0.2 F9000
G1 X8 Y5 Z0.2 E0.1 F1200
;LAYER_CHANGE
;Z:0.3
G1 Z0.3 F1080
G1 X20 Y20 Z0.3 F9000
G1 X30 Y20 Z0.3 E0.4 F1200
G1 X30 Y30 Z0.3 E0.4
G1 X20 Y20 Z0.3 E0.4
;LAYER_CHANGE
;Z:0.6
G1 Z0.6 F1080
G1 X25 Y25 Z0.6 F9000
G1 X28 Y25 Z0.6 E0.2 F1200

; --- PENTOS A/B TRANSITION ---
G91
G1 Z15 F3000
G90
G1 A70 B0 F1200
G1 X99.9615 Y19.0 F1200
G1 Z32.7362 F1200
; --- END PENTOS A/B TRANSITION ---
G1 X99.9615 Y21.0 Z32.7362 E0.3 F1200
G1 X100.6456 Y21.0 Z30.8569 E0.3
"""


def test_extract_toolpath_segments_and_transition() -> None:
    extraction = prepare_assets.extract_toolpath(GCODE_SAMPLE)
    points = extraction.points
    kinds = extraction.kinds
    assert len(points) == len(kinds) == len(extraction.seg_times)

    extrusion_indices = np.flatnonzero(kinds == prepare_assets.EXTRUSION_KIND)
    assert len(extrusion_indices) == 7
    # Skirt extrusion precedes the first layer change.
    assert extraction.layers[extrusion_indices[0]] == -1
    # Moves before the first layer change that lack extrusion are setup.
    assert kinds[1] == prepare_assets.TRAVEL_KIND
    assert kinds[2] == prepare_assets.TRAVEL_KIND

    machine = prepare_assets.DEFAULT_MACHINE_CONFIG
    first_end = np.asarray([8.0, 5.0, 0.2]) - machine.machine_offset
    assert np.allclose(points[0][1], first_end, atol=1e-6)

    transition = extraction.transitions[0]
    assert transition["a_deg"] == pytest.approx(70.0)
    assert transition["b_deg"] == pytest.approx(0.0)
    assert transition["lift_points"][1][2] == pytest.approx(
        transition["lift_points"][0][2] + 15.0
    )
    continuation = np.asarray(transition["continuation"])
    assert np.allclose(continuation, [11.0, 12.0, 1.2], atol=1e-4)

    # The first post-transition extrusion is un-rotated back into object space.
    transition_index = transition["seg_index"]
    post = points[transition_index][1]
    assert np.allclose(post, [11.0, 14.0, 1.2], atol=1e-4)

    assert extraction.has_rotary
    assert extraction.dial_time[-1] >= extraction.seg_times[-1]


def test_decimate_keeps_extrusion_and_remaps_transitions() -> None:
    extraction = prepare_assets.extract_toolpath(GCODE_SAMPLE)
    repeats = 6000
    big_points = np.tile(extraction.points, (repeats, 1, 1))
    time_offset = np.repeat(
        np.arange(repeats) * (float(extraction.seg_times[-1]) + 1.0),
        len(extraction.seg_times),
    )
    big = prepare_assets.ToolpathExtraction(
        points=big_points,
        kinds=np.tile(extraction.kinds, repeats),
        layers=np.tile(extraction.layers, repeats),
        parts=np.tile(extraction.parts, repeats),
        seg_times=np.tile(extraction.seg_times, repeats) + time_offset,
        dial_time=extraction.dial_time,
        dial_a=extraction.dial_a,
        dial_b=extraction.dial_b,
        transitions=[
            dict(
                extraction.transitions[0],
                seg_index=extraction.transitions[0]["seg_index"] + 24000,
            )
        ],
        has_rotary=True,
    )
    decimated = prepare_assets.decimate_extraction(big)
    assert len(decimated.points) <= (
        prepare_assets.MAX_EXTRUSION_SEGMENTS + prepare_assets.MAX_AUXILIARY_SEGMENTS
    )
    extrusion = int((decimated.kinds == prepare_assets.EXTRUSION_KIND).sum())
    assert extrusion > 0
    assert decimated.transitions[0]["seg_index"] < len(decimated.points)
    # Time ordering preserved (float32 accumulation allows tiny regressions).
    assert np.all(np.diff(decimated.seg_times) >= -0.01)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_render_caption_pngs_covers_all_shots(tmp_path) -> None:
    video = storyboards.videos()["tube_divide"]
    overlays = encode.render_caption_pngs(
        video, "horizontal", (1920, 1080), tmp_path, fps=30
    )
    assert len(overlays) == len(video.shots)
    assert overlays[0].start_s == pytest.approx(0.35)
    expected_end = video.shots[0].frames / 30 - 0.35
    assert overlays[0].end_s == pytest.approx(expected_end)
    for index, overlay in enumerate(overlays):
        assert overlay.png.exists()
        assert overlay.png.stat().st_size > 0
        assert overlay.end_s > overlay.start_s


def test_caption_wraps_long_text() -> None:
    font = encode._load_font(52)
    wrapped = encode._wrap_caption(
        "One merged G-code file. PrusaSlicer's temporary centering is removed "
        "so the physical path is exact.",
        font,
        1660,
    )
    assert "\n" in wrapped
    for line in wrapped.split("\n"):
        assert font.getlength(line) <= 1660


def test_build_ffmpeg_command_shape(tmp_path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(1, 5):
        (frames_dir / f"frame_{index:04d}.png").write_bytes(b"x")
    caption = tmp_path / "caption_00.png"
    caption.write_bytes(b"x")
    command = encode.build_ffmpeg_command(
        frames_dir,
        tmp_path / "out.mp4",
        [encode.CaptionOverlay(0.35, 3.0, caption)],
        30,
        (1920, 1080),
    )
    joined = " ".join(command)
    assert "overlay=" in joined
    assert "between(t,0.350,3.000)" in joined
    assert "libx264" in command
    assert "yuv420p" in joined
    assert any("scale=-2:" in part for part in command)
    assert "-framerate" in command and "30" in command


def test_probe_video_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(Exception):
        encode.probe_video(tmp_path / "nope.mp4", (1920, 1080), 30)


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------


def test_build_render_config_frames_and_scene() -> None:
    videos = storyboards.videos()
    video = videos["tube_divide"]
    config = build_render_config(
        video,
        "horizontal",
        Path("/cache/tube_x"),
        Path("/out/frames"),
        frames=[1, 2, 300],
        scale=0.5,
    )
    assert config["scene"] == "tube"
    assert config["resolution"] == (960, 540)
    assert config["frames"] == [1, 2, 300]
    assert len(config["shots"]) == len(video.shots)
    first_last = config["shots"][-1]["last_frame"]
    assert first_last == video.frames
    assert (
        config["shots"][0]["camera"]["orbit"]["center"]
        == video.shots[0].camera["horizontal"]["orbit"]["center"]
    )


def test_existing_frames_ignores_partial_files(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_0001.png").write_bytes(b"valid")
    (frames / "frame_0002.png").write_bytes(b"")
    (frames / "frame_XXXX.png").write_bytes(b"junk")
    (frames / "other.txt").write_bytes(b"nope")
    assert existing_frames(frames) == {1}


def test_resolve_cache_dirs_requires_prepared_data(tmp_path) -> None:
    with pytest.raises(PipelineError):
        resolve_cache_dirs(tmp_path)
    (tmp_path / "cache" / "machine_abc12345").mkdir(parents=True)
    (tmp_path / "cache" / "tube_def67890").mkdir(parents=True)
    (tmp_path / "cache" / "nonplanar_1234abcd").mkdir(parents=True)
    resolved = resolve_cache_dirs(tmp_path)
    assert resolved["machine"].name == "machine_abc12345"


def test_check_tools_found() -> None:
    check_tools()  # raises if anything is missing on this machine


# ---------------------------------------------------------------------------
# Blender smoke renders (low resolution, tiny synthetic caches)
# ---------------------------------------------------------------------------


def _write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    mesh = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(len(triangles) * 3).reshape(-1, 3),
        process=False,
    )
    mesh.export(path, file_type="stl")


def _box(
    center: tuple[float, float, float], size: tuple[float, float, float]
) -> np.ndarray:
    cx, cy, cz = center
    sx, sy, sz = size
    lower = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2])
    upper = np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
    v = np.array(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ]
    )
    quads = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    ]
    triangles = []
    for quad in quads:
        triangles.append([v[quad[0]], v[quad[1]], v[quad[2]]])
        triangles.append([v[quad[0]], v[quad[2]], v[quad[3]]])
    return np.asarray(triangles)


def _toolpath_npz(path: Path, with_rotary: bool, with_layers: bool = True) -> None:
    lines = np.array(
        [
            [[10, 10, 1], [30, 10, 1]],
            [[30, 10, 1], [30, 30, 1]],
            [[30, 30, 1], [10, 30, 1]],
            [[10, 10, 2], [30, 10, 2]],
            [[30, 30, 2], [10, 30, 2]],
        ],
        dtype=np.float32,
    )
    kinds = np.array([0, 2, 2, 1, 2], dtype=np.int8)
    layers = np.array([0, 0, 0, 1, 1], dtype=np.int16)
    seg_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    if with_rotary:
        dial_time = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        dial_a = np.array([0.0, 35.0, 70.0], dtype=np.float32)
        dial_b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        dial_time = np.empty(0, dtype=np.float32)
        dial_a = np.empty(0, dtype=np.float32)
        dial_b = np.empty(0, dtype=np.float32)
    if not with_layers:
        layers = np.zeros_like(layers)
    np.savez_compressed(
        path,
        points=lines,
        kinds=kinds,
        layers=layers,
        parts=np.zeros(5, np.int8),
        seg_times=seg_times,
        dial_time=dial_time,
        dial_a=dial_a,
        dial_b=dial_b,
    )


@pytest.fixture()
def machine_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "machine_smoke"
    (cache / "urdf_meshes").mkdir(parents=True)
    _write_binary_stl(
        cache / "urdf_meshes/base_link.STL", _box((0, 0, 0.5), (20, 20, 1))
    )
    _write_binary_stl(cache / "urdf_meshes/gantry.STL", _box((0, 0, 1.5), (6, 6, 2)))
    _write_binary_stl(cache / "urdf_meshes/bed.STL", _box((0, 0, 2.0), (8, 8, 0.4)))
    payload = {
        "scale_mm_per_urdf_unit": 1000.0,
        "root": "base_link",
        "links": [
            {
                "name": "base_link",
                "rgba": [0.5, 0.5, 0.5, 1.0],
                "mesh": "base_link.STL",
                "bounds_mm": [[-10, -10, 0], [10, 10, 1]],
            },
            {
                "name": "gantry_link",
                "rgba": [0.8, 0.8, 0.8, 1.0],
                "mesh": "gantry.STL",
                "bounds_mm": [[-3, -3, 1], [3, 3, 3]],
            },
            {
                "name": "bed_link",
                "rgba": [0.4, 0.45, 0.5, 1.0],
                "mesh": "bed.STL",
                "bounds_mm": [[-4, -4, 2], [4, 4, 3]],
            },
        ],
        "joints": [
            {
                "name": "z_joint",
                "type": "prismatic",
                "parent": "base_link",
                "child": "gantry_link",
                "origin_xyz_m": [0.0, 0.0, 0.001],
                "origin_rpy_rad": [0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, 1.0],
                "limit": {"lower": 0.0, "upper": 0.19},
            },
            {
                "name": "b_joint",
                "type": "revolute",
                "parent": "gantry_link",
                "child": "bed_link",
                "origin_xyz_m": [0.0, 0.0, 0.001],
                "origin_rpy_rad": [0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, -1.0],
                "limit": {"lower": -3.14, "upper": 3.14},
            },
        ],
        "machine_bounds_mm": [[-10, -10, 0], [10, 10, 20]],
        "machine_axis_mapping": {
            "A": {
                "joint": "a_joint",
                "sign": -1.0,
                "units": "degrees",
                "evidence": "test",
            },
            "B": {
                "joint": "b_joint",
                "sign": -1.0,
                "units": "degrees",
                "evidence": "test",
            },
            "X": {
                "joint": "x_joint",
                "sign": None,
                "units": "millimeters",
                "evidence": "test",
            },
            "Y": {
                "joint": "y_joint",
                "sign": None,
                "units": "millimeters",
                "evidence": "test",
            },
            "Z": {
                "joint": "z_joint",
                "sign": None,
                "units": "millimeters",
                "evidence": "test",
            },
        },
    }
    (cache / "machine.json").write_text(json.dumps(payload, indent=2))
    return cache


@pytest.fixture()
def tube_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "tube_smoke"
    cache.mkdir(parents=True)
    _write_binary_stl(cache / "model.stl", _box((45, 45, 10), (30, 30, 20)))
    _write_binary_stl(cache / "chunk_0.stl", _box((45, 45, 5), (30, 30, 10)))
    _write_binary_stl(cache / "chunk_1.stl", _box((45, 45, 15), (30, 30, 10)))
    _write_binary_stl(cache / "chunk_0_flat.stl", _box((45, 45, 5), (30, 30, 10)))
    _write_binary_stl(cache / "chunk_1_flat.stl", _box((45, 45, 15), (30, 30, 10)))
    _toolpath_npz(cache / "toolpath.npz", with_rotary=True)
    scene = {
        "sample": "smoke",
        "source_name": "smoke",
        "model_stl": "model.stl",
        "cut_planes": [{"position": [45.0, 45.0, 10.0], "wxyz": [1.0, 0.0, 0.0, 0.0]}],
        "chunks": [
            {
                "index": 0,
                "print_up_normal": None,
                "a_degrees": 0.0,
                "b_degrees": 0.0,
                "z_offset": 0.0,
                "flat_xy_offset": [0.0, 0.0],
                "mesh": "chunk_0.stl",
                "flat_mesh": "chunk_0_flat.stl",
            },
            {
                "index": 1,
                "print_up_normal": [0.0, 0.0, 1.0],
                "a_degrees": 70.0,
                "b_degrees": 0.0,
                "z_offset": 5.0,
                "flat_xy_offset": [1.0, 0.0],
                "mesh": "chunk_1.stl",
                "flat_mesh": "chunk_1_flat.stl",
            },
        ],
        "plate": {
            "size_mm": [90.0, 90.0],
            "center": [45.0, 45.0, 0.0],
            "rotation_center_local_mm": [44.0, 44.0, 2.0],
        },
        "toolpath": {
            "npz": "toolpath.npz",
            "transitions": "transitions.json",
            "n_segments": 5,
            "n_transitions": 1,
            "has_rotary": True,
        },
    }
    (cache / "scene.json").write_text(json.dumps(scene, indent=2))
    (cache / "transitions.json").write_text(
        json.dumps(
            [
                {
                    "lift_points": [[10.0, 10.0, 5.0], [10.0, 10.0, 20.0]],
                    "a_deg": 70.0,
                    "b_deg": 0.0,
                    "continuation": [12.0, 12.0, 2.0],
                    "seg_index": 3,
                    "time_s": 2.0,
                }
            ]
        )
    )
    return cache


@pytest.fixture()
def nonplanar_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "nonplanar_smoke"
    cache.mkdir(parents=True)
    original = np.array(
        [
            [30, 30, 0],
            [60, 30, 0],
            [60, 60, 0],
            [30, 60, 0],
            [30, 30, 10],
            [60, 30, 10],
            [60, 60, 10],
            [30, 60, 10],
        ],
        dtype=np.float32,
    )
    deformed = original.copy()
    deformed[4:, 2] = [12, 11, 10, 11]
    tetrahedra = np.array([], dtype=np.int32)
    boundary_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # sides
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    np.savez_compressed(
        cache / "volume.npz",
        original_vertices=original,
        deformed_vertices=deformed,
        tetrahedra=tetrahedra,
        boundary_faces=boundary_faces,
        scalar_values=np.linspace(0.0, 10.0, len(original), dtype=np.float32),
    )
    _write_binary_stl(cache / "guide_0.stl", _box((45, 45, 0.2), (10, 10, 0.4)))
    _write_binary_stl(cache / "guide_1.stl", _box((45, 45, 10.0), (10, 10, 0.4)))
    _toolpath_npz(cache / "planar_paths.npz", with_rotary=False)
    _toolpath_npz(cache / "mapped_paths.npz", with_rotary=True)
    scene = {
        "sample": "smoke",
        "source_name": "smoke",
        "model_stl": "model.stl",
        "deformed_stl": "deformed.stl",
        "guides": [
            {
                "guide_id": 0,
                "position": [45.0, 45.0, 0.0],
                "wxyz": [1, 0, 0, 0],
                "bend_x": 0.0,
                "bend_y": 0.0,
                "mesh": "guide_0.stl",
            },
            {
                "guide_id": 1,
                "position": [45.0, 45.0, 10.0],
                "wxyz": [1, 0, 0, 0],
                "bend_x": 0.0,
                "bend_y": 0.0,
                "mesh": "guide_1.stl",
            },
        ],
        "volume_npz": "volume.npz",
        "planar_toolpath": {
            "npz": "planar_paths.npz",
            "n_segments": 5,
            "space": "deformed_local",
        },
        "mapped_toolpath": {
            "npz": "mapped_paths.npz",
            "n_segments": 5,
            "space": "object_local",
            "gcode": "mapped.gcode",
        },
        "base_layers_planar": 2,
        "n_layers": 2,
        "plate": {
            "size_mm": [90.0, 90.0],
            "center": [45.0, 45.0, 0.0],
            "rotation_center_local_mm": [44.0, 44.0, 2.0],
        },
        "machine_xy_size": 235.0,
    }
    (cache / "scene.json").write_text(json.dumps(scene, indent=2))
    return cache


def _run_blender_smoke(
    tmp_path: Path,
    cache: Path,
    scene: str,
    action: dict,
    camera: dict,
) -> Path:
    out_dir = tmp_path / "frames" / scene
    config = {
        "video_id": f"smoke_{scene}",
        "orientation": "horizontal",
        "scene": scene,
        "resolution": (192, 108),
        "fps": 30,
        "frames": [1, 2, 3],
        "cache_dir": str(cache),
        "out_dir": str(out_dir),
        "shots": [
            {
                "id": "shot",
                "first_frame": 1,
                "last_frame": 3,
                "camera": camera,
                "action": action,
            }
        ],
    }
    config_path = tmp_path / f"config_{scene}.json"
    config_path.write_text(json.dumps(config))
    subprocess = __import__("subprocess")
    result = subprocess.run(
        [
            BLENDER,
            "-b",
            "--factory-startup",
            "-P",
            str(VIZ_DIR / "blender/render.py"),
            "--",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    for frame in (1, 2, 3):
        assert (out_dir / f"frame_{frame:04d}.png").stat().st_size > 0
    return out_dir


@pytest.mark.skipif(BLENDER is None, reason="blender executable not available")
def test_blender_smoke_machine(machine_cache, tmp_path) -> None:
    action = {
        "type": "machine_pose",
        "moves": [
            {
                "t0": 0.1,
                "t1": 0.9,
                "joints": {"z_joint": [0.0, 0.1], "b_joint": [0.0, 45.0]},
            }
        ],
        "labels": [
            {"text": "X", "at": [60.0, 0.0, 30.0], "t0": 0.0, "t1": 1.0, "size": 26.0}
        ],
        "arrows": [
            {"axis": "B", "at": [0.0, 0.0, 30.0], "t0": 0.0, "t1": 1.0, "length": 40.0}
        ],
        "highlight": ["bed_link"],
    }
    camera = {
        "orbit": {
            "center": [0.0, 0.0, 20.0],
            "radius": 90.0,
            "height": 50.0,
            "start_deg": -30.0,
            "end_deg": 10.0,
            "ortho_scale": None,
        }
    }
    _run_blender_smoke(tmp_path, machine_cache, "machine", action, camera)


@pytest.mark.skipif(BLENDER is None, reason="blender executable not available")
def test_blender_smoke_tube(tube_cache, tmp_path) -> None:
    action = {
        "type": "preview_reveal",
        "sample": "tube",
        "window": [0.0, 1.0],
        "dial": True,
        "transition": True,
        "transition_label": False,
    }
    camera = {
        "orbit": {
            "center": [45.0, 20.0, 25.0],
            "radius": 150.0,
            "height": 100.0,
            "start_deg": -25.0,
            "end_deg": -10.0,
            "ortho_scale": None,
        }
    }
    _run_blender_smoke(tmp_path, tube_cache, "tube", action, camera)


@pytest.mark.skipif(BLENDER is None, reason="blender executable not available")
def test_blender_smoke_nonplanar(nonplanar_cache, tmp_path) -> None:
    action = {"type": "mapped_reveal", "phase": "map", "dial": False}
    camera = {
        "orbit": {
            "center": [45.0, 25.0, 30.0],
            "radius": 150.0,
            "height": 100.0,
            "start_deg": -25.0,
            "end_deg": -10.0,
            "ortho_scale": None,
        }
    }
    _run_blender_smoke(tmp_path, nonplanar_cache, "nonplanar", action, camera)
