# Pentos Visualization Pipeline

Deterministic, resumable pipeline that produces six short caption-only
technical videos about the Pentos printer and its two slicing workflows, in
1920x1080 and 1080x1920. All visuals come from the real pipeline: `.pentos`
samples, production slicing services, generated G-code, and the URDF machine.
No narration, music, or hosted AI services.

## Prerequisites

- The project environment (`uv sync`) with Python 3.13 (Pillow is included as a
  dev dependency for caption rendering)
- `blender` (5.x, EEVEE), `ffmpeg`, `ffprobe`, and `prusa-slicer`
  on `PATH`; the FFmpeg build only needs the overlay filter (no libass)
- `samples/Tube.pentos`, `samples/flat_base_curved_top.pentos`,
  `pentos_config.ini`, and `assets/Pentos_URDF/` in the repository

## Usage

```sh
# Show the plan, cache keys, frame counts, and commands without running anything
uv run python -m visualizations.render_all --dry-run

# Prepare/refresh cached scene data only
uv run python -m visualizations.render_all --prepare-only

# Full batch: prepare, render, encode, report
uv run python -m visualizations.render_all

# Subset by video or orientation
uv run python -m visualizations.render_all --videos tube_divide tube_continue
uv run python -m visualizations.render_all --orientations horizontal
```

The batch is interruptible: completed frames are detected and skipped on the
next run, frames are written atomically, and a failure in one video is
isolated and reported. A run summary is written to
`visualizations/generated/reports/report.json` and `report.md`.

Extract a poster frame after reviewing a render:

```sh
uv run python -m visualizations.render_all --poster tube_divide horizontal 200
```

## Output layout

Everything generated lands under ignored `visualizations/generated/`:

```
generated/
  cache/                 content-addressed prepared scene data
    machine_<hash>/      parsed URDF hierarchy + copied STL meshes
    tube_<hash>/         chunk STLs, merged-G-code toolpath samples, transitions
    nonplanar_<hash>/    volume field, deformed mesh, planar + mapped paths
  frames/<video>/<orientation>/frame_0001.png ...
  videos/<video>_<orientation>.mp4 and caption_*.png overlays
  logs/                  one log per render/encode subprocess
  reports/               report.json + report.md
```

Cache keys hash the source samples, `pentos_config.ini`, the machine/slicing
services, the URDF, and the pipeline version, so editing a sample or profile
only invalidates the affected videos' prepared data.

## Video catalog

| Video | Scene | Content |
| --- | --- | --- |
| `meet_pentos` | URDF machine | Orbit, toolhead, tilting/spinning bed, five axes |
| `five_axes` | URDF machine | One axis at a time (X, Y, Z, A, B), then combined |
| `tube_divide` | `Tube.pentos` | Cut plane, chunk separation, flattening with real A/B |
| `tube_continue` | `Tube.pentos` | Merged G-code: chunk 1, safe lift, A/B swing, continuation |
| `nonplanar_guides` | `flat_base_curved_top.pentos` | Guides, scalar field, flatten morph |
| `nonplanar_map` | `flat_base_curved_top.pentos` | Planar slice paths, inverse mapping, A/B tilt |

A/B angles shown in videos come from `machine.rotation_matrix` conventions
(verified in `tests/test_machine.py`). The URDF joint axes are normalized to
machine conventions in the prepared `machine.json` (A and B joints carry sign
-1 with the derivation recorded alongside).

## Development

- `visualizations/storyboards.py` — declarative shots, captions, cameras,
  actions, and validation
- `visualizations/prepare_assets.py` — runs the production workflows and
  exports Blender-friendly JSON/NPZ/STL data
- `visualizations/blender/render.py` (plus `geometry.py`) — Blender-only
  scene construction and per-frame rendering
- `visualizations/encode.py` — ASS captions, FFmpeg encoding, ffprobe checks
- `tests/test_visualization_pipeline.py` — schema/orchestration tests plus
  low-resolution Blender smoke renders (skipped when `blender` is missing)
