# Repository Guidelines

## Project Structure & Module Organization

This is a small Python 3.13 slicer application with a flat module layout. `main.py` starts the Viser UI, handles model upload, and coordinates slicing. `plane_manager.py` manages interactive slice planes, `slice_tools.py` exports oriented STL chunks and invokes PrusaSlicer, and `gcode_tools.py` merges generated G-code with Pentos A/B transitions. Shared machine constants live in `machine.py`; colors and UI theme setup live in `theming.py`.

Static and sample inputs are kept in `assets/` and `models/`. `pentos_config.ini` is the PrusaSlicer profile used by the slicing pipeline. Generated or local runtime data belongs in `uploaded_models/`, `temp/`, and `output/`; these paths are ignored and should not be committed.

## Build, Test, and Development Commands

- `uv sync` installs the Python environment from `pyproject.toml` and `uv.lock`.
- `uv run python main.py` starts the local Viser app and prints the browser URL.
- `uv run black .` formats the Python modules using the configured dev dependency.
- `uv run python -m compileall .` performs a quick syntax check across the repository.

Slicing requires the external `prusa-slicer` executable on `PATH`; `slice_tools.py` invokes it directly with `pentos_config.ini`.

## Machine Mechanics Context

The Pentos machine has X/Y/Z Cartesian toolhead motion plus A/B bed rotation.

Coordinate frame:

- `X`, `Y`, and `Z` are literal Cartesian machine axes.
- "From the front" means standing at `Y=0` and looking toward increasing `Y`
  (`Y=235` on the current machine).
- "From the top" means looking down along `-Z`.
- The slicer-local build plate is `90mm x 90mm`.
- The slicer-local build plate center is `[45, 45, 0]`.
- The real machine build plate center is currently `[113, 52, 0]`.
- `MACHINE_OFFSET` maps slicer-local plate coordinates to the real machine
  plate center.

A/B rotation conventions:

- `A = 0` means the bed/chunk is flat.
- Positive `A` tilts clockwise when looking from the front.
- With positive `A`, the `-X` side rises and the `+X` side lowers.
- `B` is the circular bed/spindle rotation.
- Positive `B` spins clockwise when looking from the top.
- With positive `B`, a mark on the `+X` side moves toward `Y=0`.

The current code models the A/B pose with `rotation_matrix(a, b)` in
`machine.py`. The tested composition is path-independent: commanding `A` first,
`B` first, or both in the same move reaches the same final physical pose.

`ROTATION_CENTER` is the slicer-local build plate center with a provisional
`Z = 1.5mm` pivot height. That hard-coded Z value is subject to change depending
on bed height, fixture stackup, and measured distance between the modeled bed
surface and the real A/B rotation axis.

The firmware does not do A/B coordinate compensation. When A or B changes, the
firmware does not automatically transform future X/Y/Z moves into the rotated
bed frame. Any required compensation must be handled by the slicer/merge
pipeline before the G-code reaches the printer.

Slice plane normals are treated as `print_up_normal`: the direction the chunk
should print from bottom to top in the final object frame. For example, a chunk
with `print_up_normal = [-1, 0, 0]` is expected to print with `A90 B0` on the
current machine.

When a chunk is prepared for PrusaSlicer, the chunk may be rotated/flattened and
then shifted onto the slicer's local build plate. That shift is stored as
`flat_xy_offset`. It is a temporary PrusaSlicer centering move, not a physical
machine target by itself. During merge, non-base chunk G-code is translated into
machine coordinates, shifted up by `z_offset`, and adjusted by
`-flat_xy_offset` in X/Y so the temporary centering does not move the physical
continuation point.

Preview rendering is useful for sanity checks, but the real machine behavior is
the source of truth for A/B sign, pivot, and offset verification.

## Coding Style & Naming Conventions

Use Black formatting and 4-space indentation. Prefer type annotations for public functions and data containers; current modules use `dataclass`, `Protocol`, and explicit `Path`/`numpy` types where useful. Keep module names lowercase with underscores, function and variable names in `snake_case`, and constants in `UPPER_SNAKE_CASE`. Keep comments short and reserved for non-obvious geometry, machine, or G-code behavior.

## Testing Guidelines

This repository does not use an automated test suite. Validate changes manually with `uv run python main.py`, load a sample or uploaded model, exercise plane controls, run slicing when `prusa-slicer` is available, and inspect the generated files in `output/`. Use `uv run python -m compileall .` as a quick syntax check before handing off changes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Clean up logic and fix rotation bug` and `Add build plate to viewport and represent in mm`. Follow that style: one clear sentence, present tense, and focused scope.

Pull requests should describe the behavior change, note any manual testing performed, and call out dependencies such as PrusaSlicer or sample model files. Include screenshots or screen recordings for visible UI changes in the Viser scene.
