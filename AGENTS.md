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

## Coding Style & Naming Conventions

Use Black formatting and 4-space indentation. Prefer type annotations for public functions and data containers; current modules use `dataclass`, `Protocol`, and explicit `Path`/`numpy` types where useful. Keep module names lowercase with underscores, function and variable names in `snake_case`, and constants in `UPPER_SNAKE_CASE`. Keep comments short and reserved for non-obvious geometry, machine, or G-code behavior.

## Testing Guidelines

This repository does not use an automated test suite. Validate changes manually with `uv run python main.py`, load a sample or uploaded model, exercise plane controls, run slicing when `prusa-slicer` is available, and inspect the generated files in `output/`. Use `uv run python -m compileall .` as a quick syntax check before handing off changes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Clean up logic and fix rotation bug` and `Add build plate to viewport and represent in mm`. Follow that style: one clear sentence, present tense, and focused scope.

Pull requests should describe the behavior change, note any manual testing performed, and call out dependencies such as PrusaSlicer or sample model files. Include screenshots or screen recordings for visible UI changes in the Viser scene.
