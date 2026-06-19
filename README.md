# Pentos Slicer

Pentos Slicer is a small Python 3.13 web UI for preparing a model, placing
interactive slice planes, exporting oriented STL chunks, running PrusaSlicer,
and merging the generated G-code with Pentos A/B transition moves.

The app uses [Viser](https://viser.studio/) for the browser-based 3D interface
and `trimesh` for model loading and geometry operations.

## Requirements

- Python 3.13
- `uv`
- `prusa-slicer` available on `PATH` when running the full slicing pipeline

Install the Python environment:

```bash
uv sync
```

## Run

Start the local Viser app:

```bash
uv run python main.py
```

Open the URL printed in the terminal, usually:

```text
http://localhost:8080
```

## Basic Workflow

1. Upload a model (`.stl`, `.3mf`, `.obj`, or `.ply`).
2. Add one or more slice planes.
3. Move or rotate planes with the viewport gizmo or GUI controls.
4. Click **Slice**.
5. The app switches to a preview shell showing the generated G-code path.
6. Click **Back to Setup** to return to the model and plane controls.

Sample models are available in `models/`.

## Project Layout

- `main.py` starts the Viser server and mounts the app.
- `app.py` wires app state, services, and views together.
- `app_state.py` stores shared UI state such as the loaded model, plane
  snapshots, and generated G-code path.
- `views/setup_view.py` owns upload, model display, plane controls, and the
  Slice button.
- `views/preview_view.py` owns the post-slice preview shell.
- `plane_manager.py` manages interactive slice planes.
- `slice_tools.py` exports oriented STL chunks and invokes PrusaSlicer.
- `gcode_tools.py` trims and merges generated G-code with Pentos transitions.
- `machine.py` stores machine geometry constants.
- `theming.py` configures the UI theme and shared build plate/grid scene.

Generated runtime files are written to `uploaded_models/`, `temp/`, and
`output/`. These are local outputs and should not be committed.

## App Structure

`PentosApp` is the composition root. It creates shared state and injects only
the dependencies each view needs:

- `SetupView` receives app state, the slicer service, and a `show_preview`
  callback.
- `PreviewView` receives app state and a `show_setup` callback.

This keeps the views from importing `PentosApp` directly, which avoids circular
imports as the app is split across files.

## Development Checks

Format touched Python files:

```bash
uv run black .
```

Run a quick syntax check:

```bash
uv run python -m compileall .
```

For visible UI changes, run the app, load a sample model, add a plane, and
exercise the setup-to-preview flow.
