# Pentos Slicer

Pentos Slicer is a small Python 3.13 web UI for preparing a model, placing
interactive slice planes, exporting oriented STL chunks, running PrusaSlicer,
and merging the generated G-code with Pentos A/B transition moves.

The app uses [Viser](https://viser.studio/) for the browser-based 3D interface
and `trimesh` for model loading and geometry operations.

## Demo

<table>
  <tr>
    <td width="50%">
      <img src="docs/images/pentos-slicer-setup.jpg" alt="Pentos Slicer setup view with a model, slice plane, and transform gizmos">
    </td>
    <td width="50%">
      <img src="docs/images/pentos-slicer-preview.jpg" alt="Pentos Slicer preview showing two oriented parts and generated G-code">
    </td>
  </tr>
  <tr>
    <td align="center"><em>Place and orient interactive slice planes</em></td>
    <td align="center"><em>Preview the generated multi-part toolpath</em></td>
  </tr>
</table>

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

For development, restart it automatically whenever a Python file changes:

```bash
./dev.sh
```

Open the URL printed in the terminal, usually:

```text
http://localhost:8080
```

Or build and run it with Docker:

```bash
docker build -t pentos-slicer .
docker run --rm -p 8080:8080 pentos-slicer
```

At most two models are sliced concurrently by default. Set
`MAX_CONCURRENT_SLICES` to a positive integer to change that limit.
Uploads are limited to 50 MB by default. Set `MAX_UPLOAD_SIZE_MB` to a positive
integer to change that limit.

## Basic Workflow

1. Upload a model (`.stl`, `.3mf`, `.obj`, or `.ply`).
2. Add one or more slice planes.
3. Move or rotate planes with the viewport gizmo or GUI controls.
4. Choose filament and print settings, then click **Slice**.
5. The app switches to a preview shell showing the generated G-code path.
6. Click **Back to Setup** to return to the model and plane controls.

Sample models are available in `samples/`.

## Filament and Slicing Settings

The **Filament** panel offers PLA and PETG starting presets with editable nozzle
and bed temperatures, including first-layer temperatures. Cooling, extrusion
multiplier, and maximum volumetric flow use preset values.
PLA preserves the existing defaults. PETG uses the Generic PETG values from
PrusaSlicer's Creality profile (including its inherited cooling settings); adjust
these starting values for your filament and machine. **Reset to Filament Preset**
restores the selected material's values.

**Print Settings** controls layer height, first-layer height, perimeter count, and
infill percentage. The baseline 0.4 mm nozzle supports heights from 0.06 to 0.32 mm.
At 100% infill, Pentos uses rectilinear instead of the baseline grid pattern.
Speeds retain their baseline values. **Reset Print Settings** restores print
defaults without changing the filament selection.

Only the material, four temperatures, and four print settings are remembered in
browser storage and included in exported `.pentos` projects. Material-specific
constants are applied during slicing; fixed settings remain in `pentos_config.ini`.
Loading a project restores its values. Settings edits require slicing
again to update the preview. Controls are disabled while slicing.

Each slice writes its own effective INI in the client's temporary session
workspace. The repository's `pentos_config.ini` remains the baseline. Pentos
preserves its firmware macros, relative extrusion, and chunk-specific overrides.
The temporary INI is cleaned up with the session; browser storage and project
files retain the settings. Arbitrary PrusaSlicer profile import is not supported.

## Machine Configuration

The **Machine** panel imports and exports versioned `.json` machine profiles.
The active profile is saved in the browser and restored on future visits. Profiles
contain machine geometry; the PrusaSlicer profile remains in `pentos_config.ini`.

```json
{
  "format": "pentos-machine",
  "version": 1,
  "name": "My Pentos",
  "build_volume_mm": [90, 90, 90],
  "machine_plate_center_mm": [113, 52, 0],
  "rotation_center_machine_mm": [112, 51, 2]
}
```

`machine_plate_center_mm` and `rotation_center_machine_mm` are in machine
coordinates. Pentos derives the slicer-local plate center, machine offset, and
local rotation center from these values.

## Project Layout

- `main.py` starts the Viser server and mounts the application controller.
- `models/` stores shared application state, plane snapshots, and preview data.
- `controllers/` coordinates setup, preview, slicing, export, and navigation.
- `views/` contains the Viser UI, scene rendering, plane editor, and theme.
- `services/` contains model/project I/O, slicing, preview parsing, and
  Pentos-specific G-code workflows.
- `gcode_tools/` contains reusable G-code parsing and transformation utilities.
- `machine.py` stores machine geometry constants.
- `samples/` contains example models and saved Pentos scenes.

Generated runtime files are written to `uploaded_models/`, `temp/`, and
`output/`. These are local outputs and should not be committed.

## App Structure

`AppController` is the composition root. It creates the shared `AppState`,
services, screen controllers, and views:

- Models hold data independently of Viser and external processes.
- Controllers mutate application state and coordinate services.
- Services perform geometry, filesystem, slicing, parsing, and network work.
- Views create Viser handles, render state, and forward user actions.

Plane edits update controller-owned snapshots as they happen. Runtime plane IDs
are not included in version-1 `.pentos` manifests, so existing scenes remain
compatible.

## Development Checks

Format touched Python files:

```bash
uv run ruff format .
```

Run tests and a quick syntax check:

```bash
uv run pytest
uv run python -m compileall .
```

For visible UI changes, run the app, load a sample model, add a plane, and
exercise the setup-to-preview flow.

## License

Pentos is licensed under the [GNU General Public License v3.0](LICENSE).
