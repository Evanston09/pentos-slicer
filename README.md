# Pentos Slicer

Pentos Slicer is an experimental browser-based toolpath pipeline for the Pentos
five-axis FFF printer: an XYZ toolhead with an A/B rotating bed. It uses
[Viser](https://viser.studio/) for the interactive 3D UI and PrusaSlicer as the
planar toolpath engine.

The application currently supports two workflows:

- **Multiplanar** divides a model with editable planes, flattens and slices each
  chunk, then merges the results with Pentos A/B transition moves.
- **Nonplanar** uses editable guide surfaces to deform a tetrahedralized model,
  slices the flattened mesh, and inverse-maps the G-code into continuous XYZAB
  motion with extrusion and pivot compensation.

> [!WARNING]
> This is research software. Generated G-code is machine-specific and the
> pipeline does not yet perform complete collision, reachability, or local layer
> thickness validation. Inspect previews and G-code before using real hardware.

## Requirements

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/)
- `prusa-slicer` on `PATH` for slicing (not needed to inspect the UI or load a
  saved project)

Install the locked Python environment:

```bash
uv sync --locked
```

## Run

Start the Viser server:

```bash
uv run python main.py
```

Open the URL printed in the terminal, normally `http://localhost:8080`. Viser
may choose the next available port if 8080 is already occupied.

For development, restart the server whenever a Python file changes:

```bash
./dev/reload.sh
```

The Docker image includes Python, the locked dependencies, and PrusaSlicer:

```bash
docker build -t pentos-slicer .
docker run --rm -p 8080:8080 pentos-slicer
```

Two slices may run concurrently by default. Set `MAX_CONCURRENT_SLICES` to a
positive integer to change that server-wide limit. Uploads are limited to 50 MB
by default; set `MAX_UPLOAD_SIZE_MB` to a positive integer to change the limit.

## Workflow

1. Upload a mesh (`.stl`, `.3mf`, `.obj`, or `.ply`) or a saved `.pentos`
   project. New mesh uploads are normalized to millimeters, centered on the
   local build plate, and placed at Z=0.
2. Adjust X/Y placement and Z rotation. **Show Overhangs** highlights geometry
   that the current multiplanar decomposition leaves unsupported.
3. Choose a slicing mode:
   - In **Multiplanar**, add planes manually or use **Auto Planes**. Edit a plane
     with its viewport gizmo, numeric controls, or **Snap to Face**.
   - In **Nonplanar**, add at least two guide surfaces. Position, orient, and
     bend them with the guide controls; **Visualize Scalar Field** previews the
     solved field on the model boundary.
4. Optionally use **Export Scene** to save the model, placement, mode, planes,
   guides, and debug setting as a `.pentos` project.
5. Choose filament and print settings, then click **Slice**. The UI reports deformation, PrusaSlicer, merge, and mapping
   progress before opening the preview.
6. Inspect extrusion and travel paths, the estimated print time, and the A/B
   motion graph when rotary commands are present. Use **Download G-code** to
   save the result.

Example meshes and projects for both workflows live in `samples/`.

## Slicing Modes

### Multiplanar

Planes are applied in order. The base piece keeps the flat bed pose; each piece
above a cut is oriented from its plane normal, shifted onto PrusaSlicer's local
build plate, and sliced separately. Pentos then removes the temporary centering
offset, applies the configured machine offset, and inserts a 15 mm relative Z
lift before each A/B transition.

**Debug Mode** emits a shortened transition-check program instead of the full
merged print, allowing the inter-chunk poses to be inspected on the machine.

### Nonplanar

Guide surfaces constrain a volumetric scalar field. Pentos tetrahedralizes and
flattens the model, runs a normal planar slice, subdivides the resulting moves,
then inverse-maps them through the deformation. The mapper compensates XYZ for
the bed pivot, adjusts extrusion and feed rate, and smooths A/B orientation
within the configured normal-error limit.

The field solver blends local curved-guide normals near each tetrahedron. Guide
list order defines increasing layer values; adjacent intervals use the mean
normal separation of the surfaces, weighted by model volume, rather than the
distance between guide origins. Adjacent guide footprints must overlap at sampled
model locations; enlarge the patches or refine the input mesh if they do not.
Detected touching, crossing, or reversed ordering at those samples stops the solve.
These samples do not establish that guides are disjoint everywhere.

Generated G-code includes a comment reporting RMS and maximum guide fit error in
**flattened millimeters**, measured at guide/volume-edge intersections. It measures
how closely the solved field follows
the guide constraints, not physical layer thickness or error everywhere on a guide.
A coarse tetrahedral mesh can miss curvature; inspect the field and consider a
finer input mesh or simpler guides when errors are large.

The first two layers remain planar and the mapping blends in over the following
four layers. This workflow requires a closed, tetrahedralizable mesh and at
least two guide surfaces.

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

The **Machine** panel imports, exports, resets, and displays versioned JSON
profiles. The active profile is stored in browser local storage and restored for
future sessions. The default exported profile is equivalent to:

```json
{
  "format": "pentos-machine",
  "version": 1,
  "name": "Default Pentos",
  "build_volume_mm": [90.0, 90.0, 90.0],
  "machine_plate_center_mm": [113.0, 52.0, 0.0],
  "rotation_center_machine_mm": [112.0, 51.0, 2.0],
  "a_max_velocity_deg_s": 10.0,
  "a_max_acceleration_deg_s2": 50.0,
  "b_max_velocity_deg_s": 20.0,
  "b_max_acceleration_deg_s2": 100.0,
  "max_normal_error_degrees": 2.0,
  "b_degrees_min": -180.0,
  "b_degrees_max": 180.0
}
```

`build_volume_mm` defines the slicer-local volume. The plate and rotation-center
fields use real machine coordinates; Pentos derives the local plate center,
machine offset, and local A/B pivot from them. PrusaSlicer settings remain in
`pentos_config.ini`.

The preview printer uses the ideal joint origins and axes in
`assets/Pentos_URDF/urdf/Pentos_URDF.urdf`. Its bed and displayed toolpath follow
that URDF directly. Fixed `nozzle_tip` and `plate_center` reference links define
preview alignment. At zero XYZ, the nozzle is centered on the plate; XYZ joint
origins and limits encode this reference position. Machine profiles do not
reshape or recalibrate the CAD model.
`rotation_center_machine_mm` remains the real-machine calibration used for
slicing and interpreting generated G-code. Consequently, simulation illustrates
the ideal mechanism rather than verifying real-machine pivot compensation.

## Project Layout

- `main.py` creates the Viser server and one isolated application session per
  connected client.
- `models/` contains application state, machine profiles, planes, guides, and
  preview data.
- `controllers/` coordinates setup, nonplanar deformation, slicing, preview,
  downloads, and navigation.
- `views/` owns Viser controls and scene rendering.
- `services/` contains model/project I/O, automatic plane selection, planar and
  nonplanar slicing, volumetric deformation, and G-code preview parsing.
- `gcode_tools/` contains reusable G-code parsing and transformation utilities.
- `machine.py` defines the tested A/B rotation matrix.
- `pentos_config.ini` is the PrusaSlicer profile used by the pipeline.
- `samples/` contains example meshes and `.pentos` projects.
- `docs/` contains design notes, and `reference/` contains ignored local
  research and firmware checkouts.

Each browser connection gets a temporary workspace under the operating system's
temporary directory (normally `/tmp/pentos-slicer/`). Uploads, intermediate
meshes, and generated G-code are removed after that client disconnects, once any
active slice completes. Download projects or G-code through the UI if you want
to keep them.

## Development Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run python -m compileall .
```

Use `uv run ruff format .` to format changes. For visible or machine-facing
work, also run the app, load representative projects from `samples/`, exercise
both slicing modes where relevant, and inspect the downloaded G-code.

## License

Pentos is licensed under the [GNU General Public License v3.0](LICENSE).
