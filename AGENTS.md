# Repository Guidelines

## Project Structure & Module Organization

Pentos Slicer is a Python 3.13 Viser application organized into MVC-style
packages. `main.py` creates the server and one `AppController` per browser
client. `models/` contains shared session state and data containers;
`controllers/` owns workflow and navigation; `views/` owns Viser GUI and scene
handles; and `services/` contains mesh/project I/O, plane selection, slicing,
volumetric deformation, G-code mapping, and preview parsing. Reusable G-code
parsing and transformations belong in `gcode_tools/`. The A/B rotation matrix
lives in `machine.py`; machine profile data lives in
`models/machine_config.py`; theme and build-plate rendering live in
`views/theming.py`.

Static assets and example inputs live in `assets/` and `samples/`.
`pentos_config.ini` is the PrusaSlicer profile used by `services/slicing.py`.
Design notes live in `docs/`. Development helpers live in `dev/`.

Runtime data is isolated by client in a `SessionWorkspace` below the operating
system temporary directory, normally `/tmp/pentos-slicer/<client-id>-*/`.
Uploads, intermediate STLs/G-code, and final G-code are temporary and are
removed after disconnect once an active slice finishes. Do not add new
repository-root runtime paths. `uploaded_models/`, `temp/`, and `output/` remain
ignored as legacy/local scratch locations.

## Nonplanar and Firmware References

Ignored local reference checkouts are available in `reference/`:

- `reference/S3_DeformFDM/` contains the BSD-3-Clause S³-Slicer
  implementation.
- `reference/S4_Slicer/` contains the GPL-3.0 S4 Slicer implementation and
  notebook.
- `reference/klipper/` contains a Klipper source checkout for firmware-command
  and motion semantics.

Inspect S3/S4 when changing tetrahedral deformation, guide-constrained scalar
fields, inverse mapping, extrusion correction, or multi-axis motion. Pentos is
GPL-3.0 licensed, so S4 code may be adapted while preserving its copyright and
attribution. Use the Klipper checkout only to verify upstream behavior; Pentos
macros such as `HOME_A`, `HOME_B`, and `ENABLE_FIVE_AXIS` are machine-specific.

`docs/nonplanar-slicing-concept.md` records the longer-term research direction.
The core warp-slice-unwarp path is now implemented, but the document's collision,
reachability, and layer-quality checks are still future requirements.

## Build, Test, and Development Commands

- `uv sync --locked` installs the environment from `pyproject.toml` and
  `uv.lock`.
- `uv run python main.py` starts the Viser app and prints its browser URL.
- `./dev/reload.sh` restarts the app when Python files change.
- `uv run ruff format .` formats the Python tree.
- `uv run ruff format --check .` verifies formatting as CI does.
- `uv run ruff check .` runs lint checks.
- `uv run pytest` runs the automated test suite.
- `uv run python -m compileall .` performs a quick syntax check.

Slicing requires the external `prusa-slicer` executable on `PATH` and invokes it
with `pentos_config.ini`. The Docker image installs PrusaSlicer. At runtime,
`MAX_CONCURRENT_SLICES` controls the server-wide slice semaphore (default `2`),
and `MAX_UPLOAD_SIZE_MB` controls the upload limit (default `50`). Both must be
positive integers.

## Application and Persistence Context

Each connected client has its own `AppState`, controllers, views, slicer, and
temporary workspace. Only the slice semaphore is shared across clients. On
disconnect, controller handles and workspace files must be cleaned up without
affecting another session.

The active `MachineConfig` is stored in browser local storage under
`pentos-machine-config`. Machine profiles use the `pentos-machine` JSON format,
currently version 1. Saved `.pentos` projects are ZIP archives containing
`manifest.json` and `model.3mf`; new saves use manifest version 2 and include
model placement, planes, guide surfaces, slicing mode, and debug mode. Runtime
plane and guide IDs are reconstructed when loading and are not serialized.

Keep model state independent of Viser handles. Controllers update canonical
snapshots as edits occur; views render that state and forward callbacks;
services do filesystem, subprocess, geometry, and G-code work. Preserve the
import boundaries enforced by `tests/test_architecture.py`.

## Machine Mechanics Context

The Pentos machine has X/Y/Z Cartesian toolhead motion plus A/B bed rotation.
The firmware does not compensate XYZ when A or B changes, so every required
pivot transformation must be applied by the slicing/mapping pipeline before the
G-code reaches the printer.

Coordinate frame:

- `X`, `Y`, and `Z` are literal Cartesian machine axes.
- "From the front" means standing at `Y=0` and looking toward increasing `Y`
  (`Y=235` on the current machine).
- "From the top" means looking down along `-Z`.
- The default slicer-local build volume is `90 x 90 x 90 mm`, with plate center
  `[45, 45, 0]`.
- The default real machine plate center is `[113, 52, 0]`.
- The default real A/B rotation center is `[112, 51, 2]`, which maps to
  slicer-local `[44, 44, 2]`.
- `MachineConfig.machine_offset` maps slicer-local plate coordinates to the real
  machine plate center. `rotation_center_local_mm` derives the pivot used for
  transformations.

These are profile defaults, not universal constants. Geometry changes belong in
`MachineConfig` and the versioned JSON import/export path, not as new module
globals. The active profile may also carry A/B velocity and acceleration values,
the permitted B range, and the maximum normal error. Currently the nonplanar
mapper enforces the B range and normal-error bound; the velocity and acceleration
fields are recorded but are not yet enforced as hard motion limits.

A/B conventions:

- `A = 0` means the bed/chunk is flat.
- Positive A tilts clockwise from the front: the `-X` side rises and the `+X`
  side lowers.
- B is bed/spindle rotation. Positive B spins clockwise from the top: a mark on
  the `+X` side moves toward `Y=0`.
- `rotation_matrix(a, b)` composes tilt and twist as tested in
  `tests/test_machine.py`. The final pose is path-independent whether A or B is
  commanded first or both are commanded together.

Slice plane normals are `print_up_normal`: the direction a chunk prints from
bottom to top in the final object frame. For example,
`print_up_normal = [-1, 0, 0]` corresponds to `A90 B0` with the current
conventions. Preview rendering is a sanity check; measured machine behavior is
the source of truth for axis signs, pivot position, and offsets.

## Slicing Pipeline Context

For multiplanar slicing, planes decompose the transformed source mesh in list
order. Non-base pieces are rotated about the configured local pivot, placed on
the local PrusaSlicer bed, and assigned a temporary `flat_xy_offset`. During
merge, Pentos applies the machine offset, removes `flat_xy_offset`, restores the
piece's `z_offset`, and inserts a 15 mm relative Z lift before each A/B
transition. The centering offset is never a physical machine target. Debug mode
emits only the transition-check motion rather than a full print.

For nonplanar slicing, at least two guide surfaces constrain a scalar field on a
tetrahedral volume. The volume is flattened, its boundary is sliced once by
PrusaSlicer, and moves after the first two planar layers are subdivided and
inverse-mapped. Mapping blends in over four layers, adjusts extrusion and feed
rate, smooths normals/A-B angles within the configured error, and compensates
XYZ around the local rotation center before adding the machine offset. It
requires relative extrusion for mapped positive-extrusion moves.

Do not describe generated paths as collision-validated or generally
machine-safe. Full collision, reachability, local layer-thickness, and angular
limit validation remain future work. Treat real-machine testing as a deliberate
hardware validation step.

## Keep Changes Simple

Prefer the smallest coherent change that solves the requested problem. Reuse
existing control flow and data structures before introducing abstractions,
background work, debounce logic, configuration, or generalized APIs. Do not
modify unrelated code or add defensive machinery for states the application
cannot produce. Add complexity only for a concrete requirement, failure mode,
or measured performance problem.

Keep tests focused on the behavior being added or fixed. Cover the important
path and meaningful edge cases without expanding the public surface solely for
testability. After a change works, inspect the diff and remove redundant state,
duplicate refreshes, unnecessary callbacks, and one-use helpers when inline code
is clearer.

## Coding Style & Naming Conventions

Use Ruff formatting and four-space indentation. Prefer type annotations for
public functions and data containers; existing modules use `dataclass`,
`Protocol`, and explicit `Path`/NumPy types where useful. Use lowercase
underscore module names, `snake_case` functions and variables, and
`UPPER_SNAKE_CASE` constants. Keep comments short and reserve them for
non-obvious geometry, machine, or G-code behavior.

## Testing Guidelines

Run formatting, linting, and `uv run pytest` for changes. Use focused tests while
iterating, then the full suite before handoff. For visible changes, run the app
and load a representative sample or uploaded model. For multiplanar changes,
exercise manual/automatic planes and inspect transition G-code. For nonplanar
changes, load a closed sample, use at least two guides, and inspect the mapped
XYZAB path and rotary plot. Run PrusaSlicer when the external executable is
available. Generated artifacts are in the active session workspace and should be
downloaded through the UI when they need to be retained.
