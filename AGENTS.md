# Pentos Slicer

Pentos is a Python 3.13 application using Viser for its browser UI, trimesh for
geometry, and PrusaSlicer for slicing. See [README.md](README.md) for setup,
user workflows, and project details; read the sections relevant to the task.

## Project Structure

- `main.py` starts the app; `controllers/` coordinates workflows and state changes.
- `models/` holds application data independently of Viser and external processes.
- `views/` owns UI handles and rendering, forwarding user actions to controllers.
- `services/` handles geometry, I/O, slicing, and machine-specific workflows.
- `gcode_tools/` provides reusable G-code parsing and transformation utilities.
- `pentos_config.ini` is the baseline slicing profile; per-slice settings belong
  in the session's effective configuration.

Generated files in `uploaded_models/`, `temp/`, and `output/` are local runtime
artifacts and should not be committed.

## Keep Changes Simple

Make the smallest coherent change that fully solves the requested problem.
Follow existing patterns and reuse the current control flow and data structures.
Keep unrelated refactors and configuration changes out of the task.

Avoid overengineering: introduce abstractions, dependencies, configuration,
background work, or generalized APIs only when a concrete requirement or
demonstrated problem justifies them. Do not build for hypothetical future needs
or add defensive branches for states the application cannot produce. Prefer
straightforward code when a helper or extra layer would only add indirection.

Keep tests proportional to the change and focused on observable behavior and
meaningful invariants. Do not expand the public API solely to make tests easier.

## Slicer Code References

Local code references provide examples of how other slicers implement related
algorithms:

- [S4 Slicer](reference/S4_Slicer/)
- [S3 DeformFDM / S³-Slicer](reference/S3_DeformFDM/)

Consult relevant implementations when working on nonplanar slicing, tetrahedral
deformation, scalar fields, inverse mapping, extrusion correction, or multi-axis
motion. Search for the specific algorithm or operation and read the relevant
code as needed; there is no need to load both projects for every task.

Use these as implementation examples, adapting ideas to Pentos's architecture
and machine conventions. Check the source license and preserve required notices
and attribution when incorporating code.

## Machine Conventions

Pentos has X/Y/Z Cartesian toolhead motion and A/B bed rotation. Preserve these
physical conventions when changing geometry, slicing, G-code, or preview code.

### Coordinate Frames and Calibration

- X, Y, and Z are literal Cartesian machine axes. "From the front" means
  looking toward increasing Y. "From the top" means looking along -Z.
- Use the active `MachineConfig` in [models/machine_config.py](models/machine_config.py)
  for build dimensions, plate centers, and pivot calibration. Keep changeable
  values in configuration rather than duplicating them in these instructions.
- `machine_offset = machine_plate_center_mm - build_plate_center` maps
  slicer-local coordinates to machine coordinates by addition.
- The rotation pivot is configured separately from the plate center. Use
  `rotation_center_local_mm` for local transforms. Calibration depends on the
  bed, fixture stackup, and measured rotation axes.

### A/B Pose

- A=0 means the bed/chunk is flat. Positive A tilts clockwise from the front:
  the -X side rises and the +X side lowers.
- B rotates the circular bed/spindle. Positive B spins clockwise from the top:
  at A=0, a mark on the +X side moves toward decreasing Y.
- `rotation_matrix(a_degrees, b_degrees)` in [machine.py](machine.py) models the
  pose as `tilt @ twist`. Use this composition consistently. The physical final
  pose for a given A/B pair is independent of whether A, B, or both are commanded
  first; this does not mean the rotation matrices commute or the motion paths
  are identical.
- Firmware does not compensate X/Y/Z when A/B changes. The slicer/merge pipeline
  must produce the required compensated coordinates before sending G-code.

### Chunk Orientation and Merge

`print_up_normal` is the chunk's bottom-to-top print direction in the final
object frame. For example, `[-1, 0, 0]` maps to A90 B0. See
[services/slicing.py](services/slicing.py) and [tests/test_slice_tools.py](tests/test_slice_tools.py)
for the orientation implementation and examples.

Preparing a chunk rotates it around the configured pivot, lowers it by
`z_offset`, and centers it on the slicer plate using `flat_xy_offset`. During
merge, G-code is translated by `machine_offset`; non-base chunks also restore
`z_offset` and subtract `flat_xy_offset` in X/Y. Preserve these adjustments so
temporary slicer centering does not move the physical continuation point. See
[services/multiplanar_gcode.py](services/multiplanar_gcode.py).

Use preview rendering for sanity checks. Physical machine behavior is the
source of truth for A/B signs, pivot calibration, and offsets; a correct-looking
preview alone does not verify them.

## Validation

Use `uv sync` to install dependencies and `uv run python main.py` to run the app.
Full slicing requires `prusa-slicer` on `PATH`.

Format touched Python files with `uv run ruff format <paths>` and run relevant
tests with `uv run pytest <paths>`; run the full suite for changes spanning
multiple workflows. For visible UI changes, load a sample model and exercise
the affected controls and setup-to-preview flow. For slicing changes, inspect
the generated output when PrusaSlicer is available.

Report what was verified and any checks that could not be completed.

## Application and Persistence Context

Each connected client has its own `AppState`, controllers, views, slicer, and
temporary workspace. Only the slice semaphore is shared across clients. On
disconnect, controller handles and workspace files must be cleaned up without
affecting another session.

The active `MachineConfig` is stored in browser local storage under
`pentos-machine-config`. Machine profiles use the `pentos-machine` JSON format,
currently version 1. Saved `.pentos` projects are ZIP archives containing
`manifest.json` and `model.3mf`; new saves use manifest version 3 and include
model placement, planes, guide surfaces, slicing mode, slicing settings, and
debug mode. Runtime plane and guide IDs are reconstructed when loading and are
not serialized.

Keep model state independent of Viser handles. Controllers update canonical
snapshots as edits occur; views render that state and forward callbacks;
services do filesystem, subprocess, geometry, and G-code work. Preserve the
import boundaries enforced by `tests/test_architecture.py`.

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
