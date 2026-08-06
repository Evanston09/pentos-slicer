# Pentos URDF re-export notes

Ask your AI agent to apply these after replacing/re-exporting the URDF from SW2URDF.

## Joint configuration

| Joint | Axis | Lower | Upper | Notes |
|---|---:|---:|---:|---|
| `z_joint` | `0 0 1` | `0` | `0.19` | Positive control must move up; 190 mm travel. |
| `x_joint` | `-1 0 0` | `0` | `0.22` | Exported direction was backwards; 220 mm travel. |
| `y_joint` | `0 -1 0` | `0` | `0.15` | Exported direction was backwards; 150 mm travel. |
| `a_joint` | keep exported axis | `-1.5708` | `1.5708` | 90 degrees each way. |
| `b_joint` | keep exported axis | `-3.14159` | `3.14159` | 180 degrees each way. |

X, Y, and Z controls start at `0` and must not allow negative positions.

Current effort/velocity values:

- X/Y: `effort="100" velocity="0.1"`
- Z: `effort="100" velocity="0.02"`
- A/B: `effort="10" velocity="1"`

## Viser viewer

- Point `URDF_PATH` in `main.py` at the newly exported `.urdf`.
- Keep one slider per actuated joint, with `initial_value=0.0` and `step=0.001`.
- The current `Pentos_URDF` export is upright, so it needs no root rotation.
- Keep the ground grid at Z = 0 and camera target near `(0, 0, 0.25)`.

## Re-export checklist

1. Replace the package and update `URDF_PATH` if its name changed.
2. Reapply the five joint limits above; SW2URDF exports them as zero.
3. Reapply the corrected X, Y, and Z axes above.
4. Confirm all `package://.../meshes/...` paths use the new package name.
5. Start with `uv run python main.py` and verify each slider's physical direction.
6. If a future export appears upside down, inspect it first; do not automatically restore the old 180-degree rotation.
