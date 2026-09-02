# Autonomous Pentos Visualization Pipeline Plan

## Context

Create a reproducible collection of short, caption-only technical animations that explains the Pentos printer and its two working slicing workflows to funders, judges, and general viewers. The visuals must be polished and accessible without inventing machine behavior or slicing results.

The initial examples are:

- `samples/Tube.pentos` for multiplanar cutting, chunk orientation, and A/B transitions.
- `samples/flat_base_curved_top.pentos` for guide-controlled nonplanar deformation and inverse-mapped toolpaths.
- `assets/Pentos_URDF/urdf/Pentos_URDF.urdf` and its STL meshes for machine construction and joint motion.

Implementation will use the branch already prepared by the user. The pipeline will run locally without hosted AI or media-generation APIs. Blender 5.1.1, FFmpeg 8.1, and PrusaSlicer are already available on the machine.

## Approach

### Pipeline architecture

Build a deterministic, resumable pipeline with three stages:

1. **Prepare verified scene data with the project environment.** Load each `.pentos` scene through the existing project loader, run the existing decomposition/deformation/slicing services, parse generated G-code, and export Blender-friendly meshes plus compact JSON/NPZ animation data. This stage owns all `trimesh`, SciPy, TetGen, PrusaSlicer, and Pentos-specific calculations.
2. **Render headlessly with Blender Python.** Import prepared STL/PLY/JSON data and the URDF meshes, construct the machine hierarchy, animate cameras and geometry from declarative shot definitions, and render resumable PNG frame sequences. Use Eevee, restrained lighting, clean technical materials, and the Pentos palette rather than generative imagery.
3. **Caption and encode with FFmpeg.** Generate timed ASS caption files from the shot definitions, burn captions into the frame sequences, add the existing Pentos logo at the opening/closing edge of each short, and encode H.264 MP4 files in native 1920×1080 and 1080×1920 layouts.

The top-level command will prepare inputs, render all six videos in both aspect ratios, resume missing frames, encode successful outputs, and write a machine-readable and human-readable run report. A failure in one video will not block unrelated videos. Failed or stale data will never be replaced with conceptual or fabricated motion.

### Visual system

Use the existing application palette from `views/theming.py`:

- Pentos blue `#2F99EE` for primary geometry and completed toolpaths.
- Pentos orange `#FF8200` for guides, active axes, cut planes, and emphasis.
- Overhang red `#EF4444` for warnings or risky geometry only.
- Dark build-plate gray `#2D2D2D`, neutral gray machine components, and a dark background.

Use `assets/logo.png` sparingly on opening/closing frames. Captions will use plain language first and introduce the technical term second. Videos will have captions only: no narration, music, or external audio dependency. Compose shots separately for horizontal and vertical output so labels and geometry remain legible rather than relying on a blind crop.

### Initial six videos

1. **Meet Pentos** — Orbit the assembled URDF machine, isolate the Cartesian toolhead and rotating bed, then identify X/Y/Z and A/B as the five controlled axes.
2. **Five Axes in Motion** — Animate one joint at a time from neutral, then combine them. Show direction arrows and short labels. Normalize URDF transforms to the machine conventions documented in `AGENTS.md` and exercised by `machine.rotation_matrix`; do not infer G-code signs from mesh orientation alone.
3. **Multiplanar: Divide the Tube** — Load `Tube.pentos`, show its saved cut plane, call the existing mesh decomposition, separate the resulting chunks, and show each chunk being flattened for conventional slicing. Use the real sample geometry and plane pose.
4. **Multiplanar: Reorient and Continue** — Run the full multiplanar slice for the Tube scene with debug mode disabled for this visualization job, parse the merged G-code, show the first chunk, the safe Z lift, actual A/B transition, compensated move to the continuation point, and the next chunk toolpath. Clearly distinguish temporary PrusaSlicer centering from physical placement.
5. **Nonplanar: Guides to Flat Space** — Load the five saved curved guides from `flat_base_curved_top.pentos`, display the solved scalar field on the tetrahedralized model, and morph between original and deformed vertices to show curved layers becoming planar slicing space.
6. **Nonplanar: Map the Toolpath Back** — Slice the actual deformed model with PrusaSlicer, show representative planar paths, inverse-map those paths into the original object, and visualize the resulting XYZAB orientation changes. Include the working planar base-layer and transition behavior rather than implying every layer begins fully nonplanar.

For long G-code, preprocess representative, time-ordered path windows rather than loading every segment into Blender at once. Selection must remain deterministic and the captions must not imply omitted paths were simulated.

### Reliability and autonomy

- Cache prepared assets using hashes of source samples, machine configuration, Prusa profile, and relevant pipeline version.
- Write frames atomically and skip already valid frames so interrupted renders can resume.
- Validate required executables, source files, output dimensions, frame counts, and generated G-code before rendering.
- Use per-video status records and continue after isolated failures.
- Capture subprocess logs and produce a final `report.md`/JSON summary with successes, failures, durations, and output paths.
- Do not build a self-modifying runtime agent. The coding harness will iteratively fix implementation defects; the finished overnight command itself remains deterministic and retryable.

## Files to modify

Critical planned paths (exact module splitting may be adjusted during implementation while keeping this structure small):

- `.gitignore` — ignore visualization caches, frame sequences, logs, reports, and rendered MP4 files while allowing selected poster frames.
- `visualizations/README.md` — prerequisites, one-command usage, output layout, recovery behavior, and video catalog.
- `visualizations/render_all.py` — top-level orchestration, prerequisite checks, caching, subprocess execution, continuation, and reporting.
- `visualizations/prepare_assets.py` — load `.pentos` files, invoke current workflows, and export verified Blender-friendly data.
- `visualizations/storyboards.py` — declarative videos, shots, timing, captions, camera framing, and horizontal/vertical layout parameters.
- `visualizations/blender/render.py` — Blender-only scene construction, materials, URDF hierarchy, animation, and frame rendering.
- `visualizations/blender/geometry.py` — small Blender helpers for paths, arrows, guide surfaces, labels, and mesh morphs when reuse warrants separation.
- `visualizations/encode.py` — ASS generation, logo/caption composition, FFmpeg encoding, and output validation.
- `visualizations/posters/` — one selected lightweight poster image per video after review.
- `tests/test_visualization_pipeline.py` — focused tests for storyboard validation, cache keys, prepared-data schemas, command construction, and failure continuation without running full Blender renders.

Generated files will live under ignored `visualizations/generated/` subdirectories for cache data, frames, logs, reports, and MP4 outputs.

## Reuse

Use existing production behavior rather than reimplementing geometry or kinematics:

- `services.project_io.load_scene` — load model, plane, guide, mode, and placement state from both `.pentos` samples.
- `services.model_tools.transformed_model` — apply saved model placement consistently with the UI.
- `services.slicing.decompose_mesh` and `Slicer.export_stl_chunks` — produce real multiplanar pieces and their flattening/orientation metadata.
- `services.slicing.Slicer.slice` — invoke PrusaSlicer and merge the complete multiplanar output.
- `services.multiplanar_gcode.merge_gcode_files`, `apply_chunk_offsets`, and existing transition generation — retain current continuation offsets and A/B transition semantics.
- `controllers.nonplanar_controller.NonplanarController.deformed_mesh` or its underlying services — use the same tetrahedralization and deformation path as the app without duplicating algorithms.
- `services.volumetric_deformation.tetrahedralize`, `solve_guide_scalar_field`, and `solve_guide_deformation` — export the actual scalar field and original/deformed volume vertices used by the nonplanar workflow.
- `services.nonplanar_gcode.map_gcode_to_original` — produce real mapped nonplanar XYZAB G-code, including subdivision, base layers, transition blending, orientation smoothing, and extrusion correction.
- `services.gcode_preview.parse_gcode_preview` and `transform_preview_point` — classify setup/travel/extrusion paths and place merged toolpaths back in object space.
- `gcode_tools.iter_gcode_moves`/`GcodeCommand` — extract time-ordered XYZAB animation samples without adding another parser.
- `machine.rotation_matrix` and `models.machine_config.DEFAULT_MACHINE_CONFIG` — preserve the tested A/B composition, pivot, machine offset, and bounds.
- `models.guide_surface.guide_surface_mesh` — render the same curved guide geometry shown in the app.
- `views.theming` constants and `assets/logo.png` — keep visual identity aligned with the current UI.
- URDF joint origins, axes, limits, and STL meshes under `assets/Pentos_URDF/` — construct the machine scene while explicitly adapting package mesh paths and the leading-space ` y_carriage_link` name.

## Steps

- [ ] Confirm the user-prepared working branch and current status before implementation.
- [ ] Add the visualization directory, ignored generated-output layout, README, and a top-level dry-run command.
- [ ] Define validated storyboard data for all six videos, including shot timing, accessible captions, horizontal/vertical camera layouts, and expected prepared assets.
- [ ] Implement source hashing, executable checks, per-stage cache manifests, structured logs, and continue-on-failure reporting.
- [ ] Implement the preparation stage for shared machine data: parse the URDF hierarchy, resolve package mesh paths, record joint transforms/limits, and define the tested Pentos machine-axis normalization.
- [ ] Implement Tube preparation using `load_scene`, transformed geometry, `decompose_mesh`/chunk export, full PrusaSlicer output, chunk metadata, and parsed transition/toolpath samples. Override the sample's saved debug flag only in the isolated visualization state; do not alter the sample file.
- [ ] Implement nonplanar preparation using the saved guide surfaces, tetrahedral scalar field, original/deformed volume vertices, deformed PrusaSlicer output, mapped XYZAB G-code, and parsed representative paths.
- [ ] Add preparation-stage schema and sanity checks: finite geometry, matching deformation topology, nonempty chunks/toolpaths, expected transition commands, legal A/B values, and source/cache provenance.
- [ ] Implement the Blender renderer with deterministic world settings, Pentos materials, orthographic/perspective camera helpers, URDF machine hierarchy, axis arrows, guide surfaces, mesh chunk animation, path reveal animation, and original-to-deformed vertex morphing.
- [ ] Implement native 16:9 and 9:16 shot composition, safe caption regions, opening/closing logo treatment, and resumable atomic PNG rendering.
- [ ] Implement ASS caption generation and FFmpeg encoding to 1080p H.264/yuv420p MP4, retaining caption timing independently from expensive 3D rerenders.
- [ ] Add focused automated tests for non-Blender logic and low-resolution Blender smoke renders for one machine shot, one multiplanar shot, and one nonplanar shot.
- [ ] Run the complete batch, inspect reports and contact sheets, iterate on failed or unclear shots, and rerun only invalidated/missing outputs.
- [ ] Select one poster frame per approved video for tracking; leave MP4 files and intermediate frames ignored.
- [ ] Review the implementation diff to ensure no unrelated application behavior, sample data, or slicer configuration was changed.

## Verification

### Automated checks

- `uv run pytest` for the existing suite plus visualization orchestration/schema tests.
- `uv run python -m compileall .` for syntax validation.
- `uv run ruff format .` and the repository's Ruff checks for touched Python files.
- Run the visualization command in dry-run mode to verify tools, source assets, cache keys, expected outputs, and commands without rendering.
- Run preparation only and assert both sample workflows generate nonempty, provenance-tagged data from current production services.
- Run low-resolution Blender smoke renders and use `ffprobe` to assert codec, dimensions, frame rate, duration, and pixel format.

### End-to-end batch

- Execute the one-command batch from a clean visualization cache and confirm all twelve MP4 variants are generated or explicitly reported failed.
- Interrupt a render, rerun it, and verify valid frames are reused and the completed MP4 matches the requested frame count.
- Force one video preparation/render failure and verify unrelated videos finish while the report clearly identifies the failure.
- Change a sample or relevant machine/slicing input and verify its dependent cache invalidates without unnecessarily invalidating unrelated outputs.

### Visual and mechanical review

- Compare URDF neutral and extreme poses against the source meshes and joint hierarchy.
- Verify displayed A/B signs and combined poses against `machine.rotation_matrix`, repository tests, and the conventions in `AGENTS.md`.
- Confirm Tube chunks, flattening offsets, transition lift, A/B command, and continuation point come from generated workflow data.
- Confirm the nonplanar morph uses matching original/deformed tetrahedral vertices and the displayed mapped path comes from generated mapped G-code.
- Ensure captions remain readable in both aspect ratios, avoid overclaiming omitted computation, and introduce technical terms only after plain-language explanations.
- Confirm no output depicts conceptual collision validation or other functionality not exercised by the current pipeline.
