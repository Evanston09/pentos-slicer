# Nonplanar Slicing Concept

## Status

This document records the current research direction for adding true nonplanar
slicing to Pentos. It is a concept and architecture reference, not an implemented
feature.

## Decision

The preferred approach is a **tween-controlled scalar field with a
warp-slice-unwarp pipeline**:

1. Pentos suggests a small number of important curved guide surfaces.
2. The user accepts, moves, reshapes, adds, or removes those surfaces.
3. A scalar-field solver fills the model with smoothly interpolated curved layers.
4. Pentos warps the model so those curved layers become horizontal.
5. PrusaSlicer slices the warped model normally.
6. Pentos maps every G-code point back into the original shape.
7. The local layer direction is converted into compensated XYZAB motion.
8. Extrusion, speed, machine limits, and collisions are checked before export.

In short: **the user controls a few key surfaces, PrusaSlicer generates paths in
flattened space, and Pentos transforms those paths into safe five-axis curved
layers.**

## Why This Approach

It retains PrusaSlicer's mature handling of perimeters, infill, seams, thin walls,
travels, retractions, and print settings without limiting Pentos to discrete planar
chunks. It also avoids building an entirely new FFF toolpath engine at the start.

The existing multiplanar workflow should remain available. It is a useful fallback
for regions where continuous curved layers are unsafe or cannot be flattened
reliably.

## Terminology

### Tween surfaces

Tween surfaces are the small set of important layer shapes shown to the user. They
act like keyframes in animation: the user specifies a few meaningful states and the
system generates the intermediate states.

### Scalar field

The scalar field assigns a progress value to every point in the model. For example,
the starting surface may have value `0`, a middle guide value `0.5`, and the ending
surface value `1`. Equal-value surfaces form the complete set of curved layers.

The field should be optimized for smoothness, valid layer spacing, overhang
reduction, reachable A/B poses, low angular motion, and collision clearance.

### Warping

Warping transforms the original model so that equal scalar-field values become
ordinary horizontal Z levels. The inverse transformation maps PrusaSlicer's planar
toolpaths back into the original model.

## Proposed User Workflow

1. Upload a mesh and choose **Curved Layers**.
2. Click **Suggest Guide Surfaces**.
3. Compare suggestions such as minimum support, best finish, and easiest motion.
4. Select a strategy and edit its guide surfaces in the viewport.
5. Preview the generated layer field and warnings.
6. Flatten the model and run PrusaSlicer.
7. Inverse-map and preview the resulting continuous XYZAB toolpath.
8. Inspect layer thickness, extrusion, tilt, and collision visualizations.
9. Adjust the guides and regenerate if necessary.
10. Export only after the machine simulation passes.

Automatic, assisted, and manual editing should use the same underlying model. The
automatic system proposes guides; it does not remove the user's final control.

## Transformation and G-code Requirements

Long PrusaSlicer moves must be subdivided before inverse mapping so curved motion is
approximated within a defined positional and angular tolerance. At every resulting
point, Pentos must calculate:

- The inverse-mapped object position.
- The scalar-field gradient, representing local print-up direction.
- The corresponding A/B bed pose.
- XYZ compensation around the real `ROTATION_CENTER`.
- Extrusion correction for transformed path length and local layer thickness.
- Feed-rate correction and A/B velocity and acceleration limits.

The firmware does not perform A/B coordinate compensation, so the slicer must emit
already-compensated XYZ coordinates.

## Safety Requirements

Before machine-ready export, the complete motion should be checked for:

- Nozzle, heater block, fan shroud, bed, and printed-part collisions.
- Intersecting or reversed layers.
- Minimum and maximum local layer thickness.
- Unreachable A/B orientations.
- Excessive angular changes, velocity, or acceleration.
- XYZ travel violations.
- Unsupported deposition and unsafe travel moves.

Collision checking is a core requirement of the approach, not an optional preview
feature.

## Relationship to Existing Work

The broad method combines established research ideas and should not be described as
inventing nonplanar deformation or tweened slicing:

- [Print Paths Key-framing](https://doi.org/10.1145/3424630.3425408) uses
  user-defined target curves and interpolated distance fields to generate nonplanar
  paths.
- [COMPAS Slicer](https://compas.dev/compas_slicer/latest/) provides open-source,
  MIT-licensed interpolation and scalar-field slicing research code.
- [RotBot Nonplanar Slicing](https://github.com/RotBotSlicer/Nonplanar_Slicing)
  transforms an STL, runs PrusaSlicer, and retransforms the resulting G-code.
- [Adaptation of Conventional Toolpath-Generation Software for Curved-Layer
  FDM](https://doi.org/10.3390/jmmp8060270) documents and evaluates a similar
  PrusaSlicer transformation pipeline.
- [S4 Slicer](https://github.com/jyjblrd/S4_Slicer) tetrahedralizes and deforms a
  volume, conventionally slices it, and maps the paths back for multi-axis output.

These projects are references for algorithms and prior art. Pentos should not adopt
their machine assumptions without independently deriving and testing its own
kinematics.

## Potential Research Contribution

The strongest contribution is the integrated system rather than any single basic
technique:

> An interactive, key-surface-constrained volumetric deformation system that uses
> conventional planar slicing to generate inverse-mapped, collision-validated
> continuous five-axis toolpaths for a two-axis rotating-bed FFF printer.

Potentially distinguishing features include automatic guide-surface suggestions,
user-constrained volumetric deformation, Pentos-specific XYZAB pivot compensation,
joint optimization of printability and machine motion, and hybrid curved/multiplanar
planning.

## Intended Final Architecture

Pentos should eventually expose two complementary engines:

- **Multiplanar:** the current discrete chunk, reorient, PrusaSlicer, and merge
  workflow.
- **Curved:** guide surfaces, scalar field, warp, PrusaSlicer, inverse mapping,
  continuous XYZAB compensation, and collision validation.

This preserves the reliable existing workflow while providing a path toward true
continuous nonplanar printing.
