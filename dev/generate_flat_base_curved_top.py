"""AI Generated: Generate the flat-base, parabolic-top sample and its guide surfaces."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

CENTER_X = 45.0
CENTER_Y = 45.0
WIDTH = 40.0
DEPTH = 20.0
GUIDE_HEIGHTS = (0.0, 2.5, 5.0, 7.5, 10.0)


def make_mesh(top_height: float, curvature: float, segments: int) -> trimesh.Trimesh:
    """Create z = top_height - curvature * (x - CENTER_X)^2."""
    half_width = WIDTH / 2.0
    x = np.linspace(CENTER_X - half_width, CENTER_X + half_width, segments + 1)
    top_z = top_height - curvature * (x - CENTER_X) ** 2
    if np.min(top_z) <= 0.0:
        raise ValueError("the parabola must remain above the flat base")

    # Each end is the cross-section loop: base left/right, then top right-to-left.
    profile_x = np.concatenate(([x[0], x[-1]], x[::-1]))
    profile_z = np.concatenate(([0.0, 0.0], top_z[::-1]))
    profile_size = len(profile_x)
    vertices = np.vstack(
        (
            np.column_stack(
                (profile_x, np.full(profile_size, CENTER_Y - DEPTH / 2), profile_z)
            ),
            np.column_stack(
                (profile_x, np.full(profile_size, CENTER_Y + DEPTH / 2), profile_z)
            ),
        )
    )

    faces: list[list[int]] = []
    for index in range(profile_size):
        following = (index + 1) % profile_size
        faces.extend(
            (
                [index, following, profile_size + following],
                [index, profile_size + following, profile_size + index],
            )
        )
    for index in range(1, profile_size - 1):
        faces.extend(
            (
                [0, index + 1, index],
                [profile_size, profile_size + index, profile_size + index + 1],
            )
        )

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    assert mesh.is_watertight
    return mesh


def make_manifest(top_height: float, curvature: float) -> dict:
    """Recreate the guide surfaces stored in the sample manifest."""
    scale = top_height / GUIDE_HEIGHTS[-1]
    guide_heights = [height * scale for height in GUIDE_HEIGHTS]
    return {
        "format": "pentos",
        "version": 2,
        "original_model_name": "flat_base_curved_top",
        "model_xy_position": [CENTER_X, CENTER_Y],
        "model_z_degrees": 0.0,
        "plane_snapshots": [],
        "guide_surfaces": [
            {
                "position": [CENTER_X, CENTER_Y, height],
                "wxyz": [1.0, 0.0, 0.0, 0.0],
                "bend_x": -curvature * height / top_height,
                "bend_y": 0.0,
            }
            for height in guide_heights
        ],
        "slicing_mode": "nonplanar",
        "debug_mode": False,
    }


def generate(output: Path, top_height: float, curvature: float, segments: int) -> None:
    mesh = make_mesh(top_height, curvature, segments)
    manifest = make_manifest(top_height, curvature)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("model.3mf", mesh.export(file_type="3mf"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a .pentos model with a configurable parabolic top"
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("output/flat_base_curved_top.pentos"),
    )
    parser.add_argument("--top-height", type=float, default=10.0)
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.0025,
        help="positive coefficient in z = top_height - curvature * (x - 45)^2",
    )
    parser.add_argument("--segments", type=int, default=80)
    args = parser.parse_args()
    if args.top_height <= 0.0 or args.curvature < 0.0 or args.segments < 2:
        parser.error(
            "top-height must be positive, curvature nonnegative, and segments at least 2"
        )
    generate(args.output, args.top_height, args.curvature, args.segments)
    print(args.output)


if __name__ == "__main__":
    main()
