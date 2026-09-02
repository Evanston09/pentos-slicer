"""Headless Blender renderer for the Pentos visualization shorts.

Run inside Blender:
    blender -b --factory-startup -P visualizations/blender/render.py -- <config.json>

The config (written by render_all.py) lists shots with global 1-based frame
ranges and explicit frame numbers to (re)render, so interrupted batches resume
without touching valid frames. Frames are written atomically.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import (  # noqa: E402
    BUILD_PLATE_COLOR,
    DIM_GRAY,
    OVERHANG_RED,
    PENTOS_BLUE,
    PENTOS_ORANGE,
    TEXT_COLOR,
    arrow_object,
    dashed_line,
    ease_window,
    face_camera,
    get_emission_material,
    get_material,
    hide_state,
    lines_mesh,
    mesh_from_triangles,
    place_camera,
    read_binary_stl,
    rotation_arrow_object,
    set_blend_mode,
    setup_lighting,
    setup_world,
    text_object,
)

N_BUCKETS = 40
PATH_RADIUS = 0.34
TRAVEL_RADIUS = 0.24


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def configure_render(config: dict) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.fps = config["fps"]
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.eevee.taa_render_samples = 64
    for attribute in ("use_raytracing",):
        if hasattr(scene.eevee, attribute):
            setattr(scene.eevee, attribute, True)


def build_plate(size: tuple[float, float], center: tuple[float, float, float]) -> None:
    width, depth = size
    cx, cy, cz = center
    plate = get_material("plate", BUILD_PLATE_COLOR, roughness=0.6)
    top = mesh_from_triangles(
        "build_plate",
        np.array(
            [
                [
                    [cx - width / 2, cy - depth / 2, cz - 1.2],
                    [cx + width / 2, cy + depth / 2, cz - 1.2],
                    [cx + width / 2, cy - depth / 2, cz - 1.2],
                ],
                [
                    [cx - width / 2, cy - depth / 2, cz - 1.2],
                    [cx - width / 2, cy + depth / 2, cz - 1.2],
                    [cx + width / 2, cy + depth / 2, cz - 1.2],
                ],
                [
                    [cx - width / 2, cy - depth / 2, cz],
                    [cx + width / 2, cy - depth / 2, cz],
                    [cx + width / 2, cy + depth / 2, cz],
                ],
                [
                    [cx - width / 2, cy - depth / 2, cz],
                    [cx + width / 2, cy + depth / 2, cz],
                    [cx - width / 2, cy + depth / 2, cz],
                ],
            ]
        ),
        plate,
    )
    top.name = "build_plate"

    grid_material = get_emission_material(
        "plate_grid", (0.32, 0.36, 0.42), strength=0.7
    )
    segments = []
    step = 10.0
    x = 0.0
    while x <= width + 1e-6:
        segments.append([[x, 0.0, cz], [x, depth, cz]])
        x += step
    y = 0.0
    while y <= depth + 1e-6:
        segments.append([[0.0, y, cz], [width, y, cz]])
        y += step
    lines_mesh("plate_grid", np.asarray(segments), 0.1, grid_material)

    outline_material = get_emission_material(
        "plate_outline", PENTOS_ORANGE, strength=1.8
    )
    outline = np.array(
        [
            [[0, 0, cz], [width, 0, cz]],
            [[width, 0, cz], [width, depth, cz]],
            [[width, depth, cz], [0, depth, cz]],
            [[0, depth, cz], [0, 0, cz]],
        ],
        dtype=np.float64,
    )
    lines_mesh("plate_outline", outline, 0.28, outline_material)


def load_mesh_object(
    cache_dir: Path, filename: str, name: str, material
) -> bpy.types.Object:
    triangles = read_binary_stl(cache_dir / filename)
    return mesh_from_triangles(name, triangles, material)


def plane_rotation(wxyz: list[float]) -> Matrix:
    w, x, y, z = wxyz
    return Matrix(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    ).to_4x4()


def machine_rotation_matrix(a_degrees: float, b_degrees: float) -> Matrix:
    a = math.radians(a_degrees)
    b = math.radians(b_degrees)
    tilt = Matrix(
        (
            (math.cos(a), 0.0, math.sin(a)),
            (0.0, 1.0, 0.0),
            (-math.sin(a), 0.0, math.cos(a)),
        )
    )
    twist = Matrix(
        (
            (math.cos(b), math.sin(b), 0.0),
            (-math.sin(b), math.cos(b), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return (tilt @ twist).to_4x4()


def build_bucket_paths(
    points: np.ndarray,
    kinds: np.ndarray,
    seg_times: np.ndarray,
    total_time: float,
    name_prefix: str,
    bright_color,
    travel_color,
    clip_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[list[bpy.types.Object], np.ndarray]:
    """Bucket time-ordered path segments into reveal-able meshes.

    Returns (bucket_objects, bucket_start_times). Each bucket mesh has two
    material slots: 0 = extrusion (bright), 1 = travel/setup (faint).
    """
    extrusion_material = get_emission_material(
        f"{name_prefix}_extrusion", bright_color, strength=1.6
    )
    travel_material = get_emission_material(
        f"{name_prefix}_travel", travel_color, strength=0.55
    )
    if clip_bounds is not None:
        low, high = clip_bounds
        endpoints = points.reshape(-1, 3)
        inside = (
            np.all((endpoints >= low) & (endpoints <= high), axis=1)
            .reshape(-1, 2)
            .all(axis=1)
        )
        points = points[inside]
        kinds = kinds[inside]
        seg_times = seg_times[inside]

    objects = []
    start_times = []
    edges_total = len(points)
    if edges_total == 0 or total_time <= 0.0:
        return objects, np.asarray(start_times)

    order = np.argsort(seg_times, kind="stable")
    points, kinds, seg_times = points[order], kinds[order], seg_times[order]
    for bucket in range(N_BUCKETS):
        t0 = total_time * bucket / N_BUCKETS
        t1 = total_time * (bucket + 1) / N_BUCKETS
        mask = (
            (seg_times >= t0) & (seg_times < t1)
            if bucket < N_BUCKETS - 1
            else ((seg_times >= t0) & (seg_times <= t1))
        )
        bucket_points = points[mask]
        bucket_kinds = kinds[mask]
        if len(bucket_points) == 0:
            continue
        object_ref = lines_mesh(
            f"{name_prefix}_bucket_{bucket:02d}",
            bucket_points,
            PATH_RADIUS,
            extrusion_material,
        )
        travel = bucket_points[bucket_kinds != 2]
        if len(travel):
            travel_mesh = lines_mesh(
                f"{name_prefix}_bucket_{bucket:02d}_travel",
                travel,
                TRAVEL_RADIUS,
                travel_material,
            )
            travel_mesh.parent = object_ref
        hide_state(object_ref, False)
        for child in object_ref.children:
            hide_state(child, False)
        objects.append(object_ref)
        start_times.append(t0)
    return objects, np.asarray(start_times)


def make_dial(name: str, center: Vector, radius: float) -> dict:
    """A small bed indicator: disc + up arrow, rotated by A/B each frame.

    All parts are built in dial-local coordinates (centered on the origin) and
    parented to the disc pivot, whose matrix basis is T(center) @ R(a, b).
    """
    disc_material = get_material(f"{name}_disc", (0.16, 0.18, 0.22), roughness=0.5)
    ring_material = get_emission_material(f"{name}_ring", PENTOS_ORANGE, strength=2.0)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48, radius=radius, depth=1.6, location=(0, 0, 0)
    )
    disc = bpy.context.active_object
    disc.name = f"{name}_disc"
    disc.data.materials.append(disc_material)

    ring_segments = []
    for index in range(48):
        a0 = 2 * math.pi * index / 48
        a1 = 2 * math.pi * (index + 1) / 48
        ring_segments.append(
            [
                [radius * math.cos(a0), radius * math.sin(a0), 0.9],
                [radius * math.cos(a1), radius * math.sin(a1), 0.9],
            ]
        )
    ring = lines_mesh(f"{name}_ring", np.asarray(ring_segments), 0.32, ring_material)

    normal = arrow_object(
        f"{name}_normal",
        np.array([0.0, 0.0, 0.8]),
        np.array([0.0, 0.0, 1.0]),
        radius * 1.35,
        radius * 0.055,
        get_emission_material(f"{name}_arrow", PENTOS_BLUE, strength=2.4),
    )
    ring.parent = disc
    normal.parent = disc
    label = text_object(
        f"{name}_label",
        "A 0° B 0°",
        (center.x, center.y - radius - 9.0, center.z + 1.0),
        5.2,
        TEXT_COLOR,
    )
    for part in (disc, label):
        hide_state(part, False)
    return {"pivot": disc, "label": label, "center": center}


def rotate_dial(dial: dict, a_degrees: float, b_degrees: float) -> None:
    pivot = dial["pivot"]
    pivot.matrix_basis = Matrix.Translation(dial["center"]) @ machine_rotation_matrix(
        a_degrees, b_degrees
    )
    dial["label"].data.body = f"A {a_degrees:.0f}° B {b_degrees:.0f}°"
    face_camera(dial["label"], bpy.context.scene.camera.location)


def interp_dial(data: dict, t: float) -> tuple[float, float]:
    dial_time = data["dial_time"]
    if len(dial_time) < 2:
        return 0.0, 0.0
    a = float(np.interp(t, dial_time, data["dial_a"]))
    b = float(np.interp(t, dial_time, data["dial_b"]))
    return a, b


def pen_position(data: dict, t: float) -> np.ndarray | None:
    seg_times = data["seg_times"]
    points = data["points"]
    kinds = data["kinds"]
    extrusion = np.flatnonzero(kinds == 2)
    if len(extrusion) == 0:
        return None
    index = int(np.searchsorted(seg_times[extrusion], t))
    index = min(max(index - 1, 0), len(extrusion) - 1)
    segment = points[extrusion[index]]
    return np.asarray(segment.mean(axis=0))


# ---------------------------------------------------------------------------
# Scene kinds
# ---------------------------------------------------------------------------


def build_machine_scene(cache_dir: Path) -> dict:
    machine = json.loads((cache_dir / "machine.json").read_text())
    scale = machine["scale_mm_per_urdf_unit"]
    links = {}
    for link in machine["links"]:
        rgba = link.get("rgba") or [0.55, 0.58, 0.62, 1.0]
        color = (rgba[0], rgba[1], rgba[2])
        material = get_material(
            f"link_{link['name']}",
            color,
            roughness=0.42,
            metallic=0.25,
        )
        triangles = read_binary_stl(cache_dir / "urdf_meshes" / link["mesh"])
        triangles = triangles * scale
        obj = mesh_from_triangles(f"link_{link['name']}", triangles, material)
        links[link["name"]] = {
            "object": obj,
            "material": material,
            "dim_material": get_material("link_dim", DIM_GRAY, roughness=0.7),
        }

    joints = []
    for joint in machine["joints"]:
        child = links[joint["child"]]["object"]
        parent = links[joint["parent"]]["object"]
        child.parent = parent
        child.matrix_parent_inverse = Matrix.Identity(4)
        origin = Matrix.Translation(Vector(joint["origin_xyz_m"]) * scale) @ (
            Matrix.Rotation(joint["origin_rpy_rad"][2], 4, "Z")
            @ Matrix.Rotation(joint["origin_rpy_rad"][1], 4, "Y")
            @ Matrix.Rotation(joint["origin_rpy_rad"][0], 4, "X")
        )
        joints.append(
            {
                "name": joint["name"],
                "type": joint["type"],
                "axis": Vector(joint["axis"]).normalized(),
                "child": child,
                "origin": origin,
            }
        )

    mapping = machine["machine_axis_mapping"]
    del mapping  # arrows show machine-positive directions; mapping is provenance
    arrow_directions = {
        "X": Vector((-1.0, 0.0, 0.0)),
        "Y": Vector((0.0, -1.0, 0.0)),
        "Z": Vector((0.0, 0.0, 1.0)),
        "A": Vector((0.0, 1.0, 0.0)),
        "B": Vector((0.0, 0.0, -1.0)),
    }
    return {
        "kind": "machine",
        "links": links,
        "joints": joints,
        "axis_directions": arrow_directions,
        "machine_arrow_material": get_emission_material(
            "machine_arrow", PENTOS_ORANGE, strength=2.2
        ),
    }


def build_tube_scene(cache_dir: Path) -> dict:
    scene_json = json.loads((cache_dir / "scene.json").read_text())
    plate = scene_json["plate"]
    build_plate(plate["size_mm"], plate["center"])

    blue = get_material("pento_blue", PENTOS_BLUE, roughness=0.38)
    model = load_mesh_object(cache_dir, scene_json["model_stl"], "model", blue)

    chunks = []
    for record in scene_json["chunks"]:
        obj = load_mesh_object(
            cache_dir, record["mesh"], f"chunk_{record['index']}", blue
        )
        chunks.append({"object": obj, "record": record})

    cut_plane = None
    if scene_json["cut_planes"]:
        plane_data = scene_json["cut_planes"][0]
        rotation = plane_rotation(plane_data["wxyz"])
        size = 58.0
        quad = np.array(
            [
                [[-size, -size, 0], [size, -size, 0], [size, size, 0]],
                [[-size, -size, 0], [size, size, 0], [-size, size, 0]],
            ],
            dtype=np.float64,
        )
        plane_material = get_material(
            "cut_plane", PENTOS_ORANGE, roughness=0.5, alpha=0.16
        )
        cut_plane = mesh_from_triangles("cut_plane", quad, plane_material)
        cut_plane.matrix_basis = (
            Matrix.Translation(Vector(plane_data["position"])) @ rotation
        )

    npz = np.load(cache_dir / scene_json["toolpath"]["npz"])
    points = npz["points"].astype(np.float64)
    ends = points[:, 1, :]
    extrusion_ends = ends[npz["kinds"] == 2]
    margin = np.array([9.0, 9.0, 10.0])
    clip_bounds = (
        (
            extrusion_ends.min(axis=0) - margin,
            extrusion_ends.max(axis=0) + margin,
        )
        if len(extrusion_ends)
        else None
    )

    dial_total = float(npz["dial_time"][-1]) if len(npz["dial_time"]) else 0.0
    data = {
        "points": points,
        "kinds": npz["kinds"],
        "seg_times": npz["seg_times"],
        "dial_time": npz["dial_time"],
        "dial_a": npz["dial_a"],
        "dial_b": npz["dial_b"],
    }
    bright_buckets, bucket_starts = build_bucket_paths(
        points,
        npz["kinds"],
        npz["seg_times"],
        dial_total,
        "tube_bright",
        PENTOS_BLUE,
        (0.55, 0.58, 0.62),
        clip_bounds,
    )
    dim_material_extrusion = get_emission_material(
        "tube_dim_extrusion", PENTOS_BLUE, strength=0.28
    )
    dim_buckets = []
    if clip_bounds is not None:
        low, high = clip_bounds
        inside = np.all((ends >= low) & (ends <= high), axis=1)
        dim_extrusion = points[inside & (npz["kinds"] == 2)]
        if len(dim_extrusion):
            dim_buckets.append(
                lines_mesh(
                    "tube_dim", dim_extrusion, PATH_RADIUS, dim_material_extrusion
                )
            )

    transitions = json.loads(
        (cache_dir / scene_json["toolpath"]["transitions"]).read_text()
    )
    transition_visuals = []
    for transition in transitions:
        lift_points = transition.get("lift_points")
        if lift_points is None:
            continue
        lift_material = get_emission_material("lift_dash", PENTOS_ORANGE, strength=2.4)
        dashes = dashed_line(
            "lift_dashes",
            np.asarray(lift_points[0]),
            np.asarray(lift_points[1]),
            1.6,
            1.1,
            0.34,
            lift_material,
        )
        marker_material = get_emission_material(
            "continuation_marker", OVERHANG_RED, strength=2.6
        )
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=24,
            ring_count=16,
            radius=1.5,
            location=transition["continuation"],
        )
        marker = bpy.context.active_object
        marker.name = "continuation_marker"
        marker.data.materials.append(marker_material)
        transition_visuals.append(
            {
                "dashes": dashes,
                "marker": marker,
                "time": transition["time_s"],
                "seg_index": transition["seg_index"],
            }
        )

    model_top = float(model.dimensions.z + model.location.z)
    dial = make_dial("dial", Vector((45.0, 45.0, model_top + 26.0)), 13.0)
    dial_objects = [dial["pivot"], dial["label"]]
    for child in dial["pivot"].children:
        dial_objects.append(child)

    pen_material = get_emission_material("pen", PENTOS_ORANGE, strength=3.0)
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=20, ring_count=12, radius=1.5, location=(45, 45, -10)
    )
    pen = bpy.context.active_object
    pen.name = "pen"
    pen.data.materials.append(pen_material)

    return {
        "kind": "tube",
        "scene": scene_json,
        "model": model,
        "chunks": chunks,
        "cut_plane": cut_plane,
        "plane_material": cut_plane.data.materials[0] if cut_plane else None,
        "bright_buckets": bright_buckets,
        "bucket_starts": bucket_starts,
        "dim_buckets": dim_buckets,
        "dim_material_extrusion": dim_material_extrusion,
        "transitions": transition_visuals,
        "dial": dial,
        "dial_objects": dial_objects,
        "pen": pen,
        "data": data,
        "dial_total": dial_total,
    }


def build_nonplanar_scene(cache_dir: Path) -> dict:
    scene_json = json.loads((cache_dir / "scene.json").read_text())
    plate = scene_json["plate"]
    build_plate(plate["size_mm"], plate["center"])

    volume = np.load(cache_dir / scene_json["volume_npz"])
    original = volume["original_vertices"].astype(np.float64)
    deformed = volume["deformed_vertices"].astype(np.float64)
    faces = volume["boundary_faces"].astype(np.int32)
    scalars = volume["scalar_values"].astype(np.float64)

    def make_surface(name: str, material: bpy.types.Material) -> bpy.types.Object:
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(len(original))
        mesh.vertices.foreach_set("co", original.ravel())
        mesh.loops.add(faces.size)
        mesh.loops.foreach_set("vertex_index", faces.ravel())
        mesh.polygons.add(len(faces))
        mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3))
        mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3))
        mesh.update()
        mesh.validate()
        mesh.materials.append(material)
        object_ref = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(object_ref)
        object_ref.shape_key_add(name="Basis")
        shape = object_ref.shape_key_add(name="deformed", from_mix=False)
        shape.data.foreach_set("co", deformed.ravel())
        return object_ref

    plain_material = get_material("pento_blue", PENTOS_BLUE, roughness=0.38)
    morph_object = make_surface("morph_model", plain_material)

    # Scalar field coloring via a point-domain float attribute and ColorRamp.
    scalar_min, scalar_max = float(scalars.min()), float(scalars.max())
    span = max(scalar_max - scalar_min, 1e-6)
    normalized = ((scalars - scalar_min) / span).clip(0.0, 1.0)
    field_material = bpy.data.materials.new("scalar_field")
    field_material.use_nodes = True
    nodes = field_material.node_tree.nodes
    links = field_material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "scalar"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*PENTOS_BLUE, 1.0)
    ramp.color_ramp.elements[1].color = (*PENTOS_ORANGE, 1.0)
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0.92, 0.94, 0.97, 1.0)
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.35
    principled.inputs["Emission Strength"].default_value = 0.35
    links.new(attribute.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(ramp.outputs["Color"], principled.inputs["Emission Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    set_blend_mode(field_material, "BLEND")
    field_object = make_surface("field_model", field_material)
    scalar_attribute = field_object.data.attributes.new(
        name="scalar", type="FLOAT", domain="POINT"
    )
    scalar_attribute.data.foreach_set("value", normalized)
    field_object.data.update()

    guides = []
    guide_material = get_material("guide", PENTOS_ORANGE, roughness=0.5, alpha=0.14)
    for record in scene_json["guides"]:
        obj = load_mesh_object(
            cache_dir, record["mesh"], f"guide_{record['guide_id']}", guide_material
        )
        guides.append(obj)

    planar_npz = np.load(cache_dir / scene_json["planar_toolpath"]["npz"])
    mapped_npz = np.load(cache_dir / scene_json["mapped_toolpath"]["npz"])
    mapped_total = (
        float(mapped_npz["dial_time"][-1])
        if len(mapped_npz["dial_time"])
        else float(mapped_npz["seg_times"][-1])
    )

    planar_data = {
        "points": planar_npz["points"].astype(np.float64),
        "kinds": planar_npz["kinds"],
        "seg_times": planar_npz["seg_times"],
        "dial_time": planar_npz["dial_time"],
        "dial_a": planar_npz["dial_a"],
        "dial_b": planar_npz["dial_b"],
    }
    mapped_data = {
        "points": mapped_npz["points"].astype(np.float64),
        "kinds": mapped_npz["kinds"],
        "seg_times": mapped_npz["seg_times"],
        "layers": mapped_npz["layers"],
        "dial_time": mapped_npz["dial_time"],
        "dial_a": mapped_npz["dial_a"],
        "dial_b": mapped_npz["dial_b"],
    }
    path_margin = np.array([9.0, 9.0, 10.0])
    planar_ends = planar_data["points"][:, 1, :]
    planar_clip = (
        (
            planar_ends[planar_data["kinds"] == 2].min(axis=0) - path_margin,
            planar_ends[planar_data["kinds"] == 2].max(axis=0) + path_margin,
        )
        if np.any(planar_data["kinds"] == 2)
        else None
    )
    mapped_ends = mapped_data["points"][:, 1, :]
    mapped_clip = (
        (
            mapped_ends[mapped_data["kinds"] == 2].min(axis=0) - path_margin,
            mapped_ends[mapped_data["kinds"] == 2].max(axis=0) + path_margin,
        )
        if np.any(mapped_data["kinds"] == 2)
        else None
    )
    planar_buckets, planar_starts = build_bucket_paths(
        planar_data["points"],
        planar_data["kinds"],
        planar_data["seg_times"],
        float(planar_data["seg_times"][-1]),
        "planar_paths",
        PENTOS_BLUE,
        (0.55, 0.58, 0.62),
        planar_clip,
    )
    mapped_buckets, mapped_starts = build_bucket_paths(
        mapped_data["points"],
        mapped_data["kinds"],
        mapped_data["seg_times"],
        float(mapped_data["seg_times"][-1]),
        "mapped_paths",
        PENTOS_BLUE,
        (0.55, 0.58, 0.62),
        mapped_clip,
    )
    base_buckets = []
    base_material = get_emission_material("base_paths", PENTOS_ORANGE, strength=2.4)
    base_mask = mapped_data["kinds"] == 2
    for layer in range(scene_json["base_layers_planar"]):
        layer_points = mapped_data["points"][
            (mapped_data["layers"] == layer) & base_mask
        ]
        if len(layer_points):
            base_buckets.append(
                lines_mesh(
                    f"base_layer_{layer}", layer_points, PATH_RADIUS, base_material
                )
            )

    model_top = float(original[:, 2].max())
    dial = make_dial("dial", Vector((45.0, 45.0, model_top + 26.0)), 13.0)
    dial_objects = [dial["pivot"], dial["label"]]
    for child in dial["pivot"].children:
        dial_objects.append(child)

    return {
        "kind": "nonplanar",
        "scene": scene_json,
        "morph_object": morph_object,
        "field_object": field_object,
        "field_material": field_material,
        "guide_material": guide_material,
        "guides": guides,
        "planar_buckets": planar_buckets,
        "planar_starts": planar_starts,
        "mapped_buckets": mapped_buckets,
        "mapped_starts": mapped_starts,
        "base_buckets": base_buckets,
        "dial": dial,
        "dial_objects": dial_objects,
        "planar_data": planar_data,
        "mapped_data": mapped_data,
        "dial_total": mapped_total,
    }


# ---------------------------------------------------------------------------
# Shot state application
# ---------------------------------------------------------------------------


def joint_value_at(action: dict, joint: str, t: float) -> float:
    value = 0.0
    for move in action["moves"]:
        window = move["joints"].get(joint)
        if window is None:
            continue
        t0, t1 = move["t0"], move["t1"]
        if t < t0:
            continue
        amount = ease_window(t, t0, t1) if t <= t1 else 1.0
        value = window[0] + (window[1] - window[0]) * amount
    return value


def apply_machine_action(state: dict, shot: dict, u: float) -> None:
    action = shot["action"]
    duration = (
        shot["last_frame"] - shot["first_frame"] + 1
    ) / bpy.context.scene.render.fps
    t = u * duration

    for joint in state["joints"]:
        value = joint_value_at(action, joint["name"], t)
        motion = Matrix.Identity(4)
        if joint["type"] == "prismatic":
            motion = Matrix.Translation(joint["axis"] * value * 1000.0)
        elif joint["type"] == "revolute":
            motion = Matrix.Rotation(math.radians(value), 4, joint["axis"])
        joint["child"].matrix_basis = joint["origin"] @ motion

    highlight = set(action["highlight"])
    for name, link in state["links"].items():
        material = (
            link["dim_material"]
            if highlight and name not in highlight
            else link["material"]
        )
        link["object"].data.materials.clear()
        link["object"].data.materials.append(material)

    # Labels and arrows persist between shots; hide any not in this action.
    active_labels = set()
    for label in action["labels"]:
        name = f"label_{label['text']}"
        active_labels.add(name)
        object_ref = state.get(name)
        if object_ref is None:
            object_ref = text_object(
                name,
                label["text"],
                tuple(label["at"]),
                label["size"],
                TEXT_COLOR,
            )
            state[name] = object_ref
        visible = label["t0"] <= t <= label["t1"]
        hide_state(object_ref, visible)
        if visible:
            face_camera(object_ref, bpy.context.scene.camera.location)
    for name, object_ref in state.items():
        if name.startswith("label_") and name not in active_labels:
            hide_state(object_ref, False)

    active_arrows = set()
    for arrow in action["arrows"]:
        name = f"arrow_{arrow['axis']}_{arrow['at'][0]:.0f}_{arrow['at'][1]:.0f}"
        active_arrows.add(name)
        object_ref = state.get(name)
        if object_ref is None:
            if arrow["axis"] in ("A", "B"):
                object_ref = rotation_arrow_object(
                    name,
                    np.asarray(arrow["at"]),
                    np.asarray(state["axis_directions"][arrow["axis"]]),
                    arrow["length"],
                    state["machine_arrow_material"],
                )
            else:
                object_ref = arrow_object(
                    name,
                    np.asarray(arrow["at"]),
                    np.asarray(state["axis_directions"][arrow["axis"]]),
                    arrow["length"],
                    arrow["length"] * 0.028,
                    state["machine_arrow_material"],
                )
            state[name] = object_ref
        visible = arrow["t0"] <= t <= arrow["t1"]
        hide_state(object_ref, visible)
    for name, object_ref in state.items():
        if name.startswith("arrow_") and name not in active_arrows:
            hide_state(object_ref, False)


def apply_plane_cut(state: dict, shot: dict, u: float) -> None:
    action = shot["action"]
    chunks = state["chunks"]
    plane_normal = None
    if state["cut_plane"] is not None:
        plane_record = state["scene"]["cut_planes"][0]
        plane_normal = np.asarray(
            plane_rotation(plane_record["wxyz"]).to_3x3() @ Vector((0.0, 0.0, 1.0))
        )

    if action["phase"] == "plane":
        hide_state(state["model"], True)
        for chunk in chunks:
            hide_state(chunk["object"], False)
        for chunk in chunks:
            chunk["object"].matrix_basis = Matrix.Identity(4)
        if state["cut_plane"] is not None:
            hide_state(state["cut_plane"], True)
            grow = 0.55 + 0.45 * ease_window(u, 0.05, 0.6)
            state["cut_plane"].scale = (grow, grow, 1.0)
            state["plane_material"].node_tree.nodes["Principled BSDF"].inputs[
                "Alpha"
            ].default_value = 0.16 * ease_window(u, 0.05, 0.6)
        for dial_object in state["dial_objects"]:
            hide_state(dial_object, False)
        rotate_dial(state["dial"], 0.0, 0.0)
        hide_state(state["pen"], False)
        state["pen"].location = (45.0, 45.0, -10.0)
        hide_reveal_extras(state)
    else:
        hide_state(state["model"], False)
        explode = 16.0 * ease_window(u, 0.05, 0.85)
        for index, chunk in enumerate(chunks):
            hide_state(chunk["object"], True)
            if plane_normal is not None:
                sign = 1.0 if chunk["record"]["print_up_normal"] is not None else -1.0
                offset = Vector(plane_normal) * (explode * sign)
                chunk["object"].matrix_basis = Matrix.Translation(offset)
        if state["cut_plane"] is not None:
            hide_state(state["cut_plane"], True)
            state["plane_material"].node_tree.nodes["Principled BSDF"].inputs[
                "Alpha"
            ].default_value = 0.12
        for dial_object in state["dial_objects"]:
            hide_state(dial_object, False)
        rotate_dial(state["dial"], 0.0, 0.0)
        hide_state(state["pen"], False)
        state["pen"].location = (45.0, 45.0, -10.0)
    hide_reveal_extras(state)


def flatten_matrix(record: dict, plate: dict, s: float) -> Matrix:
    center = Vector(plate["rotation_center_local_mm"])
    rotation = machine_rotation_matrix(record["a_degrees"] * s, record["b_degrees"] * s)
    translation = Vector((0.0, 0.0, -record["z_offset"] * s)) + Vector(
        (record["flat_xy_offset"][0] * s, record["flat_xy_offset"][1] * s, 0.0)
    )
    # v' = R(s) @ (v - c) + c + t  ==  R @ v + (c - R @ c + t)
    return (
        Matrix.Translation(center + translation - rotation.to_3x3() @ center) @ rotation
    )


def apply_chunk_flatten(state: dict, shot: dict, u: float) -> None:
    action = shot["action"]
    plate = state["scene"]["plate"]
    hide_state(state["model"], False)
    for index, chunk in enumerate(state["chunks"]):
        chunk_object = chunk["object"]
        hide_state(chunk_object, True)
        if index == action["chunk"]:
            hold = action["flat_hold"]
            s = ease_window(u, 0.05, 1.0 - hold)
            chunk_object.matrix_basis = flatten_matrix(chunk["record"], plate, s)
        else:
            chunk_object.matrix_basis = Matrix.Identity(4)
    if state["cut_plane"] is not None:
        hide_state(state["cut_plane"], u < 0.1)
        state["plane_material"].node_tree.nodes["Principled BSDF"].inputs[
            "Alpha"
        ].default_value = 0.14 * (1.0 - ease_window(u, 0.05, 0.5))
    for dial_object in state["dial_objects"]:
        hide_state(dial_object, False)
    record = state["chunks"][action["chunk"]]["record"]
    rotate_dial(state["dial"], record["a_degrees"], record["b_degrees"])
    hide_state(state["pen"], False)
    state["pen"].location = (45.0, 45.0, -10.0)
    hide_reveal_extras(state)


def apply_chunks_flat_overview(state: dict, shot: dict, u: float) -> None:
    plate = state["scene"]["plate"]
    hide_state(state["model"], False)
    for index, chunk in enumerate(state["chunks"]):
        chunk_object = chunk["object"]
        hide_state(chunk_object, True)
        if chunk["record"]["print_up_normal"] is not None:
            chunk_object.matrix_basis = flatten_matrix(chunk["record"], plate, 1.0)
        else:
            chunk_object.matrix_basis = Matrix.Identity(4)
    if state["cut_plane"] is not None:
        hide_state(state["cut_plane"], False)
        state["plane_material"].node_tree.nodes["Principled BSDF"].inputs[
            "Alpha"
        ].default_value = 0.08
    for dial_object in state["dial_objects"]:
        hide_state(dial_object, False)
    for chunk in state["chunks"]:
        if chunk["record"]["print_up_normal"] is not None:
            record = chunk["record"]
            rotate_dial(state["dial"], record["a_degrees"], record["b_degrees"])
    hide_state(state["pen"], False)
    state["pen"].location = (45.0, 45.0, -10.0)
    hide_reveal_extras(state)


def reveal_buckets(
    buckets: list, starts: np.ndarray, t: float, visible: bool = True
) -> None:
    for index, bucket in enumerate(buckets):
        shown = visible and t >= starts[index] - 1e-6
        hide_state(bucket, shown)
        for child in bucket.children:
            hide_state(child, shown)


def hide_reveal_extras(state: dict) -> None:
    """Transition markers and the pen only belong to preview_reveal shots."""
    for visual in state["transitions"]:
        hide_state(visual["dashes"], False)
        hide_state(visual["marker"], False)
    hide_state(state["pen"], False)
    for bucket in state["bright_buckets"]:
        hide_state(bucket, False)
        for child in bucket.children:
            hide_state(child, False)
    for bucket in state["dim_buckets"]:
        hide_state(bucket, False)


def apply_preview_reveal(state: dict, shot: dict, u: float) -> None:
    action = shot["action"]
    data = state["data"]
    dial_total = state["dial_total"]
    window = action.get("window") or [0.0, 1.0]
    t = dial_total * (window[0] + (window[1] - window[0]) * u)

    is_transition = action.get("transition")
    for bucket in state["dim_buckets"]:
        hide_state(bucket, bool(is_transition))
    reveal_buckets(state["bright_buckets"], state["bucket_starts"], t)

    transition_time = None
    for visual in state["transitions"]:
        in_window = is_transition and abs(visual["time"] - t) < dial_total * 0.2
        transition_time = visual["time"] if in_window else transition_time
        show = (
            is_transition
            and t >= visual["time"] - dial_total * 0.06
            and t <= visual["time"] + dial_total * 0.14
        )
        hide_state(visual["dashes"], show)
        hide_state(visual["marker"], show)

    show_dial = action.get("dial", False)
    for dial_object in state["dial_objects"]:
        hide_state(dial_object, show_dial)
    if show_dial:
        a, b = interp_dial(data, t)
        rotate_dial(state["dial"], a, b)
    else:
        rotate_dial(state["dial"], 0.0, 0.0)

    hide_state(state["pen"], show_dial)
    if show_dial:
        position = pen_position(data, t)
        if position is not None:
            state["pen"].location = position.tolist()


def apply_guide_field(state: dict, shot: dict, u: float) -> None:
    action = shot["action"]
    phase = action["phase"]
    morph_object = state["morph_object"]
    field_object = state["field_object"]
    solid = morph_object.data.materials[0].node_tree.nodes.get("Principled BSDF")
    if solid is not None:
        solid.inputs["Alpha"].default_value = 1.0

    guides_visible = phase in ("guides", "field")
    guide_alpha = 0.14 if phase == "field" else 0.14 * ease_window(u, 0.0, 0.5)
    for guide in state["guides"]:
        hide_state(guide, guides_visible)
    if guides_visible:
        nodes = state["guide_material"].node_tree.nodes
        principled = nodes.get("Principled BSDF")
        if principled is not None:
            principled.inputs["Alpha"].default_value = max(guide_alpha, 0.02)

    morph_value = 0.0
    field_alpha = 0.0
    if phase == "guides":
        morph_value = 0.0
    elif phase == "field":
        field_alpha = ease_window(u, 0.05, 0.7)
    elif phase == "morph":
        field_alpha = 1.0
        morph_value = ease_window(u, 0.1, 0.95)
    elif phase == "flat":
        morph_value = 1.0

    shape_key = morph_object.data.shape_keys.key_blocks["deformed"]
    shape_key.value = morph_value
    field_object.data.shape_keys.key_blocks["deformed"].value = morph_value

    if phase == "guides":
        hide_state(field_object, False)
        hide_state(morph_object, True)
    elif phase in ("field", "morph", "flat"):
        nodes = state["field_material"].node_tree.nodes
        principled = nodes.get("Principled BSDF")
        if principled is not None:
            principled.inputs["Alpha"].default_value = field_alpha
        hide_state(field_object, field_alpha < 0.01)
        hide_state(morph_object, field_alpha > 0.01)
    else:
        hide_state(field_object, False)
        hide_state(morph_object, False)

    for bucket in (
        state["planar_buckets"] + state["mapped_buckets"] + state["base_buckets"]
    ):
        hide_state(bucket, False)
        for child in bucket.children:
            hide_state(child, False)


def apply_mapped_reveal(state: dict, shot: dict, u: float) -> None:
    action = shot["action"]
    phase = action["phase"]
    mapped_data = state["mapped_data"]
    morph_object = state["morph_object"]
    field_object = state["field_object"]

    for guide in state["guides"]:
        hide_state(guide, True)
    ghost = morph_object.data.materials[0].node_tree.nodes.get("Principled BSDF")
    if ghost is not None:
        ghost.inputs["Alpha"].default_value = 0.32

    morph_value = 1.0 if phase == "planar" else 0.0
    if phase == "map":
        morph_value = 1.0 - ease_window(u, 0.05, 0.75)
    shape_key = morph_object.data.shape_keys.key_blocks["deformed"]
    shape_key.value = morph_value
    field_object.data.shape_keys.key_blocks["deformed"].value = morph_value
    hide_state(field_object, False)
    hide_state(morph_object, True)

    planar_total = float(state["planar_data"]["seg_times"][-1])
    mapped_end = float(mapped_data["seg_times"][-1])
    if phase == "planar":
        reveal_buckets(
            state["planar_buckets"],
            state["planar_starts"],
            planar_total * ease_window(u, 0.05, 0.9),
        )
    elif phase == "map":
        reveal_buckets(
            state["planar_buckets"], state["planar_starts"], -1.0, visible=False
        )
        reveal_buckets(
            state["mapped_buckets"],
            state["mapped_starts"],
            mapped_end * ease_window(u, 0.05, 0.9),
        )
    elif phase == "curved":
        reveal_buckets(
            state["planar_buckets"], state["planar_starts"], -1.0, visible=False
        )
        reveal_buckets(
            state["mapped_buckets"],
            state["mapped_starts"],
            mapped_end * ease_window(u, 0.0, 0.9),
        )
    elif phase == "base":
        reveal_buckets(
            state["planar_buckets"], state["planar_starts"], -1.0, visible=False
        )
        reveal_buckets(
            state["mapped_buckets"], state["mapped_starts"], -1.0, visible=False
        )
        for bucket in state["base_buckets"]:
            hide_state(bucket, True)

    show_dial = bool(action.get("dial")) and phase in ("curved",)
    for dial_object in state["dial_objects"]:
        hide_state(dial_object, show_dial)
    if show_dial:
        t_dial = float(mapped_data["seg_times"][-1]) * ease_window(u, 0.0, 0.9)
        a, b = interp_dial(mapped_data, t_dial)
        rotate_dial(state["dial"], a, b)
    else:
        rotate_dial(state["dial"], 0.0, 0.0)


ACTION_HANDLERS = {
    "machine_pose": apply_machine_action,
    "plane_cut": apply_plane_cut,
    "chunk_flatten": apply_chunk_flatten,
    "chunks_flat_overview": apply_chunks_flat_overview,
    "preview_reveal": apply_preview_reveal,
    "guide_field": apply_guide_field,
    "mapped_reveal": apply_mapped_reveal,
}


def apply_camera(scene_state: dict, shot: dict, u: float) -> None:
    camera = bpy.context.scene.camera
    orbit = shot["camera"]["orbit"]
    azimuth = orbit["start_deg"] + (orbit["end_deg"] - orbit["start_deg"]) * u
    place_camera(
        camera,
        orbit["center"],
        orbit["radius"],
        orbit["height"],
        azimuth,
        orbit["ortho_scale"],
    )


def main() -> int:
    argv = sys.argv
    config_path = argv[argv.index("--") + 1]
    config = json.loads(Path(config_path).read_text())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_render(config)
    setup_world()
    setup_lighting()

    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("camera")
    camera_data.clip_start = 1.0
    camera_data.clip_end = 20000.0
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    cache_dir = Path(config["cache_dir"])
    if config["scene"] == "machine":
        state = build_machine_scene(cache_dir)
    elif config["scene"] == "tube":
        state = build_tube_scene(cache_dir)
    else:
        state = build_nonplanar_scene(cache_dir)

    debug_prefixes = tuple(config.get("debug_hide_prefixes", []))

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "current"
    scene.render.filepath = str(scratch)

    wanted = set(config["frames"])
    for shot in config["shots"]:
        shot_frames = [
            frame
            for frame in range(shot["first_frame"], shot["last_frame"] + 1)
            if frame in wanted
        ]
        if not shot_frames:
            continue
        handler = ACTION_HANDLERS[shot["action"]["type"]]
        total = shot["last_frame"] - shot["first_frame"] + 1
        for frame in shot_frames:
            u = (frame - shot["first_frame"]) / max(total - 1, 1)
            apply_camera(state, shot, u)
            handler(state, shot, u)
            if debug_prefixes:
                for obj in bpy.context.scene.objects:
                    if obj.name.startswith(debug_prefixes):
                        obj.hide_render = True
            scene.frame_set(frame)
            final = out_dir / f"frame_{frame:04d}.png"
            if final.exists():
                continue
            bpy.ops.render.render(write_still=True)
            produced = scratch.with_name(scratch.name + ".png")
            if not produced.exists():
                raise RuntimeError(f"Blender did not write {produced}")
            os.replace(produced, final)
            print(f"rendered {final.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("RENDER FAILED", file=sys.stderr)
        raise SystemExit(1)
