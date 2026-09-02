"""Blender-side helpers for the Pentos visualization renderer.

This module runs inside Blender's Python (no project imports). Everything is
deterministic: meshes are built from numpy arrays with low-level bpy.data
APIs instead of operators.
"""

from __future__ import annotations

import struct
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

# Pentos palette (views/theming.py)
PENTOS_BLUE = (47 / 255, 153 / 255, 238 / 255)
PENTOS_ORANGE = (255 / 255, 130 / 255, 0 / 255)
OVERHANG_RED = (239 / 255, 68 / 255, 68 / 255)
BUILD_PLATE_COLOR = (45 / 255, 45 / 255, 45 / 255)
MACHINE_GRAY = (138 / 255, 143 / 255, 152 / 255)
DIM_GRAY = (0.10, 0.10, 0.11)
BACKGROUND = (0.035, 0.043, 0.055)
TEXT_COLOR = (0.92, 0.94, 0.97)


def set_blend_mode(material: bpy.types.Material, mode: str) -> None:
    """EEVEE needs both the legacy blend method and the render method set."""
    try:
        material.blend_method = mode
    except (AttributeError, TypeError):
        pass
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED" if mode == "BLEND" else "DITHERED"


def get_material(
    name: str,
    color: tuple[float, float, float],
    roughness: float = 0.45,
    metallic: float = 0.0,
    alpha: float = 1.0,
    emission: float = 0.0,
) -> bpy.types.Material:
    """Fetch or create a Principled material by name."""
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Alpha"].default_value = alpha
    principled.inputs["Emission Color"].default_value = (*color, 1.0)
    principled.inputs["Emission Strength"].default_value = emission
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    if alpha < 1.0:
        set_blend_mode(material, "BLEND")
    return material


def get_emission_material(
    name: str,
    color: tuple[float, float, float],
    strength: float = 2.0,
    alpha: float = 1.0,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength
    out_socket = emission.outputs["Emission"]
    if alpha < 1.0:
        set_blend_mode(material, "BLEND")
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = alpha
        links.new(transparent.outputs["BSDF"], mix.inputs[1])
        links.new(emission.outputs["Emission"], mix.inputs[2])
        links.new(mix.outputs["Shader"], output.inputs["Surface"])
    else:
        links.new(out_socket, output.inputs["Surface"])
    return material


def read_binary_stl(path: Path) -> np.ndarray:
    """Read an STL into (n, 3, 3) triangle vertices (binary or ASCII)."""
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        return _read_ascii_stl(data.decode("utf-8", errors="replace"))
    count = struct.unpack("<I", data[80:84])[0]
    records = np.frombuffer(data, dtype=np.uint8, count=count * 50, offset=84)
    records = records.reshape(count, 50)
    return records[:, 12:48].copy().view("<f4").reshape(count, 3, 3)


def _read_ascii_stl(text: str) -> np.ndarray:
    vertices = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 3, 3)


def mesh_from_triangles(
    name: str,
    triangles: np.ndarray,
    material: bpy.types.Material | None = None,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    """Create a mesh object from (n, 3, 3) triangles, welding shared vertices."""
    triangles = np.asarray(triangles, dtype=np.float64)
    flat = triangles.reshape(-1, 3)
    welded, faces = np.unique(flat, axis=0, return_inverse=True)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(welded))
    mesh.vertices.foreach_set("co", welded.ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.ravel())
    face_count = faces.size // 3
    mesh.polygons.add(face_count)
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3))
    mesh.polygons.foreach_set("loop_total", np.full(face_count, 3))
    mesh.update()
    mesh.validate()
    object_ref = bpy.data.objects.new(name, mesh)
    object_ref.location = location
    bpy.context.scene.collection.objects.link(object_ref)
    if material is not None:
        object_ref.data.materials.append(material)
    return object_ref


def lines_mesh(
    name: str,
    segments: np.ndarray,
    radius: float,
    material: bpy.types.Material,
    material_index: int = 0,
) -> bpy.types.Object:
    """Build one mesh of thin square-profile tubes along each segment."""
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if len(segments) == 0:
        return mesh_from_triangles(name, np.zeros((0, 3, 3)), material)
    starts = segments[:, 0]
    ends = segments[:, 1]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    lengths[lengths == 0.0] = 1.0
    directions = directions / lengths[:, None]
    references = np.tile(np.array([0.0, 0.0, 1.0]), (len(directions), 1))
    references[np.abs(directions[:, 2]) >= 0.9] = (1.0, 0.0, 0.0)
    side_a = np.cross(directions, references)
    side_a /= np.linalg.norm(side_a, axis=1, keepdims=True)
    side_b = np.cross(directions, side_a)
    side_a = side_a * radius
    side_b = side_b * radius

    corners = []
    for sign_a, sign_b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        corners.append(
            (
                starts + side_a * sign_a + side_b * sign_b,
                ends + side_a * sign_a + side_b * sign_b,
            )
        )
    # Group per segment: start ring slots 0-3 then end ring slots 4-7.
    start_rings = np.stack([corners[slot][0] for slot in range(4)], axis=1)
    end_rings = np.stack([corners[slot][1] for slot in range(4)], axis=1)
    vertices = np.concatenate([start_rings, end_rings], axis=1).reshape(-1, 3)

    faces = []
    for index in range(len(segments)):
        base = index * 8
        for slot in range(4):
            nxt = (slot + 1) % 4
            faces.append((base + slot, base + nxt, base + 4 + nxt, base + 4 + slot))
        faces.append((base + 0, base + 3, base + 2, base + 1))
        faces.append((base + 4, base + 5, base + 6, base + 7))

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.ravel())
    loop_total = np.full(len(faces), 4)
    loop_start = np.arange(0, len(faces) * 4, 4)
    mesh.loops.add(len(faces) * 4)
    mesh.polygons.add(len(faces))
    flat_faces = np.asarray(faces, dtype=np.int32).ravel()
    mesh.loops.foreach_set("vertex_index", flat_faces)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)
    if material_index:
        mesh.polygons.foreach_set("material_index", np.full(len(faces), material_index))
    mesh.update()
    mesh.validate()
    object_ref = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(object_ref)
    object_ref.data.materials.append(material)
    return object_ref


def dashed_line(
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    dash_length: float,
    gap: float,
    radius: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    total = float(np.linalg.norm(direction))
    if total <= 0.0:
        return lines_mesh(name, np.zeros((0, 2, 3)), radius, material)
    unit = direction / total
    segments = []
    distance = 0.0
    while distance < total:
        seg_end = min(distance + dash_length, total)
        segments.append([start + unit * distance, start + unit * seg_end])
        distance = seg_end + gap
    return lines_mesh(name, np.asarray(segments), radius, material)


def text_object(
    name: str,
    body: str,
    location: tuple[float, float, float],
    size: float,
    color: tuple[float, float, float],
    align: str = "CENTER",
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name=name, type="FONT")
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.align_y = "CENTER"
    curve.extrude = size * 0.04
    object_ref = bpy.data.objects.new(name, curve)
    object_ref.location = location
    object_ref.color = (*color, 1.0)
    bpy.context.scene.collection.objects.link(object_ref)
    object_ref.data.materials.append(
        get_emission_material(f"text_{color[0]:.2f}", color, strength=1.4)
    )
    return object_ref


def arrow_object(
    name: str,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Straight arrow: shaft cylinder plus cone tip, oriented along direction."""
    origin = Vector(np.asarray(origin, dtype=np.float64))
    direction = Vector(np.asarray(direction, dtype=np.float64))
    if direction.length == 0.0:
        direction = Vector((0.0, 0.0, 1.0))
    direction.normalize()
    shaft_length = length * 0.72
    tip_length = length * 0.28

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=radius,
        depth=shaft_length,
        location=(0, 0, shaft_length / 2.0),
    )
    shaft = bpy.context.active_object
    shaft.name = f"{name}_shaft"

    bpy.ops.mesh.primitive_cone_add(
        vertices=24,
        radius1=radius * 2.4,
        radius2=0.0,
        depth=tip_length,
        location=(0, 0, shaft_length + tip_length / 2.0),
    )
    tip = bpy.context.active_object
    tip.name = f"{name}_tip"

    bpy.ops.object.select_all(action="DESELECT")
    shaft.select_set(True)
    tip.select_set(True)
    bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    arrow = bpy.context.active_object
    arrow.name = name

    rotation = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
    arrow.matrix_basis = Matrix.Translation(origin) @ rotation
    arrow.data.materials.append(material)
    return arrow


def rotation_arrow_object(
    name: str,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """Curved right-hand-rule arrow around the rotation axis at center."""
    import math

    axis = Vector(np.asarray(axis, dtype=np.float64)).normalized()
    center = Vector(np.asarray(center, dtype=np.float64))
    reference = Vector((1.0, 0.0, 0.0))
    if abs(reference.dot(axis)) > 0.9:
        reference = Vector((0.0, 0.0, 1.0))
    start = (reference - axis * reference.dot(axis)).normalized()

    segments = []
    steps = 28
    sweep = math.radians(150.0)
    begin = math.radians(-115.0)
    for index in range(steps):
        t0 = begin + sweep * index / steps
        t1 = begin + sweep * (index + 1) / steps
        p0 = center + (Matrix.Rotation(t0, 3, axis) @ start) * radius
        p1 = center + (Matrix.Rotation(t1, 3, axis) @ start) * radius
        segments.append([list(p0), list(p1)])
    arc = lines_mesh(name, np.asarray(segments), radius * 0.05, material)

    tip_angle = begin + sweep
    tip_end = center + (Matrix.Rotation(tip_angle, 3, axis) @ start) * radius
    tangent = Matrix.Rotation(tip_angle, 3, axis) @ axis.cross(start)
    bpy.ops.mesh.primitive_cone_add(
        vertices=20,
        radius1=radius * 0.14,
        radius2=0.0,
        depth=radius * 0.4,
        location=(0, 0, 0),
    )
    cone = bpy.context.active_object
    cone.name = f"{name}_cone"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    cone.matrix_basis = Matrix.Translation(tip_end) @ (
        tangent.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    )
    cone.parent = arc
    return arc


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_world() -> None:
    world = bpy.data.worlds.get("PentosWorld") or bpy.data.worlds.new("PentosWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*BACKGROUND, 1.0)
    background.inputs["Strength"].default_value = 1.6


def setup_lighting() -> None:
    scene = bpy.context.scene
    key = bpy.data.lights.new("key", type="SUN")
    key.energy = 3.4
    key.angle = 0.35
    key_object = bpy.data.objects.new("key", key)
    key_object.rotation_euler = Euler((np.radians(50), 0.0, np.radians(-35)), "XYZ")
    scene.collection.objects.link(key_object)

    fill = bpy.data.lights.new("fill", type="AREA")
    fill.energy = 1400.0
    fill.size = 6.0
    fill.color = (0.75, 0.82, 0.95)
    fill_object = bpy.data.objects.new("fill", fill)
    fill_object.location = (-5.0, -4.0, 3.0)
    fill_object.rotation_euler = Euler((np.radians(55), 0.0, np.radians(-120)), "XYZ")
    scene.collection.objects.link(fill_object)


def place_camera(
    camera: bpy.types.Object,
    center: list[float],
    radius: float,
    height: float,
    azimuth_deg: float,
    ortho_scale: float | None,
) -> None:
    azimuth = np.radians(azimuth_deg)
    location = Vector(center) + Vector(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), height)
    )
    camera.location = location
    look_at(camera, Vector(center))
    if ortho_scale is not None:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = ortho_scale
    else:
        camera.data.type = "PERSP"
        camera.data.lens = 50.0


def face_camera(object_ref: bpy.types.Object, camera_location: Vector) -> None:
    """Rotate a text object so its face (+Z) points at the camera."""
    direction = Vector(camera_location) - object_ref.location
    if direction.length > 0.0:
        object_ref.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()


def hide_state(object_ref: bpy.types.Object, visible: bool) -> None:
    object_ref.hide_viewport = not visible
    object_ref.hide_render = not visible


def set_text_body(object_ref: bpy.types.Object, body: str) -> None:
    object_ref.data.body = body


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def ease_window(u: float, t0: float, t1: float) -> float:
    if t1 <= t0:
        return 1.0 if u >= t1 else 0.0
    return smoothstep((u - t0) / (t1 - t0))
