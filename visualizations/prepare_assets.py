"""Preparation stage: run the real Pentos workflows and export Blender data.

Everything here uses the production services (load_scene, decompose_mesh,
Slicer, solve_guide_deformation, map_gcode_to_original, parse_gcode_preview)
so the videos depict actual pipeline output. Outputs are validated and written
under a content-addressed cache directory.
"""

import json
import hashlib
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_module_dir = Path(__file__).resolve().parent
for _path in (str(_module_dir.parent), str(_module_dir)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import trimesh

from gcode_tools import GcodeCommand, iter_gcode_moves
from machine import MACHINE_XY_SIZE
from models import DEFAULT_MACHINE_CONFIG, AppState, guide_surface_mesh
from services.gcode_preview import parse_gcode_preview, transform_preview_point
from services.model_tools import transformed_model
from services.nonplanar_gcode import map_gcode_to_original
from services.project_io import load_scene
from services.slicing import Slicer, decompose_mesh
from services.volumetric_deformation import (
    TetrahedralVolume,
    solve_guide_deformation,
    tetrahedralize,
)
from trimesh import transformations as tf

import storyboards

REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = REPO_ROOT / "assets/Pentos_URDF/urdf/Pentos_URDF.urdf"
URDF_MESH_DIR = REPO_ROOT / "assets/Pentos_URDF/meshes"
SAMPLES = {
    "tube": REPO_ROOT / "samples/Tube.pentos",
    "nonplanar": REPO_ROOT / "samples/flat_base_curved_top.pentos",
}
CONFIG_INI = REPO_ROOT / "pentos_config.ini"

MAX_TOOLPATH_SEGMENTS = 12000
MAX_EXTRUSION_SEGMENTS = 60000
MAX_AUXILIARY_SEGMENTS = 2500
SETUP_KIND = 0
TRAVEL_KIND = 1
EXTRUSION_KIND = 2


class PreparationError(RuntimeError):
    pass


@dataclass
class SourceHash:
    path: Path
    sha256: str

    def as_dict(self) -> dict:
        return {"path": str(self.path.relative_to(REPO_ROOT)), "sha256": self.sha256}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            for child in sorted(path.rglob("*")):
                digest.update(str(child.relative_to(REPO_ROOT)).encode())
                digest.update(child.read_bytes() if child.is_file() else b"")
    digest.update(storyboards.PIPELINE_VERSION.encode())
    return digest.hexdigest()


def cache_key_tube() -> str:
    return _sha256_paths(
        [
            SAMPLES["tube"],
            CONFIG_INI,
            REPO_ROOT / "machine.py",
            REPO_ROOT / "models/machine_config.py",
            REPO_ROOT / "services/slicing.py",
            REPO_ROOT / "services/multiplanar_gcode.py",
        ]
    )


def cache_key_nonplanar() -> str:
    return _sha256_paths(
        [
            SAMPLES["nonplanar"],
            CONFIG_INI,
            REPO_ROOT / "machine.py",
            REPO_ROOT / "models/machine_config.py",
            REPO_ROOT / "services/slicing.py",
            REPO_ROOT / "services/nonplanar_gcode.py",
            REPO_ROOT / "services/volumetric_deformation.py",
        ]
    )


def cache_key_machine() -> str:
    return _sha256_paths([URDF_PATH, URDF_MESH_DIR])


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a binary STL into (tri_vertices (n,3,3), normals (n,3))."""
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        mesh = trimesh.load_mesh(path, file_type="stl")
        triangles = np.asarray(mesh.triangles, dtype=np.float32)
        return triangles, np.asarray(mesh.face_normals, dtype=np.float32)
    count = struct.unpack("<I", data[80:84])[0]
    records = np.frombuffer(data, dtype=np.uint8, count=count * 50, offset=84)
    records = records.reshape(count, 50)
    normals = records[:, :12].copy().view("<f4").reshape(count, 3)
    triangles = records[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
    return triangles, normals


def stl_bounds(path: Path) -> tuple[list[float], list[float]]:
    triangles, _ = read_binary_stl(path)
    flat = triangles.reshape(-1, 3)
    return flat.min(axis=0).tolist(), flat.max(axis=0).tolist()


def export_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    mesh.export(path, file_type="stl")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _finite(array: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(array)))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


# ---------------------------------------------------------------------------
# Toolpath extraction (shared by both workflows)
# ---------------------------------------------------------------------------


@dataclass
class ToolpathExtraction:
    points: np.ndarray  # (n, 2, 3) object-space mm
    kinds: np.ndarray  # (n,) int8
    layers: np.ndarray  # (n,) int16
    parts: np.ndarray  # (n,) int8
    seg_times: np.ndarray  # (n,) motion seconds at segment start
    dial_time: np.ndarray
    dial_a: np.ndarray
    dial_b: np.ndarray
    transitions: list[dict]
    has_rotary: bool
    types: np.ndarray | None = None  # (n,) int8 index into type_names
    type_names: list[str] | None = None


def extract_toolpath(
    text: str,
    machine_config=DEFAULT_MACHINE_CONFIG,
) -> ToolpathExtraction:
    """Walk G-code with the production parsers and build a timed segment stream.

    Segments are transformed to object space with transform_preview_point, the
    same convention the app preview uses. Transition blocks are recorded
    separately (lift segment, A/B target, continuation point). Extrusion
    feature types (perimeter, infill, ...) are tracked via ";TYPE:" comments.
    """
    lines = text.splitlines()
    preview = parse_gcode_preview(text, machine_config)

    points: list[list[float]] = []
    kinds: list[int] = []
    layers: list[int] = []
    parts: list[int] = []
    seg_times: list[float] = []
    types: list[int] = []
    type_names: list[str] = []
    type_ids: dict[str, int] = {}
    current_type = -1
    transitions: list[dict] = []

    layer_index = -1
    seen_layer = False
    in_transition = False
    part_index = 0
    transition: dict | None = None
    motion_time = 0.0
    has_rotary = False

    moves = {move.index: move for move in iter_gcode_moves(lines)}
    for index, line in enumerate(lines):
        parsed = GcodeCommand.parse(line)
        if parsed.comment == "LAYER_CHANGE":
            layer_index += 1
            seen_layer = True
            in_transition = False
            continue
        if parsed.comment is not None and parsed.comment.startswith("TYPE:"):
            type_name = parsed.comment[len("TYPE:") :].strip()
            if type_name not in type_ids:
                type_ids[type_name] = len(type_names)
                type_names.append(type_name)
            current_type = type_ids[type_name]
            continue
        if parsed.comment == "--- PENTOS A/B TRANSITION ---":
            in_transition = True
            transition = {
                "lift_points": None,
                "a_deg": None,
                "b_deg": None,
                "continuation": None,
                "seg_index": len(points),
                "time_s": motion_time,
            }
            continue
        if parsed.comment == "--- END PENTOS A/B TRANSITION ---":
            in_transition = False
            if transition is not None:
                transitions.append(transition)
                transition = None
            continue

        move = moves.get(index)
        if move is None:
            continue
        if "A" in move.parsed.args or "B" in move.parsed.args:
            has_rotary = True

        duration = 0.0
        if (
            move.start_xyz is not None
            and move.end_xyz is not None
            and move.feedrate is not None
            and move.feedrate > 0.0
        ):
            duration = float(
                np.linalg.norm(move.end_xyz - move.start_xyz) / move.feedrate * 60.0
            )
        seg_start_time = motion_time
        motion_time += duration

        if in_transition and transition is not None and move.has_xyz:
            if not move.is_absolute_xyz and move.end_xyz is not None:
                # Relative safe lift: record in object space at the old pose.
                start_obj = transform_preview_point(
                    move.start_xyz, *move.start_ab, machine_config
                )
                end_obj = transform_preview_point(
                    move.end_xyz, *move.start_ab, machine_config
                )
                transition["lift_points"] = [start_obj.tolist(), end_obj.tolist()]
            elif move.is_absolute_xyz and "A" not in move.parsed.args:
                end_obj = transform_preview_point(
                    move.end_xyz, *move.end_ab, machine_config
                )
                transition["continuation"] = end_obj.tolist()
            continue

        if in_transition and transition is not None and not move.has_xyz:
            transition["a_deg"] = float(move.end_ab[0])
            transition["b_deg"] = float(move.end_ab[1])
            continue

        if not (
            move.has_xyz and move.start_xyz is not None and move.end_xyz is not None
        ):
            continue

        if move.extrusion_delta > 0:
            kind = EXTRUSION_KIND
        elif seen_layer:
            kind = TRAVEL_KIND
        else:
            kind = SETUP_KIND

        start_obj = transform_preview_point(
            move.start_xyz, *move.start_ab, machine_config
        )
        end_obj = transform_preview_point(move.end_xyz, *move.end_ab, machine_config)
        points.append([start_obj.tolist(), end_obj.tolist()])
        kinds.append(kind)
        layers.append(layer_index)
        parts.append(part_index)
        seg_times.append(seg_start_time)
        types.append(current_type if kind == EXTRUSION_KIND else -1)

    return ToolpathExtraction(
        points=np.asarray(points, dtype=np.float32).reshape(-1, 2, 3),
        kinds=np.asarray(kinds, dtype=np.int8),
        layers=np.asarray(layers, dtype=np.int16),
        parts=np.asarray(parts, dtype=np.int8),
        seg_times=np.asarray(seg_times, dtype=np.float32),
        dial_time=preview.motion_time_seconds.astype(np.float32),
        dial_a=preview.a_degrees.astype(np.float32),
        dial_b=preview.b_degrees.astype(np.float32),
        transitions=transitions,
        has_rotary=has_rotary,
        types=np.asarray(types, dtype=np.int8),
        type_names=type_names,
    )


def decimate_extraction(extraction: ToolpathExtraction) -> ToolpathExtraction:
    """Thin travel/setup segments but keep extrusion paths intact when possible.

    Extrusion segments are the visual subject, so striding them (which would
    dash every line) is a last resort. Travel/setup lines are auxiliary and
    can be thinned freely while staying time-ordered.
    """
    total = len(extraction.points)
    if total <= MAX_TOOLPATH_SEGMENTS:
        return extraction

    keep_extrusion = np.flatnonzero(extraction.kinds == EXTRUSION_KIND)
    keep_other = np.flatnonzero(extraction.kinds != EXTRUSION_KIND)
    if len(keep_extrusion) > MAX_EXTRUSION_SEGMENTS:
        keep_extrusion = keep_extrusion[
            np.linspace(0, len(keep_extrusion) - 1, MAX_EXTRUSION_SEGMENTS).astype(
                np.int64
            )
        ]
    if len(keep_other) > MAX_AUXILIARY_SEGMENTS:
        keep_other = keep_other[
            np.linspace(0, len(keep_other) - 1, MAX_AUXILIARY_SEGMENTS).astype(np.int64)
        ]
    indices = np.sort(np.concatenate((keep_extrusion, keep_other)))

    seg_position = np.searchsorted(
        indices, [t["seg_index"] for t in extraction.transitions]
    )
    transitions = []
    for transition, position in zip(extraction.transitions, seg_position):
        remapped = dict(transition)
        remapped["seg_index"] = int(min(position, len(indices) - 1))
        transitions.append(remapped)

    return ToolpathExtraction(
        points=extraction.points[indices],
        kinds=extraction.kinds[indices],
        layers=extraction.layers[indices],
        parts=extraction.parts[indices],
        seg_times=extraction.seg_times[indices],
        dial_time=extraction.dial_time,
        dial_a=extraction.dial_a,
        dial_b=extraction.dial_b,
        transitions=transitions,
        has_rotary=extraction.has_rotary,
        types=extraction.types[indices] if extraction.types is not None else None,
        type_names=extraction.type_names,
    )


def keep_walls_only(
    extraction: ToolpathExtraction,
    wall_types: set[str],
    type_names: list[str],
) -> ToolpathExtraction:
    """Thin dense extrusion to perimeter walls so curved paths stay legible.

    Only whole features are dropped (perimeters vs infill), so every shown
    path remains a real, continuous extrusion and the stream stays
    time-ordered. Falls back to keeping everything when the G-code carries no
    ";TYPE:" metadata.
    """
    if extraction.types is None:
        return extraction
    wall_ids = {
        type_id
        for type_id, type_name in enumerate(type_names)
        if type_name in wall_types
    }
    if not wall_ids:
        return extraction
    keep = (extraction.kinds != EXTRUSION_KIND) | np.isin(
        extraction.types, list(wall_ids)
    )
    indices = np.flatnonzero(keep)
    return ToolpathExtraction(
        points=extraction.points[indices],
        kinds=extraction.kinds[indices],
        layers=extraction.layers[indices],
        parts=extraction.parts[indices],
        seg_times=extraction.seg_times[indices],
        dial_time=extraction.dial_time,
        dial_a=extraction.dial_a,
        dial_b=extraction.dial_b,
        transitions=extraction.transitions,
        has_rotary=extraction.has_rotary,
        types=extraction.types[indices],
    )


def save_toolpath_npz(path: Path, extraction: ToolpathExtraction) -> None:
    np.savez_compressed(
        path,
        points=extraction.points,
        kinds=extraction.kinds,
        layers=extraction.layers,
        parts=extraction.parts,
        seg_times=extraction.seg_times,
        dial_time=extraction.dial_time,
        dial_a=extraction.dial_a,
        dial_b=extraction.dial_b,
        types=extraction.types,
    )


# ---------------------------------------------------------------------------
# Machine (URDF) preparation
# ---------------------------------------------------------------------------


def _urdf_text(value: str) -> list[float]:
    return [float(item) for item in value.replace(",", " ").split()]


def prepare_machine(cache_root: Path) -> Path:
    out_dir = cache_root / f"machine_{cache_key_machine()[:8]}"
    mesh_dir = out_dir / "urdf_meshes"
    if (out_dir / "machine.json").exists():
        return out_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)

    root = ET.parse(URDF_PATH).getroot()
    links = []
    mesh_files = {}
    for link in root.findall("link"):
        visual = link.find("visual")
        mesh = visual.find("geometry/mesh") if visual is not None else None
        material = visual.find("material") if visual is not None else None
        rgba = None
        if material is not None:
            color = material.find("color")
            if color is not None:
                rgba = _urdf_text(color.attrib["rgba"])
        entry = {"name": link.attrib["name"], "rgba": rgba}
        if mesh is not None:
            filename = mesh.attrib["filename"]
            _require(
                filename.startswith("package://Pentos_URDF/meshes/"),
                f"Unexpected URDF mesh path: {filename}",
            )
            source = URDF_MESH_DIR / Path(filename).name
            _require(source.exists(), f"Missing URDF mesh: {source}")
            cache_name = source.name.strip()  # URDF has a leading-space link name
            shutil.copyfile(source, mesh_dir / cache_name)
            minimum, maximum = stl_bounds(mesh_dir / cache_name)
            entry["mesh"] = cache_name
            entry["bounds_mm"] = [minimum, maximum]
            mesh_files[entry["name"]] = cache_name
        links.append(entry)

    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib["type"],
                "parent": joint.find("parent").attrib["link"],
                "child": joint.find("child").attrib["link"],
                "origin_xyz_m": _urdf_text(origin.attrib.get("xyz", "0 0 0"))
                if origin is not None
                else [0.0] * 3,
                "origin_rpy_rad": _urdf_text(origin.attrib.get("rpy", "0 0 0"))
                if origin is not None
                else [0.0] * 3,
                "axis": _urdf_text(axis.attrib["xyz"])
                if axis is not None
                else [0.0, 0.0, 1.0],
                "limit": {
                    "lower": float(limit.attrib["lower"]),
                    "upper": float(limit.attrib["upper"]),
                }
                if limit is not None
                else None,
            }
        )

    machine_bounds = [
        [
            min(link["bounds_mm"][0][axis] for link in links if "bounds_mm" in link)
            for axis in range(3)
        ],
        [
            max(link["bounds_mm"][1][axis] for link in links if "bounds_mm" in link)
            for axis in range(3)
        ],
    ]

    payload = {
        "source_urdf": str(URDF_PATH.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(URDF_PATH),
        "scale_mm_per_urdf_unit": 1000.0,
        "root": "base_link",
        "links": links,
        "joints": joints,
        "machine_bounds_mm": machine_bounds,
        "machine_axis_mapping": {
            "A": {
                "joint": "a_joint",
                "sign": -1.0,
                "units": "degrees",
                "evidence": "machine.rotation_matrix positive A is a right-hand rotation about +Y; the URDF a_joint axis is (0,-1,0), so positive machine A equals negative URDF joint motion.",
            },
            "B": {
                "joint": "b_joint",
                "sign": -1.0,
                "units": "degrees",
                "evidence": "machine.rotation_matrix positive B carries +X toward -Y (tests/test_machine.py); the URDF b_joint axis is ~(0,0,-1) whose positive rotation carries +X toward +Y, so positive machine B equals negative URDF joint motion.",
            },
            "X": {
                "joint": "x_joint",
                "sign": None,
                "units": "millimeters",
                "evidence": "URDF actuator direction only; no G-code sign is claimed for Cartesian axes.",
            },
            "Y": {
                "joint": "y_joint",
                "sign": None,
                "units": "millimeters",
                "evidence": "URDF actuator direction only; no G-code sign is claimed for Cartesian axes.",
            },
            "Z": {
                "joint": "z_joint",
                "sign": None,
                "units": "millimeters",
                "evidence": "URDF actuator direction only; no G-code sign is claimed for Cartesian axes.",
            },
        },
        "machine_config": {
            "machine_plate_center_mm": list(
                DEFAULT_MACHINE_CONFIG.machine_plate_center_mm
            ),
            "rotation_center_machine_mm": list(
                DEFAULT_MACHINE_CONFIG.rotation_center_machine_mm
            ),
            "rotation_center_local_mm": list(
                DEFAULT_MACHINE_CONFIG.rotation_center_local_mm
            ),
            "build_volume_mm": list(DEFAULT_MACHINE_CONFIG.build_volume_mm),
        },
    }
    write_json(out_dir / "machine.json", payload)
    return out_dir


# ---------------------------------------------------------------------------
# Tube (multiplanar) preparation
# ---------------------------------------------------------------------------


def prepare_tube(cache_root: Path) -> Path:
    key = cache_key_tube()
    out_dir = cache_root / f"tube_{key[:8]}"
    scene_path = out_dir / "scene.json"
    if scene_path.exists():
        return out_dir
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    state = _load_state(SAMPLES["tube"])
    # Visualizations always need the real merged output; override the saved
    # debug flag in this isolated state without touching the sample file.
    state.debug_mode = False

    mesh, source_name = transformed_model(state)
    export_mesh(mesh, out_dir / "model.stl")

    slicer = Slicer(
        out_dir=work / "out",
        temp_dir=work / "temp",
        machine_config=state.machine_config,
    )
    chunks = slicer.export_stl_chunks(mesh, state.plane_snapshots, source_name)
    _require(chunks, "Tube decomposition produced no chunks")

    # Original-pose pieces for the separation/flatten animation.
    pieces = decompose_mesh(mesh, state.plane_snapshots, cap=True)
    _require(
        len(pieces) == len(chunks),
        "Chunk and piece counts diverge during Tube preparation",
    )
    for index, piece in enumerate(pieces):
        export_mesh(piece.mesh, out_dir / f"chunk_{index}.stl")
        shutil.copyfile(chunks[index].path, out_dir / f"chunk_{index}_flat.stl")

    gcode_path = slicer.slice(mesh, state.plane_snapshots, source_name)
    extraction = decimate_extraction(
        extract_toolpath(gcode_path.read_text(), state.machine_config)
    )
    save_toolpath_npz(out_dir / "toolpath.npz", extraction)
    write_json(out_dir / "transitions.json", extraction.transitions)

    chunk_records = []
    for index, chunk in enumerate(chunks):
        normal = (
            None if chunk.print_up_normal is None else chunk.print_up_normal.tolist()
        )
        chunk_records.append(
            {
                "index": index,
                "print_up_normal": normal,
                "a_degrees": chunk.a_degrees,
                "b_degrees": chunk.b_degrees,
                "z_offset": chunk.z_offset,
                "flat_xy_offset": chunk.flat_xy_offset,
                "mesh": f"chunk_{index}.stl",
                "flat_mesh": f"chunk_{index}_flat.stl",
            }
        )

    scene = {
        "sample": str(SAMPLES["tube"].relative_to(REPO_ROOT)),
        "sample_sha256": sha256_file(SAMPLES["tube"]),
        "cache_key": key,
        "source_name": source_name,
        "debug_mode_overridden": True,
        "model_stl": "model.stl",
        "cut_planes": [
            {"position": plane.position.tolist(), "wxyz": plane.wxyz.tolist()}
            for plane in state.plane_snapshots
        ],
        "chunks": chunk_records,
        "plate": {
            "size_mm": list(state.machine_config.build_volume_mm[:2]),
            "center": list(state.machine_config.build_plate_center),
            "rotation_center_local_mm": list(
                state.machine_config.rotation_center_local_mm
            ),
        },
        "toolpath": {
            "npz": "toolpath.npz",
            "transitions": "transitions.json",
            "n_segments": int(len(extraction.points)),
            "n_transitions": len(extraction.transitions),
            "has_rotary": extraction.has_rotary,
        },
    }
    write_json(scene_path, scene)
    _validate_tube_scene(out_dir, scene, extraction)
    return out_dir


def _validate_tube_scene(
    out_dir: Path, scene: dict, extraction: ToolpathExtraction
) -> None:
    _require(len(scene["chunks"]) >= 2, "Tube scene expected at least two chunks")
    model_min, model_max = stl_bounds(out_dir / scene["model_stl"])
    _require(
        _finite(np.asarray([model_min, model_max])), "Tube model bounds not finite"
    )
    for record in scene["chunks"]:
        _require(
            -180.0 - 1e-6 <= record["a_degrees"] <= 180.0 + 1e-6,
            f"Chunk {record['index']} A out of range: {record['a_degrees']}",
        )
        _require(
            -180.0 - 1e-6 <= record["b_degrees"] <= 180.0 + 1e-6,
            f"Chunk {record['index']} B out of range: {record['b_degrees']}",
        )
        flat_min, flat_max = stl_bounds(out_dir / record["flat_mesh"])
        _require(
            _finite(np.asarray([flat_min, flat_max])),
            f"Chunk {record['index']} flat mesh not finite",
        )
    _require(extraction.has_rotary, "Merged Tube G-code contains no A/B commands")
    _require(
        len(extraction.transitions) >= 1,
        "Merged Tube G-code contains no A/B transitions",
    )
    _require(len(extraction.points) > 0, "Tube toolpath has no segments")
    extrusion = int((extraction.kinds == EXTRUSION_KIND).sum())
    _require(extrusion > 0, "Tube toolpath has no extrusion segments")
    for transition in extraction.transitions:
        _require(
            transition["a_deg"] is not None and transition["b_deg"] is not None,
            "Transition is missing its A/B command",
        )
        _require(
            transition["lift_points"] is not None, "Transition is missing the safe lift"
        )


# ---------------------------------------------------------------------------
# Nonplanar preparation
# ---------------------------------------------------------------------------


def prepare_nonplanar(cache_root: Path) -> Path:
    key = cache_key_nonplanar()
    out_dir = cache_root / f"nonplanar_{key[:8]}"
    scene_path = out_dir / "scene.json"
    if scene_path.exists():
        return out_dir
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    state = _load_state(SAMPLES["nonplanar"])
    mesh, source_name = transformed_model(state)
    export_mesh(mesh, out_dir / "model.stl")

    volume = tetrahedralize(mesh)
    deformed_vertices = solve_guide_deformation(volume, state.guide_surfaces)
    np.savez_compressed(
        out_dir / "volume.npz",
        original_vertices=volume.original_vertices.astype(np.float32),
        deformed_vertices=deformed_vertices.astype(np.float32),
        tetrahedra=volume.tetrahedra.astype(np.int32),
        boundary_faces=volume.boundary_faces.astype(np.int32),
        scalar_values=volume.scalar_values.astype(np.float32),
    )

    deformed_surface = trimesh.Trimesh(
        vertices=volume.deformed_vertices.copy(),
        faces=volume.boundary_faces,
        process=False,
    )
    _require(deformed_surface.is_volume, "Deformed surface is not a valid volume")
    export_mesh(deformed_surface, out_dir / "deformed.stl")

    guides = []
    for guide in state.guide_surfaces:
        local_vertices, local_faces = _guide_mesh_transformed(guide)
        surface = trimesh.Trimesh(
            vertices=local_vertices, faces=local_faces, process=False
        )
        export_mesh(surface, out_dir / f"guide_{guide.guide_id}.stl")
        guides.append(
            {
                "guide_id": guide.guide_id,
                "position": guide.position.tolist(),
                "wxyz": guide.wxyz.tolist(),
                "bend_x": guide.bend_x,
                "bend_y": guide.bend_y,
                "mesh": f"guide_{guide.guide_id}.stl",
            }
        )

    slicer = Slicer(
        out_dir=work / "out",
        temp_dir=work / "temp",
        machine_config=state.machine_config,
    )
    planar_gcode = slicer.slice(deformed_surface, [], f"{source_name}_deformed")
    planar = decimate_extraction(
        extract_toolpath(planar_gcode.read_text(), state.machine_config)
    )
    save_toolpath_npz(out_dir / "planar_paths.npz", planar)

    mapped_text = map_gcode_to_original(
        planar_gcode.read_text(), volume, machine_config=state.machine_config
    )
    mapped_path = out_dir / "mapped.gcode"
    mapped_path.write_text(mapped_text)
    mapped = decimate_extraction(extract_toolpath(mapped_text, state.machine_config))
    mapped = keep_walls_only(
        mapped,
        wall_types={"Perimeter", "External perimeter"},
        type_names=mapped.type_names or [],
    )
    save_toolpath_npz(out_dir / "mapped_paths.npz", mapped)

    n_layers = int(mapped.layers.max()) + 1 if len(mapped.layers) else 0
    scene = {
        "sample": str(SAMPLES["nonplanar"].relative_to(REPO_ROOT)),
        "sample_sha256": sha256_file(SAMPLES["nonplanar"]),
        "cache_key": key,
        "source_name": source_name,
        "model_stl": "model.stl",
        "deformed_stl": "deformed.stl",
        "guides": guides,
        "volume_npz": "volume.npz",
        "planar_toolpath": {
            "npz": "planar_paths.npz",
            "n_segments": int(len(planar.points)),
            "space": "deformed_local",
        },
        "mapped_toolpath": {
            "npz": "mapped_paths.npz",
            "n_segments": int(len(mapped.points)),
            "space": "object_local",
            "gcode": "mapped.gcode",
        },
        "base_layers_planar": 2,
        "n_layers": n_layers,
        "plate": {
            "size_mm": list(state.machine_config.build_volume_mm[:2]),
            "center": list(state.machine_config.build_plate_center),
            "rotation_center_local_mm": list(
                state.machine_config.rotation_center_local_mm
            ),
        },
        "machine_xy_size": MACHINE_XY_SIZE,
    }
    write_json(scene_path, scene)
    _validate_nonplanar_scene(out_dir, scene, volume, planar, mapped)
    return out_dir


def _guide_mesh_transformed(guide) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = guide_surface_mesh(guide.bend_x, guide.bend_y)
    rotation = tf.quaternion_matrix(guide.wxyz)[:3, :3]
    return vertices @ rotation.T + guide.position, faces


def _validate_nonplanar_scene(
    out_dir: Path,
    scene: dict,
    volume: TetrahedralVolume,
    planar: ToolpathExtraction,
    mapped: ToolpathExtraction,
) -> None:
    data = np.load(out_dir / scene["volume_npz"])
    original = data["original_vertices"]
    deformed = data["deformed_vertices"]
    _require(_finite(original) and _finite(deformed), "Volume vertices not finite")
    _require(len(original) == len(deformed), "Deformation topology mismatch (vertices)")
    _require(
        np.array_equal(data["tetrahedra"], volume.tetrahedra), "Tetrahedra mismatch"
    )
    _require(len(data["scalar_values"]) == len(original), "Scalar field size mismatch")
    _require(len(scene["guides"]) >= 2, "Expected at least two guides")
    _require(
        len(planar.points) > 0 and int((planar.kinds == EXTRUSION_KIND).sum()) > 0,
        "Planar toolpath has no extrusion segments",
    )
    _require(
        len(mapped.points) > 0 and int((mapped.kinds == EXTRUSION_KIND).sum()) > 0,
        "Mapped toolpath has no extrusion segments",
    )
    _require(mapped.has_rotary, "Mapped G-code contains no A/B commands")
    ab = mapped.points  # object-space; A/B legality checked via dial arrays
    _require(_finite(ab), "Mapped toolpath not finite")
    base = mapped.layers[mapped.kinds == EXTRUSION_KIND]
    _require(bool(np.any(base >= 0)), "Mapped toolpath missing layer data")
    base_extrusion_ab = _ab_for_layers(
        mapped, max_layer=scene["base_layers_planar"] - 1
    )
    if base_extrusion_ab is not None:
        _require(
            bool(np.allclose(base_extrusion_ab, 0.0, atol=1e-4)),
            "Base layers of mapped G-code are not planar",
        )


def _ab_for_layers(extraction: ToolpathExtraction, max_layer: int) -> np.ndarray | None:
    """Return the A/B dial values while the given layers print, if present."""
    mask = extraction.layers <= max_layer
    if not np.any(mask):
        return None
    times = extraction.seg_times[mask]
    low, high = float(times.min()), float(times.max())
    dial_time = extraction.dial_time
    if len(dial_time) < 2:
        return None
    window = (dial_time >= low) & (dial_time <= high)
    return np.column_stack((extraction.dial_a[window], extraction.dial_b[window]))


def _load_state(sample: Path) -> AppState:
    return load_scene(sample.read_bytes())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def prepare_all(generated_root: Path, only: set[str] | None = None) -> dict:
    cache_root = generated_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    results = {}

    def wanted(name: str) -> bool:
        return only is None or name in only

    if wanted("machine"):
        started = datetime.now(UTC)
        out = prepare_machine(cache_root)
        results["machine"] = {
            "dir": str(out),
            "seconds": (datetime.now(UTC) - started).total_seconds(),
        }
    if wanted("tube"):
        started = datetime.now(UTC)
        out = prepare_tube(cache_root)
        results["tube"] = {
            "dir": str(out),
            "seconds": (datetime.now(UTC) - started).total_seconds(),
        }
    if wanted("nonplanar"):
        started = datetime.now(UTC)
        out = prepare_nonplanar(cache_root)
        results["nonplanar"] = {
            "dir": str(out),
            "seconds": (datetime.now(UTC) - started).total_seconds(),
        }

    manifest_path = cache_root / "manifest.json"
    manifest = {
        "pipeline_version": storyboards.PIPELINE_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "sources": {
            "tube": SourceHash(SAMPLES["tube"], sha256_file(SAMPLES["tube"])).as_dict(),
            "nonplanar": SourceHash(
                SAMPLES["nonplanar"], sha256_file(SAMPLES["nonplanar"])
            ).as_dict(),
            "pentos_config_ini": SourceHash(
                CONFIG_INI, sha256_file(CONFIG_INI)
            ).as_dict(),
            "urdf": SourceHash(URDF_PATH, sha256_file(URDF_PATH)).as_dict(),
        },
        "results": results,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Pentos visualization assets")
    parser.add_argument(
        "--generated-root", type=Path, default=REPO_ROOT / "visualizations/generated"
    )
    parser.add_argument("--only", nargs="*", choices=["machine", "tube", "nonplanar"])
    args = parser.parse_args()
    prepare_all(args.generated_root, set(args.only) if args.only else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
