import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


def load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(path)
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = tuple(mesh_or_scene.geometry.values())
        if not meshes:
            raise ValueError(f"No meshes found in {path}")
        return trimesh.util.concatenate(meshes)
    return mesh_or_scene


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def quat_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize(source)
    target = normalize(target)
    dot = float(np.dot(source, target))

    if dot < -0.999999:
        axis = normalize(np.cross(source, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-6:
            axis = normalize(np.cross(source, np.array([0.0, 1.0, 0.0])))
        return np.array([0.0, axis[0], axis[1], axis[2]])

    cross = np.cross(source, target)
    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
    return normalize(quat)


def quat_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def intersect_ray_mesh(
    mesh: trimesh.Trimesh, ray_origin: np.ndarray, ray_direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return nearest ray/triangle hit point and face normal.

    This avoids requiring trimesh's optional rtree/embree acceleration packages.
    It is fine for prototype placement on modest meshes.
    """
    vertices = mesh.vertices
    triangles = vertices[mesh.faces]
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0

    pvec = np.cross(ray_direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(det) > 1e-9
    if not np.any(valid):
        return None

    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = ray_origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", ray_direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det

    hits = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
    if not np.any(hits):
        return None

    hit_index = np.where(hits)[0][np.argmin(t[hits])]
    hit_point = ray_origin + t[hit_index] * ray_direction
    normal = normalize(np.cross(edge1[hit_index], edge2[hit_index]))
    return hit_point, normal


def plane_record(gizmo) -> dict[str, list[float]]:
    origin = np.asarray(gizmo.position, dtype=float)
    rotation = quat_to_matrix(np.asarray(gizmo.wxyz, dtype=float))
    normal = normalize(rotation @ np.array([0.0, 0.0, 1.0]))
    return {
        "origin": origin.round(6).tolist(),
        "normal": normal.round(6).tolist(),
        "wxyz": np.asarray(gizmo.wxyz, dtype=float).round(6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="models/t_shape.3mf")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output", default="planes.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    mesh = load_trimesh(model_path)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    extents = mesh.extents
    plane_size = float(np.linalg.norm(extents) * 0.75)
    gizmo_scale = float(np.linalg.norm(extents) * 0.08)
    center = np.asarray(mesh.centroid)

    server.scene.add_mesh_simple(
        "/model",
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        color=(170, 170, 170),
        opacity=0.45,
        wireframe=True,
        side="double",
    )

    half = plane_size / 2.0
    plane_vertices = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float32,
    )
    plane_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    plane_mesh = server.scene.add_mesh_simple(
        "/cut_plane",
        vertices=plane_vertices,
        faces=plane_faces,
        color=(255, 80, 80),
        opacity=0.28,
        side="double",
        position=center,
    )
    plane_frame = server.scene.add_frame(
        "/cut_plane_frame",
        axes_length=plane_size * 0.18,
        axes_radius=plane_size * 0.008,
        position=center,
    )
    gizmo = server.scene.add_transform_controls(
        "/cut_plane_gizmo",
        scale=gizmo_scale,
        position=center,
    )

    snap_to_face = server.gui.add_checkbox("Click aligns to face", True)
    output_text = server.gui.add_text("Output", str(output_path))
    current_text = server.gui.add_text("Current plane", "", multiline=True, disabled=True)
    save_button = server.gui.add_button("Save plane")
    align_x = server.gui.add_button("Normal +X")
    align_y = server.gui.add_button("Normal +Y")
    align_z = server.gui.add_button("Normal +Z")

    def sync_plane() -> None:
        plane_mesh.position = gizmo.position
        plane_mesh.wxyz = gizmo.wxyz
        plane_frame.position = gizmo.position
        plane_frame.wxyz = gizmo.wxyz
        current_text.value = json.dumps(plane_record(gizmo), indent=2)

    @gizmo.on_update
    def _(_) -> None:
        sync_plane()

    @align_x.on_click
    def _(_) -> None:
        gizmo.wxyz = quat_from_two_vectors(
            np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
        )
        sync_plane()

    @align_y.on_click
    def _(_) -> None:
        gizmo.wxyz = quat_from_two_vectors(
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        )
        sync_plane()

    @align_z.on_click
    def _(_) -> None:
        gizmo.wxyz = quat_from_two_vectors(
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
        )
        sync_plane()

    @save_button.on_click
    def _(_) -> None:
        path = Path(output_text.value)
        existing = []
        if path.exists():
            existing = json.loads(path.read_text())
        existing.append(plane_record(gizmo))
        path.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"Saved plane {len(existing)} to {path}")

    @server.scene.get_handle_by_name("/model").on_click
    def _(event) -> None:
        hit = intersect_ray_mesh(
            mesh,
            np.asarray(event.ray_origin, dtype=float),
            normalize(np.asarray(event.ray_direction, dtype=float)),
        )
        if hit is None:
            return

        point, normal = hit
        gizmo.position = point
        if snap_to_face.value:
            gizmo.wxyz = quat_from_two_vectors(np.array([0.0, 0.0, 1.0]), normal)
        sync_plane()

    sync_plane()
    print(f"Loaded {model_path}")
    print(f"Open http://localhost:{args.port}")
    print("Drag the gizmo, click the mesh to place the plane, then use Save plane.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
