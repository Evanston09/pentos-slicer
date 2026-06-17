import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from machine import BUILD_PLATE_CENTER, BUILD_PLATE_SIZE
from plane_manager import PlaneManager
from slice_tools import Slicer

BUILD_PLATE_COLOR = (45, 45, 45)
BUILD_PLATE_EDGE_COLOR = np.array([255, 130, 0])
MODEL_COLOR = (47, 153, 238)


def add_build_plate(server: viser.ViserServer) -> None:
    size = BUILD_PLATE_SIZE
    vertices = np.array(
        [
            [0.0, 0.0, -0.02],
            [size, 0.0, -0.02],
            [size, size, -0.02],
            [0.0, size, -0.02],
        ],
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    server.scene.add_mesh_simple(
        "/build_plate/surface",
        vertices=vertices,
        faces=faces,
        color=BUILD_PLATE_COLOR,
        opacity=0.18,
        side="double",
    )

    server.scene.add_line_segments(
        "/build_plate/outline",
        points=np.array(
            [
                [[0.0, 0.0, 0.0], [size, 0.0, 0.0]],
                [[size, 0.0, 0.0], [size, size, 0.0]],
                [[size, size, 0.0], [0.0, size, 0.0]],
                [[0.0, size, 0.0], [0.0, 0.0, 0.0]],
            ],
        ),
        colors=BUILD_PLATE_EDGE_COLOR,
        line_width=2.0,
    )


server = viser.ViserServer()
server.scene.add_grid(
    "/world/grid",
    width=BUILD_PLATE_SIZE,
    height=BUILD_PLATE_SIZE,
    cell_size=5.0,
    section_size=10.0,
    position=BUILD_PLATE_CENTER,
)
add_build_plate(server)

plane_manager = PlaneManager(server)
slicer = Slicer()
current_model: tuple[trimesh.Trimesh, str] | None = None

upload = server.gui.add_upload_button(
    "Upload Model",
    mime_type=".stl,.3mf,.obj,.ply",
)
status = server.gui.add_text("Status", "No model loaded", disabled=True)

add_plane_button = server.gui.add_button("Add Plane", icon=viser.Icon.SQUARES_DIAGONAL)
slice_button = server.gui.add_button("Slice", icon=viser.Icon.CLOUD_COMPUTING)


def normalize_mesh_units(mesh: trimesh.Trimesh) -> None:
    if mesh.units is None:
        mesh.units = "m" if float(mesh.extents.max()) < 1.0 else "mm"

    if mesh.units != "mm":
        mesh.convert_units("mm")


def show_mesh(path: Path) -> trimesh.Trimesh:
    server.scene.remove_by_name("/model")

    mesh = trimesh.load_mesh(path)
    normalize_mesh_units(mesh)

    lower, upper = mesh.bounds
    mesh_center_xy = (lower[:2] + upper[:2]) / 2.0
    mesh.apply_translation(
        [
            BUILD_PLATE_CENTER[0] - mesh_center_xy[0],
            BUILD_PLATE_CENTER[1] - mesh_center_xy[1],
            -lower[2],
        ]
    )

    server.scene.add_mesh_simple(
        "/model",
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        color=MODEL_COLOR,
        opacity=0.45,
        side="double",
    )
    return mesh


@upload.on_upload
def _(event):
    global current_model
    uploaded = event.target.value
    upload_dir = Path("uploaded_models")
    upload_dir.mkdir(exist_ok=True)

    path = upload_dir / uploaded.name
    path.write_bytes(uploaded.content)

    try:
        status.value = f"Loaded {uploaded.name}: "
        mesh = show_mesh(path)
        current_model = (mesh, path.stem)
        print(status.value)
    except Exception as exc:
        status.value = f"Failed to load {uploaded.name}: {exc}"
        print(status.value)


@add_plane_button.on_click
def _(_):
    plane_manager.add_plane()


@slice_button.on_click
def _(_):
    if current_model is None:
        status.value = "Load a model before slicing"
        return

    if not plane_manager.planes:
        status.value = "Add a plane before slicing"
        return

    planes = plane_manager.get_all_planes()
    mesh, source_name = current_model
    output_path = slicer.slice(mesh, planes, source_name)
    status.value = f"Saved merged G-code to {output_path}"


print(f"Open your browser to http://localhost:{server.get_port()}")
print("Press Ctrl+C to exit")

while True:
    time.sleep(10.0)
