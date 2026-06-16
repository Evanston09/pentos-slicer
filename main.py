import time
from pathlib import Path

import trimesh
import viser
from trimesh.visual.color import ColorVisuals

from plane_manager import PlaneManager
from slice_tools import Slicer

BLUE = [47, 153, 238, 255]


server = viser.ViserServer()
server.scene.add_grid(
    "/world/grid",
    width=100,
    height=100,
)

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


def show_mesh(path: Path) -> trimesh.Trimesh:
    server.scene.remove_by_name("/model")

    mesh = trimesh.load_mesh(path)
    mesh.visual = ColorVisuals(mesh, face_colors=[47, 153, 238, 140])
    server.scene.add_mesh_trimesh("/model", mesh)
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
def _(event):
    plane_manager.add_plane()


@slice_button.on_click
def _(event):
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
