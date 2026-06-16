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
current_mesh = None

upload = server.gui.add_upload_button(
    "Upload Model",
    mime_type=".stl,.3mf,.obj,.ply",
)
status = server.gui.add_text("Status", "No model loaded", disabled=True)

add_plane_button = server.gui.add_button("Add Plane", icon=viser.Icon.SQUARES_DIAGONAL)
slice_button = server.gui.add_button("Slice", icon=viser.Icon.CLOUD_COMPUTING)


def show_mesh(path: Path):
    global current_mesh
    server.scene.remove_by_name("/model")

    mesh = trimesh.load_mesh(path)
    mesh.visual = ColorVisuals(mesh, face_colors=[47, 153, 238, 255])
    server.scene.add_mesh_trimesh("/model", mesh)
    current_mesh = mesh


@upload.on_upload
def _(event):
    uploaded = event.target.value
    upload_dir = Path("uploaded_models")
    upload_dir.mkdir(exist_ok=True)

    path = upload_dir / uploaded.name
    path.write_bytes(uploaded.content)

    try:
        status.value = f"Loaded {uploaded.name}: "
        show_mesh(path)
        print(status.value)
    except Exception as exc:
        status.value = f"Failed to load {uploaded.name}: {exc}"
        print(status.value)


@add_plane_button.on_click
def _(event):
    plane_manager.add_plane()


@slice_button.on_click
def _(event):
    if current_mesh is None:
        status.value = "Load a model before slicing"
        return

    if not plane_manager.planes:
        status.value = "Add a plane before slicing"
        return

    planes = plane_manager.get_all_planes()
    paths = slicer.cut(current_mesh, planes)
    slicer.slice_parts(paths)
    status.value = f"Saved {len(paths)} sliced part(s) to {slicer.out_dir}"


print(f"Open your browser to http://localhost:{server.get_port()}")
print("Press Ctrl+C to exit")

while True:
    time.sleep(10.0)
