import pyvista as pv

mesh = pv.read("models/t_shape.stl")

plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgray", opacity=0.45, show_edges=True)


def plane_changed(normal, origin):
    print("origin:", origin)
    print("normal:", normal)


plotter.add_plane_widget(
    callback=plane_changed,
    normal=(0, 0, 1),
    origin=mesh.center,
    assign_to_axis=None,
    implicit=True,
)

plotter.show()
