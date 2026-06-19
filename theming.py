import base64
from pathlib import Path

import numpy as np
import viser
from machine import BUILD_PLATE_CENTER, BUILD_PLATE_SIZE
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

BUILD_PLATE_COLOR = (45, 45, 45)
PENTOS_BLUE = (47, 153, 238)
PENTOS_ORANGE = (255, 130, 0)


def logo_to_data_url() -> str:
    base64_str = base64.b64encode(Path("assets/logo.png").read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def titlebar_config() -> TitlebarConfig:
    logo_url = logo_to_data_url()
    return TitlebarConfig(
        buttons=(
            TitlebarButton(
                text="Made by Evan Kim",
                icon=None,
                href="https://evankim.me",
            ),
        ),
        image=TitlebarImage(
            image_url_light=logo_url,
            image_url_dark=logo_url,
            image_alt="Pentos Logo",
            href=None,
        ),
    )


def configure_theme(server: viser.ViserServer) -> None:
    server.gui.configure_theme(
        titlebar_content=titlebar_config(),
        brand_color=PENTOS_BLUE,
        dark_mode=True,
    )


def add_build_plate_scene(server: viser.ViserServer) -> None:
    server.scene.add_grid(
        "/shared/grid",
        width=BUILD_PLATE_SIZE,
        height=BUILD_PLATE_SIZE,
        cell_size=5.0,
        section_size=10.0,
        position=BUILD_PLATE_CENTER,
    )

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
        "/shared/build_plate/surface",
        vertices=vertices,
        faces=faces,
        color=BUILD_PLATE_COLOR,
        opacity=0.18,
        side="double",
    )

    server.scene.add_line_segments(
        "/shared/build_plate/outline",
        points=np.array(
            [
                [[0.0, 0.0, 0.0], [size, 0.0, 0.0]],
                [[size, 0.0, 0.0], [size, size, 0.0]],
                [[size, size, 0.0], [0.0, size, 0.0]],
                [[0.0, size, 0.0], [0.0, 0.0, 0.0]],
            ],
        ),
        colors=np.array(PENTOS_ORANGE),
        line_width=2.0,
    )
