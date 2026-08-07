import base64
from pathlib import Path

import numpy as np
import viser
from machine import BUILD_PLATE_CENTER, BUILD_PLATE_SIZE, BUILD_VOLUME_SIZE
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

BUILD_PLATE_COLOR = (45, 45, 45)
PENTOS_BLUE = (47, 153, 238)
PENTOS_ORANGE = (255, 130, 0)
OVERHANG_RED = (239, 68, 68)
BUILD_VOLUME_COLOR = (120, 120, 120)


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


def configure_theme(client: viser.ClientHandle) -> None:
    client.gui.configure_theme(
        titlebar_content=titlebar_config(),
        control_width="large",
        brand_color=PENTOS_BLUE,
        dark_mode=True,
    )


def add_build_plate_scene(client: viser.ClientHandle) -> None:
    client.scene.add_frame("/shared/build_plate", show_axes=False)
    client.scene.add_grid(
        "/shared/build_plate/grid",
        width=BUILD_PLATE_SIZE,
        height=BUILD_PLATE_SIZE,
        cell_size=5.0,
        section_size=10.0,
        position=np.asarray(BUILD_PLATE_CENTER),
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
    client.scene.add_mesh_simple(
        "/shared/build_plate/surface",
        vertices=vertices,
        faces=faces,
        color=BUILD_PLATE_COLOR,
        opacity=0.18,
        side="double",
    )

    client.scene.add_line_segments(
        "/shared/build_plate/outline",
        points=np.array(
            [
                [[0.0, 0.0, 0.0], [size, 0.0, 0.0]],
                [[size, 0.0, 0.0], [size, size, 0.0]],
                [[size, size, 0.0], [0.0, size, 0.0]],
                [[0.0, size, 0.0], [0.0, 0.0, 0.0]],
            ],
        ),
        colors=PENTOS_ORANGE,
        line_width=2.0,
    )

    width, depth, height = BUILD_VOLUME_SIZE
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, depth, 0.0],
            [0.0, depth, 0.0],
            [0.0, 0.0, height],
            [width, 0.0, height],
            [width, depth, height],
            [0.0, depth, height],
        ]
    )
    edge_indices = (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
    )
    client.scene.add_line_segments(
        "/shared/build_plate/build_volume/outline",
        points=np.array(
            [[corners[start], corners[end]] for start, end in edge_indices]
        ),
        colors=BUILD_VOLUME_COLOR,
        line_width=1.5,
    )
