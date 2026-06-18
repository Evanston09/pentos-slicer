import base64
from pathlib import Path

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


def configure_theme(server: "viser.ViserServer") -> None:
    server.gui.configure_theme(
        titlebar_content=titlebar_config(),
        brand_color=PENTOS_BLUE,
        dark_mode=True,
    )
