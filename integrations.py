import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests


def send_to_moonraker(
    gcode: Path,
    server: str | None = None,
    api_key: str | None = None,
    start_print: bool = False,
) -> dict[str, Any]:
    load_dotenv()

    server = server or os.getenv("MOONRAKER_SERVER")
    api_key = api_key or os.getenv("MOONRAKER_API_KEY")

    if not server:
        raise ValueError("Set MOONRAKER_SERVER in .env or pass server=...")

    headers = {"X-Api-Key": api_key} if api_key else {}

    with gcode.open("rb") as file:
        response = requests.post(
            f"{server}/server/files/upload",
            files={"file": file},
            data = {
                "print": str(start_print).lower(),
            },
            headers=headers,
        )

    return response.json()
