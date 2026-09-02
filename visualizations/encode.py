"""Caption + encoding stage: caption PNGs, logo composition, H.264 output.

Captions are pre-rendered as transparent PNGs with Pillow and burned with
FFmpeg's overlay filter, which keeps encoding independent from the expensive
3D rerenders and works with any FFmpeg build (no libass/drawtext required).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_module_dir = Path(__file__).resolve().parent
for _path in (str(_module_dir.parent), str(_module_dir)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from storyboards import FPS, ORIENTATIONS, RESOLUTIONS, Video  # noqa: E402

LOGO_PATH = Path(__file__).resolve().parents[1] / "assets/logo.png"
LOGO_WINDOW_S = 1.6
CAPTION_PAD_S = 0.35
FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


class EncodeError(RuntimeError):
    pass


@dataclass
class CaptionOverlay:
    start_s: float
    end_s: float
    png: Path


def _load_font(size: int):
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _font_size(orientation: str) -> int:
    return 52 if orientation == "horizontal" else 46


def _caption_bottom_margin(orientation: str) -> int:
    return 72 if orientation == "horizontal" else 150


def _wrap_caption(text: str, font, max_width_px: float) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.getlength(candidate) <= max_width_px or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def render_caption_pngs(
    video: Video,
    orientation: str,
    resolution: tuple[int, int],
    out_dir: Path,
    fps: int = FPS,
) -> list[CaptionOverlay]:
    """Render one transparent caption PNG per shot with timed reveal windows."""
    from PIL import Image, ImageDraw

    if orientation not in ORIENTATIONS:
        raise EncodeError(f"Unknown orientation: {orientation}")
    width, height = resolution
    font = _load_font(_font_size(orientation))
    side_margin = 130 if orientation == "horizontal" else 90
    max_text_width = width - 2 * side_margin

    out_dir.mkdir(parents=True, exist_ok=True)
    overlays = []
    frame_cursor = 0
    for index, shot in enumerate(video.shots):
        start_s = frame_cursor / fps + CAPTION_PAD_S
        end_s = (frame_cursor + shot.frames) / fps - CAPTION_PAD_S
        frame_cursor += shot.frames

        wrapped = _wrap_caption(shot.caption, font, max_text_width)
        line_heights = []
        line_widths = []
        for line in wrapped.split("\n"):
            box = font.getbbox(line)
            line_widths.append(box[2] - box[0])
            line_heights.append(box[3] - box[1])
        line_spacing = 10
        block_w = max(line_widths)
        block_h = sum(line_heights) + line_spacing * (len(line_heights) - 1)
        pad = 18
        plate_w = block_w + 2 * pad
        plate_h = block_h + 2 * pad

        image = Image.new("RGBA", (plate_w, plate_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (0, 0, plate_w - 1, plate_h - 1), radius=16, fill=(12, 12, 16, 168)
        )
        y = pad
        for line, line_w, line_h in zip(wrapped.split("\n"), line_widths, line_heights):
            x = (plate_w - line_w) / 2
            draw.text(
                (x, y),
                line,
                font=font,
                fill=(248, 248, 248, 255),
                stroke_width=3,
                stroke_fill=(10, 10, 12, 220),
            )
            y += line_h + line_spacing

        png = out_dir / f"caption_{index:02d}.png"
        image.save(png)
        overlays.append(CaptionOverlay(start_s, end_s, png))
    del height
    return overlays


def _escape_filter_path(path: Path) -> str:
    text = str(path)
    text = text.replace("\\", "\\\\").replace(":", "\\:")
    return text.replace("'", "\\'")


def build_ffmpeg_command(
    frames_dir: Path,
    out_path: Path,
    captions: list[CaptionOverlay],
    fps: int,
    resolution: tuple[int, int],
    logo_path: Path = LOGO_PATH,
    logo_height: int | None = None,
) -> list[str]:
    """Compose frames + logo + caption overlays into a single FFmpeg command."""
    width, height = resolution
    if height > width:
        logo_height = logo_height or 150
        logo_margin = 64
    else:
        logo_height = logo_height or 110
        logo_margin = 44
    caption_margin = _caption_bottom_margin(
        "horizontal" if width >= height else "vertical"
    )

    frame_count = len(list(frames_dir.glob("frame_*.png")))
    total_s = frame_count / fps

    inputs = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-i",
        str(logo_path),
    ]
    for caption in captions:
        inputs += ["-i", str(caption.png)]

    logo_enable = f"lt(t,{LOGO_WINDOW_S})+gt(t,{total_s - LOGO_WINDOW_S:.3f})"
    parts = ["[0:v]format=rgba[base]"]
    current = "base"
    for index, caption in enumerate(captions):
        input_index = 2 + index
        output = f"cap{index}"
        parts.append(
            f"[{current}][{input_index}:v]overlay="
            f"(main_w-overlay_w)/2:main_h-overlay_h-{caption_margin}:"
            f"enable='between(t,{caption.start_s:.3f},{caption.end_s:.3f})'[{output}]"
        )
        current = output
    parts.append(
        f"[1:v]scale=-2:{logo_height}[logo];"
        f"[{current}][logo]overlay=(main_w-overlay_w)/2:{logo_margin}:"
        f"enable='{logo_enable}',format=yuv420p[v]"
    )
    filter_complex = ";".join(parts)

    return inputs + [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-r",
        str(fps),
        "-movflags",
        "+faststart",
        str(out_path),
    ]


def encode_video(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise EncodeError(f"ffmpeg failed (exit {process.returncode}), log: {log_path}")


def probe_video(path: Path, resolution: tuple[int, int], fps: int) -> dict:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(probe.stdout)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1:
        raise EncodeError(f"{path.name}: expected exactly one video stream")
    stream = streams[0]
    width, height = resolution
    issues = []
    if stream.get("codec_name") != "h264":
        issues.append(f"codec {stream.get('codec_name')}")
    if stream.get("pix_fmt") != "yuv420p":
        issues.append(f"pix_fmt {stream.get('pix_fmt')}")
    if int(stream.get("width", 0)) != width or int(stream.get("height", 0)) != height:
        issues.append(f"size {stream.get('width')}x{stream.get('height')}")
    rate = stream.get("avg_frame_rate", "0/1")
    if rate:
        numerator, _, denominator = rate.partition("/")
        try:
            measured = float(numerator) / float(denominator or 1)
        except ZeroDivisionError:
            measured = 0.0
        if abs(measured - fps) > 0.5:
            issues.append(f"fps {measured:.2f}")
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    expected_frames = int(stream.get("nb_frames", 0) or 0)
    if expected_frames <= 0:
        issues.append("no frames reported")
    if issues:
        raise EncodeError(f"{path.name}: " + ", ".join(issues))
    return {"duration_s": duration, "frames": expected_frames, "codec": "h264"}


def extract_poster(
    frames_dir: Path, frame_number: int, out_path: Path, width: int
) -> Path:
    """Downscale one rendered frame into a lightweight poster image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = frames_dir / f"frame_{frame_number:04d}.png"
    if not source.exists():
        raise EncodeError(f"Poster frame missing: {source}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-vf", f"scale={width}:-2", str(out_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return out_path


def orientation_resolution(orientation: str) -> tuple[int, int]:
    if orientation not in ORIENTATIONS:
        raise EncodeError(f"Unknown orientation: {orientation}")
    return RESOLUTIONS[orientation]
