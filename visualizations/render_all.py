"""Top-level orchestration for the Pentos visualization batch.

One command prepares cached scene data, renders resumable frame sequences with
Blender, encodes captioned MP4s with FFmpeg, and writes a run report. Failures
in one video never block unrelated videos.

Usage:
    uv run python -m visualizations.render_all            # full batch
    uv run python -m visualizations.render_all --dry-run  # plan only
    uv run python -m visualizations.render_all --prepare-only
    uv run python -m visualizations.render_all --videos tube_divide
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_module_dir = Path(__file__).resolve().parent
for _path in (str(_module_dir.parent), str(_module_dir)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import prepare_assets
import storyboards
from encode import (
    LOGO_PATH,
    EncodeError,
    build_ffmpeg_command,
    encode_video,
    extract_poster,
    probe_video,
    render_caption_pngs,
)
from storyboards import (
    FPS,
    ORIENTATIONS,
    RESOLUTIONS,
    VIDEO_IDS,
    Video,
    shot_frame_ranges,
    validate_storyboards,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender/render.py"
DEFAULT_OUT_ROOT = REPO_ROOT / "visualizations/generated"

REQUIRED_TOOLS = ("blender", "ffmpeg", "ffprobe", "prusa-slicer")

VIDEO_SCENES = storyboards.VIDEO_SCENES


class PipelineError(RuntimeError):
    pass


@dataclass
class StepResult:
    status: str  # "ok" | "failed" | "skipped"
    seconds: float = 0.0
    detail: str = ""
    outputs: list[str] = field(default_factory=list)


def check_tools() -> dict[str, str]:
    found = {}
    missing = []
    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        if path is None:
            missing.append(tool)
        else:
            found[tool] = path
    if missing:
        raise PipelineError(f"Missing required executables: {', '.join(missing)}")
    return found


def check_sources() -> None:
    for sample in prepare_assets.SAMPLES.values():
        if not sample.exists():
            raise PipelineError(f"Missing source sample: {sample}")
    if not prepare_assets.URDF_PATH.exists():
        raise PipelineError(f"Missing URDF: {prepare_assets.URDF_PATH}")
    if not LOGO_PATH.exists():
        raise PipelineError(f"Missing logo: {LOGO_PATH}")


def resolve_cache_dirs(generated_root: Path) -> dict[str, Path]:
    cache_root = generated_root / "cache"
    dirs = {}
    for prefix, key in (
        ("machine_", "machine"),
        ("tube_", "tube"),
        ("nonplanar_", "nonplanar"),
    ):
        matches = sorted(cache_root.glob(f"{prefix}*"))
        if not matches:
            raise PipelineError(
                f"No prepared data for {key} (expected {prefix}<hash> under {cache_root})"
            )
        dirs[key] = matches[-1]
    return dirs


def build_render_config(
    video: Video,
    orientation: str,
    cache_dir: Path,
    out_dir: Path,
    frames: list[int],
    scale: float = 1.0,
    fps: int = FPS,
) -> dict:
    resolution = RESOLUTIONS[orientation]
    scaled = (round(resolution[0] * scale), round(resolution[1] * scale))
    shots = []
    for shot, first_frame, last_frame in shot_frame_ranges(video):
        shots.append(
            {
                "id": shot.id,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "camera": shot.camera[orientation],
                "action": shot.action,
            }
        )
    return {
        "video_id": video.id,
        "orientation": orientation,
        "scene": VIDEO_SCENES[video.id],
        "resolution": scaled,
        "fps": fps,
        "frames": frames,
        "cache_dir": str(cache_dir),
        "out_dir": str(out_dir),
        "shots": shots,
    }


def existing_frames(frames_dir: Path) -> set[int]:
    frames = set()
    if frames_dir.exists():
        for path in frames_dir.glob("frame_*.png"):
            if path.is_file() and path.stat().st_size > 0:
                stem = path.stem.removeprefix("frame_")
                if stem.isdigit():
                    frames.add(int(stem))
    return frames


def frames_dir_for(out_root: Path, video_id: str, orientation: str) -> Path:
    return out_root / "frames" / video_id / orientation


def run_blender_render(
    config: dict,
    blender_path: str,
    log_path: Path,
    total_frames: int,
) -> StepResult:
    started = time.monotonic()
    config_path = Path(config["out_dir"]) / "render_config.json"
    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    command = [
        blender_path,
        "-b",
        "--factory-startup",
        "-P",
        str(BLENDER_SCRIPT),
        "--",
        str(config_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    seconds = time.monotonic() - started
    if process.returncode != 0:
        return StepResult(
            "failed", seconds, f"blender exit {process.returncode}, log: {log_path}"
        )
    rendered = existing_frames(Path(config["out_dir"]))
    if len(rendered) < total_frames:
        return StepResult(
            "failed",
            seconds,
            f"frames incomplete after render: {len(rendered)}/{total_frames}, log: {log_path}",
        )
    return StepResult(
        "ok", seconds, f"{len(rendered)} frames present", [str(Path(config["out_dir"]))]
    )


def run_encode(
    video: Video,
    orientation: str,
    frames_dir: Path,
    videos_dir: Path,
    logs_dir: Path,
    scale: float,
    fps: int = FPS,
) -> tuple[StepResult, Path | None]:
    started = time.monotonic()
    resolution = RESOLUTIONS[orientation]
    scaled = (round(resolution[0] * scale), round(resolution[1] * scale))
    captions = render_caption_pngs(video, orientation, scaled, videos_dir, fps)
    out_path = videos_dir / f"{video.id}_{orientation}.mp4"
    command = build_ffmpeg_command(frames_dir, out_path, captions, fps, scaled)
    log_path = logs_dir / f"encode_{video.id}_{orientation}.log"
    try:
        encode_video(command, log_path)
        info = probe_video(out_path, scaled, fps)
    except (EncodeError, subprocess.CalledProcessError) as exc:
        return StepResult("failed", time.monotonic() - started, str(exc)), None
    detail = f"{info['frames']} frames, {info['duration_s']:.2f}s, {info['codec']}"
    return StepResult(
        "ok", time.monotonic() - started, detail, [str(out_path)]
    ), out_path


def write_report(
    out_root: Path, results: dict, started_utc: str, dry_run: bool
) -> tuple[Path, Path]:
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_json = reports_dir / "report.json"
    report_md = reports_dir / "report.md"

    payload = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "videos": results,
    }
    report_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Pentos visualization run report",
        "",
        f"Started: {started_utc}",
        f"Finished: {payload['finished_utc']}",
        f"Dry run: {dry_run}",
        "",
    ]
    ok = sum(1 for entry in results.values() if entry.get("status") == "ok")
    failed = sum(1 for entry in results.values() if entry.get("status") == "failed")
    lines.append(f"**{ok} succeeded, {failed} failed, {len(results)} total**")
    lines.append("")
    lines.append("| Video | Orientation | Status | Seconds | Detail |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key, entry in results.items():
        lines.append(
            f"| {entry.get('video', '')} | {entry.get('orientation', '')} "
            f"| {entry.get('status', '')} | {entry.get('seconds', 0):.1f} "
            f"| {entry.get('detail', '').replace('|', '/')} |"
        )
    report_md.write_text("\n".join(lines) + "\n")
    return report_json, report_md


def poster_command(out_root: Path, video_id: str, orientation: str, frame: int) -> Path:
    videos = storyboards.videos()
    if video_id not in videos:
        raise PipelineError(f"Unknown video: {video_id}")
    frames_dir = frames_dir_for(out_root, video_id, orientation)
    posters_dir = REPO_ROOT / "visualizations/posters"
    width = 720 if orientation == "horizontal" else 480
    out_path = posters_dir / f"{video_id}_{orientation}_f{frame:04d}.png"
    return extract_poster(frames_dir, frame, out_path, width)


def run_batch(
    out_root: Path,
    video_ids: list[str],
    orientations: list[str],
    scale: float = 1.0,
    fps: int = FPS,
    dry_run: bool = False,
    prepare_only: bool = False,
    keep_frames: bool = False,
    force_reencode: bool = False,
) -> dict:
    started_utc = datetime.now(UTC).isoformat()
    check_tools()
    check_sources()
    videos = storyboards.videos()
    validate_storyboards(videos)

    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    tool_paths = {tool: shutil.which(tool) for tool in REQUIRED_TOOLS}
    print("Tools:", ", ".join(f"{k}={v}" for k, v in tool_paths.items()))

    if dry_run:
        print("Dry run: computing cache keys (no slicing will run)")
        cache_dirs = {
            "machine": Path(
                f"<cache>/machine_{prepare_assets.cache_key_machine()[:8]}"
            ),
            "tube": Path(f"<cache>/tube_{prepare_assets.cache_key_tube()[:8]}"),
            "nonplanar": Path(
                f"<cache>/nonplanar_{prepare_assets.cache_key_nonplanar()[:8]}"
            ),
        }
    else:
        print("Preparing scene data ...")
        prepare_assets.prepare_all(out_root)
        cache_dirs = resolve_cache_dirs(out_root)
        print("Cache:", {k: str(v) for k, v in cache_dirs.items()})
        if prepare_only:
            print("Preparation complete; stopping (--prepare-only)")
            return results

    for video_id in video_ids:
        video = videos[video_id]
        for orientation in orientations:
            key = f"{video_id}_{orientation}"
            frames_dir = frames_dir_for(out_root, video_id, orientation)
            total_frames = video.frames
            expected = set(range(1, total_frames + 1))
            missing = sorted(expected - existing_frames(frames_dir))
            mp4_path = out_root / "videos" / f"{video_id}_{orientation}.mp4"

            if not dry_run and mp4_path.exists() and not force_reencode:
                try:
                    info = probe_video(mp4_path, RESOLUTIONS[orientation], fps)
                    print(f"{key}: {mp4_path.name} already valid, skipping")
                    results[key] = {
                        "video": video_id,
                        "orientation": orientation,
                        "status": "ok",
                        "seconds": 0.0,
                        "detail": (
                            f"existing mp4 reused: {info['frames']} frames, "
                            f"{info['duration_s']:.2f}s"
                        ),
                        "outputs": [str(mp4_path)],
                    }
                    continue
                except EncodeError:
                    print(f"{key}: invalid existing mp4, re-encoding")

            try:
                config = build_render_config(
                    video,
                    orientation,
                    cache_dirs[VIDEO_SCENES[video_id]],
                    frames_dir,
                    missing,
                    scale,
                    fps,
                )
            except PipelineError as exc:
                results[key] = {
                    "video": video_id,
                    "orientation": orientation,
                    "status": "failed",
                    "seconds": 0.0,
                    "detail": str(exc),
                }
                continue

            if dry_run:
                results[key] = {
                    "video": video_id,
                    "orientation": orientation,
                    "status": "ok",
                    "seconds": 0.0,
                    "detail": f"would render {len(missing)}/{total_frames} frames at {config['resolution']}",
                }
                print(
                    f"[dry] {key}: {total_frames} frames, {config['resolution']}, "
                    f"{len(missing)} missing"
                )
                continue

            if not missing:
                print(
                    f"{key}: all {total_frames} frames already present, skipping render"
                )
                results[key] = {
                    "video": video_id,
                    "orientation": orientation,
                    "status": "ok",
                    "seconds": 0.0,
                    "detail": "render skipped (frames present)",
                }
            else:
                print(f"{key}: rendering {len(missing)}/{total_frames} frames ...")
                results[key] = {
                    "video": video_id,
                    "orientation": orientation,
                    **vars(
                        run_blender_render(
                            config,
                            tool_paths["blender"],
                            logs_dir / f"render_{key}.log",
                            total_frames,
                        )
                    ),
                }
                if results[key]["status"] != "ok":
                    print(f"{key}: RENDER FAILED - {results[key]['detail']}")
                    continue

            encode_result, _ = run_encode(
                video,
                orientation,
                frames_dir,
                out_root / "videos",
                logs_dir,
                scale,
                fps,
            )
            if encode_result.status == "ok":
                if not keep_frames:
                    shutil.rmtree(frames_dir, ignore_errors=True)
                results[key] = {
                    "video": video_id,
                    "orientation": orientation,
                    "status": "ok",
                    "seconds": results[key].get("seconds", 0.0) + encode_result.seconds,
                    "detail": f"render+encode: {encode_result.detail}",
                    "outputs": [
                        str(out_root / "videos" / f"{video_id}_{orientation}.mp4")
                    ],
                }
                print(f"{key}: encoded ({encode_result.detail})")
            else:
                results[key] = {
                    "video": video_id,
                    "orientation": orientation,
                    "status": "failed",
                    "seconds": results[key].get("seconds", 0.0) + encode_result.seconds,
                    "detail": encode_result.detail,
                }
                print(f"{key}: ENCODE FAILED - {encode_result.detail}")

    report_json, report_md = write_report(out_root, results, started_utc, dry_run)
    print(f"Report: {report_json}")
    print(f"Report: {report_md}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the Pentos visualization batch"
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--videos", nargs="*", choices=VIDEO_IDS, default=list(VIDEO_IDS)
    )
    parser.add_argument(
        "--orientations", nargs="*", choices=ORIENTATIONS, default=list(ORIENTATIONS)
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Resolution scale factor (use <1 for smoke tests)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep frame PNGs after a successful encode (they are deleted by "
        "default to limit disk usage)",
    )
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Re-render and re-encode even when a valid MP4 already exists",
    )
    parser.add_argument(
        "--poster",
        nargs=3,
        metavar=("VIDEO", "ORIENTATION", "FRAME"),
        help="Extract a poster frame and exit",
    )
    args = parser.parse_args()

    if args.poster:
        video_id, orientation, frame = args.poster
        if orientation not in ORIENTATIONS:
            raise SystemExit(f"Unknown orientation {orientation}")
        path = poster_command(args.out_root, video_id, orientation, int(frame))
        print(f"Poster written: {path}")
        return 0

    results = run_batch(
        args.out_root,
        args.videos,
        args.orientations,
        scale=args.scale,
        fps=args.fps,
        dry_run=args.dry_run,
        prepare_only=args.prepare_only,
        keep_frames=args.keep_frames,
        force_reencode=args.force_reencode,
    )
    failed = [key for key, entry in results.items() if entry.get("status") == "failed"]
    return 1 if failed and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
