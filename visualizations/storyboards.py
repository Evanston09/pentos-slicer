"""Declarative storyboards for the Pentos visualization shorts.

This module is pure data plus validation: no rendering, no file I/O beyond
what the orchestration layer passes in. Every shot defines its own duration,
caption, per-orientation camera, and a renderer action dict. The Blender
renderer interprets actions; the tests validate this schema.
"""

from dataclasses import dataclass

FPS = 30
PIPELINE_VERSION = "1"

VIDEO_IDS = (
    "meet_pentos",
    "five_axes",
    "tube_divide",
    "tube_continue",
    "nonplanar_guides",
    "nonplanar_map",
)

ORIENTATIONS = ("horizontal", "vertical")
RESOLUTIONS = {"horizontal": (1920, 1080), "vertical": (1080, 1920)}

MACHINE_CAMERA = "machine"
TUBE_SCENE = "tube"
NONPLANAR_SCENE = "nonplanar"

VIDEO_SCENES = {
    "meet_pentos": MACHINE_CAMERA,
    "five_axes": MACHINE_CAMERA,
    "tube_divide": TUBE_SCENE,
    "tube_continue": TUBE_SCENE,
    "nonplanar_guides": NONPLANAR_SCENE,
    "nonplanar_map": NONPLANAR_SCENE,
}

# URDF machine cameras (scene units are millimeters, URDF meters scaled x1000).
# Machine bbox from the URDF STLs is roughly x -215..230, y -250..165, z 0..610.
_ORBIT_MACHINE_WIDE = {
    "orbit": {
        "center": [10.0, -45.0, 260.0],
        "radius": 1350.0,
        "height": 650.0,
        "start_deg": -35.0,
        "end_deg": 20.0,
        "ortho_scale": None,
    },
}
_ORBIT_MACHINE_CLOSE = {
    "orbit": {
        "center": [0.0, -35.0, 330.0],
        "radius": 880.0,
        "height": 500.0,
        "start_deg": -25.0,
        "end_deg": 25.0,
        "ortho_scale": None,
    },
}
_ORBIT_MACHINE_FRONT = {
    "orbit": {
        "center": [0.0, -30.0, 170.0],
        "radius": 900.0,
        "height": 430.0,
        "start_deg": -100.0,
        "end_deg": -80.0,
        "ortho_scale": None,
    },
}


def _machine_camera(horizontal: dict, vertical: dict) -> dict:
    return {"horizontal": horizontal, "vertical": vertical}


def _orbit_variant(base: dict, **overrides: float) -> dict:
    orbit = {**base["orbit"], **overrides}
    return {"orbit": orbit}


# Vertical machine orbits: pull back and up so tall content stays framed.
def _wide_vertical(scale: float) -> dict:
    return {
        "orbit": {
            "center": [10.0, -45.0, 260.0],
            "radius": 1700.0 * scale,
            "height": 800.0 * scale,
            "start_deg": -30.0,
            "end_deg": 15.0,
            "ortho_scale": None,
        }
    }


# Action helpers -----------------------------------------------------------


def _moves(*segments: dict) -> list[dict]:
    return list(segments)


def _move(t0: float, t1: float, joints: dict[str, tuple[float, float]]) -> dict:
    return {"t0": t0, "t1": t1, "joints": {k: list(v) for k, v in joints.items()}}


def _label(
    text: str, at: list[float], t0: float, t1: float, size: float = 26.0
) -> dict:
    return {"text": text, "at": at, "t0": t0, "t1": t1, "size": size}


def _arrow(
    axis: str, at: list[float], t0: float, t1: float, length: float = 90.0
) -> dict:
    return {"axis": axis, "at": at, "t0": t0, "t1": t1, "length": length}


def _machine_action(
    moves: list[dict],
    labels: list[dict] | None = None,
    arrows: list[dict] | None = None,
    highlight: list[str] | None = None,
) -> dict:
    return {
        "type": "machine_pose",
        "moves": moves,
        "labels": labels or [],
        "arrows": arrows or [],
        "highlight": highlight or [],
    }


def _plate_camera(
    center: list[float], radius: float, height: float, start_deg: float, end_deg: float
) -> dict:
    return {
        "orbit": {
            "center": center,
            "radius": radius,
            "height": height,
            "start_deg": start_deg,
            "end_deg": end_deg,
            "ortho_scale": None,
        }
    }


def _reveal_action(
    sample: str,
    window: tuple[int, int] | None = None,
    dial: bool = False,
    transition: bool = False,
    transition_label: bool = False,
) -> dict:
    return {
        "type": "preview_reveal",
        "sample": sample,
        "window": list(window) if window else None,
        "dial": dial,
        "transition": transition,
        "transition_label": transition_label,
    }


# Shot definitions ---------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    id: str
    duration_s: float
    caption: str
    camera: dict  # {"horizontal": ..., "vertical": ...}
    action: dict
    scene: str = ""  # normalized to the owning video's scene in videos()

    @property
    def frames(self) -> int:
        return max(1, round(self.duration_s * FPS))


@dataclass(frozen=True)
class Video:
    id: str
    title: str
    scene: str
    shots: tuple[Shot, ...]

    @property
    def duration_s(self) -> float:
        return sum(shot.duration_s for shot in self.shots)

    @property
    def frames(self) -> int:
        return sum(shot.frames for shot in self.shots)


def _video_meet_pentos() -> Video:
    return Video(
        id="meet_pentos",
        title="Meet Pentos",
        scene=MACHINE_CAMERA,
        shots=(
            Shot(
                id="orbit",
                duration_s=6.0,
                caption="This is Pentos: a 3D printer whose bed can tilt and spin while it prints.",
                camera=_machine_camera(
                    _orbit_variant(_ORBIT_MACHINE_WIDE, start_deg=-40.0, end_deg=15.0),
                    _wide_vertical(1.05),
                ),
                action=_machine_action([]),
            ),
            Shot(
                id="toolhead",
                duration_s=5.5,
                caption="The print head moves in three directions: X, Y, and Z.",
                camera=_machine_camera(_ORBIT_MACHINE_CLOSE, _wide_vertical(0.62)),
                action=_machine_action(
                    moves=[],
                    labels=[
                        _label("X", [0.0, -70.0, 300.0], 0.15, 5.0),
                        _label("Y", [0.0, -220.0, 300.0], 0.15, 5.0),
                        _label("Z", [170.0, -70.0, 660.0], 0.15, 5.0),
                    ],
                    arrows=[
                        _arrow("X", [-127.0, -70.0, 200.0], 0.15, 5.0, length=80.0),
                        _arrow("Y", [0.0, -110.0, 110.0], 0.15, 5.0, length=80.0),
                        _arrow("Z", [100.0, -60.0, 430.0], 0.15, 5.0, length=80.0),
                    ],
                    highlight=["x_carriage_link", "z_gantry_link"],
                ),
            ),
            Shot(
                id="bed_axes",
                duration_s=7.0,
                caption="The bed tilts on A and spins on B, so layers can leave the flat world.",
                camera=_machine_camera(_ORBIT_MACHINE_FRONT, _wide_vertical(0.62)),
                action=_machine_action(
                    moves=[
                        _move(0.5, 3.2, {"a_joint": (0.0, 55.0)}),
                        _move(3.8, 6.5, {"b_joint": (0.0, 120.0)}),
                    ],
                    labels=[
                        _label("A", [70.0, 80.0, 200.0], 0.5, 7.0),
                        _label("B", [70.0, -140.0, 200.0], 3.8, 7.0),
                    ],
                    arrows=[
                        _arrow("A", [0.0, -25.0, 120.0], 0.5, 7.0, length=70.0),
                        _arrow("B", [0.0, -25.0, 120.0], 3.8, 7.0, length=70.0),
                    ],
                    highlight=[" y_carriage_link", "a_tilt_link", "b_rotary_link"],
                ),
            ),
            Shot(
                id="five_axes",
                duration_s=5.5,
                caption="Five controlled axes in total: X, Y, Z, A, and B.",
                camera=_machine_camera(_ORBIT_MACHINE_WIDE, _wide_vertical(1.05)),
                action=_machine_action(
                    moves=[
                        _move(
                            0.2,
                            4.8,
                            {
                                "z_joint": (0.0, 0.12),
                                "x_joint": (0.0, 0.14),
                                "y_joint": (0.0, 0.1),
                                "a_joint": (0.0, -35.0),
                                "b_joint": (0.0, 60.0),
                            },
                        )
                    ],
                    labels=[
                        _label("X", [0.0, -70.0, 300.0], 0.2, 5.5),
                        _label("Y", [0.0, -220.0, 300.0], 0.2, 5.5),
                        _label("Z", [170.0, -70.0, 660.0], 0.2, 5.5),
                        _label("A", [70.0, 80.0, 200.0], 0.2, 5.5),
                        _label("B", [70.0, -140.0, 200.0], 0.2, 5.5),
                    ],
                ),
            ),
        ),
    )


def _video_five_axes() -> Video:
    close_h = {
        "orbit": {
            "center": [120.0, -40.0, 210.0],
            "radius": 430.0,
            "height": 300.0,
            "start_deg": -18.0,
            "end_deg": 14.0,
            "ortho_scale": None,
        }
    }
    return Video(
        id="five_axes",
        title="Five Axes in Motion",
        scene=MACHINE_CAMERA,
        shots=(
            Shot(
                id="axis_x",
                duration_s=5.0,
                caption="X: the print head slides across the gantry.",
                camera=_machine_camera(close_h, _wide_vertical(0.62)),
                action=_machine_action(
                    moves=[_move(0.8, 4.2, {"x_joint": (0.0, 0.18)})],
                    labels=[_label("X", [0.0, -70.0, 300.0], 0.8, 5.0)],
                    arrows=[_arrow("X", [-127.0, -70.0, 200.0], 0.8, 5.0, length=80.0)],
                    highlight=["x_carriage_link", "z_gantry_link"],
                ),
            ),
            Shot(
                id="axis_y",
                duration_s=5.0,
                caption="Y: the bed sled travels toward and away from the front.",
                camera=_machine_camera(close_h, _wide_vertical(0.62)),
                action=_machine_action(
                    moves=[_move(0.8, 4.2, {"y_joint": (0.0, 0.13)})],
                    labels=[_label("Y", [0.0, -220.0, 300.0], 0.8, 5.0)],
                    arrows=[_arrow("Y", [0.0, -110.0, 110.0], 0.8, 5.0, length=80.0)],
                    highlight=[" y_carriage_link", "a_tilt_link", "b_rotary_link"],
                ),
            ),
            Shot(
                id="axis_z",
                duration_s=5.0,
                caption="Z: the gantry lifts the whole print head.",
                camera=_machine_camera(close_h, _wide_vertical(0.62)),
                action=_machine_action(
                    moves=[_move(0.8, 4.2, {"z_joint": (0.0, 0.17)})],
                    labels=[_label("Z", [170.0, -70.0, 660.0], 0.8, 5.0)],
                    arrows=[_arrow("Z", [100.0, -60.0, 430.0], 0.8, 5.0, length=80.0)],
                    highlight=["x_carriage_link", "z_gantry_link"],
                ),
            ),
            Shot(
                id="axis_a",
                duration_s=6.5,
                caption="A tilts the bed, matching the machine's own rotation math at A-positive.",
                camera=_machine_camera(_ORBIT_MACHINE_FRONT, _wide_vertical(0.62)),
                action=_machine_action(
                    moves=[
                        _move(0.6, 3.4, {"a_joint": (0.0, -60.0)}),
                        _move(3.4, 6.0, {"a_joint": (-60.0, 0.0)}),
                    ],
                    labels=[_label("A+", [70.0, 80.0, 200.0], 0.6, 6.5)],
                    arrows=[_arrow("A", [0.0, -25.0, 120.0], 0.6, 6.5, length=70.0)],
                    highlight=[" y_carriage_link", "a_tilt_link", "b_rotary_link"],
                ),
            ),
            Shot(
                id="axis_b",
                duration_s=6.5,
                caption="B spins the bed about its center, so a point on the right sweeps to the front.",
                camera=_machine_camera(_ORBIT_MACHINE_FRONT, _wide_vertical(0.62)),
                action=_machine_action(
                    moves=[_move(0.6, 5.8, {"b_joint": (0.0, -210.0)})],
                    labels=[_label("B+", [70.0, -140.0, 200.0], 0.6, 6.5)],
                    arrows=[_arrow("B", [0.0, -25.0, 120.0], 0.6, 6.5, length=70.0)],
                    highlight=[" y_carriage_link", "a_tilt_link", "b_rotary_link"],
                ),
            ),
            Shot(
                id="combined",
                duration_s=7.0,
                caption="Together the five axes hold the print surface where the nozzle needs it.",
                camera=_machine_camera(_ORBIT_MACHINE_WIDE, _wide_vertical(1.05)),
                action=_machine_action(
                    moves=[
                        _move(
                            0.2,
                            3.4,
                            {
                                "x_joint": (0.0, 0.16),
                                "z_joint": (0.0, 0.1),
                                "a_joint": (0.0, -45.0),
                            },
                        ),
                        _move(
                            3.4,
                            6.6,
                            {
                                "y_joint": (0.0, 0.08),
                                "b_joint": (0.0, 90.0),
                                "a_joint": (-45.0, 20.0),
                            },
                        ),
                    ],
                ),
            ),
        ),
    )


def _video_tube_divide() -> Video:
    wide_h = _plate_camera([45.0, 30.0, 28.0], 195.0, 135.0, -30.0, 5.0)
    wide_v = _plate_camera([45.0, 30.0, 28.0], 225.0, 160.0, -30.0, 5.0)
    return Video(
        id="tube_divide",
        title="Multiplanar: Divide the Tube",
        scene=TUBE_SCENE,
        shots=(
            Shot(
                id="plane_appears",
                duration_s=6.0,
                caption="This tube is a normal 3D model. A tilted cut plane will split it into pieces.",
                camera={"horizontal": wide_h, "vertical": wide_v},
                action={"type": "plane_cut", "phase": "plane", "separate": 0.0},
            ),
            Shot(
                id="chunks_separate",
                duration_s=6.5,
                caption="The cut divides the model into chunks, each with its own print-up direction.",
                camera={"horizontal": wide_h, "vertical": wide_v},
                action={"type": "plane_cut", "phase": "separate", "separate": 1.0},
            ),
            Shot(
                id="chunk_flattens",
                duration_s=7.5,
                caption="Each chunk is rotated until it lies flat, using the real A and B angles.",
                camera={"horizontal": wide_h, "vertical": wide_v},
                action={"type": "chunk_flatten", "chunk": 1, "flat_hold": 0.35},
            ),
            Shot(
                id="flat_on_plate",
                duration_s=6.0,
                caption="Flat pieces slice like any everyday print. The bed pose A, B does the rest.",
                camera={"horizontal": wide_h, "vertical": wide_v},
                action={"type": "chunks_flat_overview"},
            ),
        ),
    )


def _video_tube_continue() -> Video:
    cam_h = _plate_camera([45.0, 30.0, 36.0], 205.0, 140.0, -22.0, -14.0)
    cam_v = _plate_camera([45.0, 30.0, 36.0], 235.0, 165.0, -22.0, -14.0)
    return Video(
        id="tube_continue",
        title="Multiplanar: Reorient and Continue",
        scene=TUBE_SCENE,
        shots=(
            Shot(
                id="chunk_one",
                duration_s=7.5,
                caption="The first chunk prints as a normal flat layer cake on the untilted bed.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action=_reveal_action("tube", window=(0.0, 0.62), dial=True),
            ),
            Shot(
                id="transition",
                duration_s=8.0,
                caption="At the cut, the nozzle lifts to safe Z, the bed swings to the next A and B, and seeks the continuation point.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action=_reveal_action(
                    "tube",
                    window=(0.56, 0.82),
                    dial=True,
                    transition=True,
                    transition_label=True,
                ),
            ),
            Shot(
                id="chunk_two",
                duration_s=7.5,
                caption="Printing resumes exactly where chunk one stopped, now on the tilted bed.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action=_reveal_action("tube", window=(0.78, 1.0), dial=True),
            ),
            Shot(
                id="merged",
                duration_s=7.0,
                caption="One merged G-code file. PrusaSlicer's temporary centering is removed so the physical path is exact.",
                camera={
                    "horizontal": _plate_camera(
                        [45.0, 25.0, 42.0], 215.0, 150.0, -40.0, 0.0
                    ),
                    "vertical": _plate_camera(
                        [45.0, 25.0, 42.0], 245.0, 175.0, -40.0, 0.0
                    ),
                },
                action=_reveal_action("tube", window=None, dial=False),
            ),
        ),
    )


def _video_nonplanar_guides() -> Video:
    cam_h = _plate_camera([45.0, 32.0, 32.0], 200.0, 135.0, -30.0, -5.0)
    cam_v = _plate_camera([45.0, 32.0, 32.0], 230.0, 160.0, -30.0, -5.0)
    return Video(
        id="nonplanar_guides",
        title="Nonplanar: Guides to Flat Space",
        scene=NONPLANAR_SCENE,
        shots=(
            Shot(
                id="guides",
                duration_s=6.5,
                caption="Curved guide surfaces are placed where the layers should bend.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action={"type": "guide_field", "phase": "guides"},
            ),
            Shot(
                id="field",
                duration_s=7.0,
                caption="A smooth height field is solved through the whole solid, one value per point.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action={"type": "guide_field", "phase": "field"},
            ),
            Shot(
                id="morph",
                duration_s=8.0,
                caption="The solid is flattened: curved layers become perfectly flat slicing space.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action={"type": "guide_field", "phase": "morph"},
            ),
            Shot(
                id="flat_space",
                duration_s=6.0,
                caption="In flat space it is an everyday print, sliced by stock PrusaSlicer.",
                camera={
                    "horizontal": _plate_camera(
                        [45.0, 28.0, 32.0], 210.0, 145.0, -55.0, -10.0
                    ),
                    "vertical": _plate_camera(
                        [45.0, 28.0, 32.0], 240.0, 170.0, -55.0, -10.0
                    ),
                },
                action={"type": "guide_field", "phase": "flat"},
            ),
        ),
    )


def _video_nonplanar_map() -> Video:
    cam_h = _plate_camera([45.0, 32.0, 32.0], 200.0, 135.0, -25.0, 5.0)
    cam_v = _plate_camera([45.0, 32.0, 32.0], 230.0, 160.0, -25.0, 5.0)
    return Video(
        id="nonplanar_map",
        title="Nonplanar: Map the Toolpath Back",
        scene=NONPLANAR_SCENE,
        shots=(
            Shot(
                id="planar_paths",
                duration_s=7.5,
                caption="First the flattened model is sliced: simple, planar layer paths.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action={"type": "mapped_reveal", "phase": "planar", "dial": False},
            ),
            Shot(
                id="map_back",
                duration_s=8.0,
                caption="Each path point is mapped back through the deformation onto the curved solid.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action={"type": "mapped_reveal", "phase": "map", "dial": False},
            ),
            Shot(
                id="curved_toolpath",
                duration_s=8.0,
                caption="The nozzle tilts with A and B to lay material along the curves.",
                camera={"horizontal": cam_h, "vertical": cam_v},
                action={"type": "mapped_reveal", "phase": "curved", "dial": True},
            ),
            Shot(
                id="base_layers",
                duration_s=6.5,
                caption="The first layers stay flat, then blend smoothly into curved space.",
                camera={
                    "horizontal": _plate_camera(
                        [45.0, 45.0, 12.0], 135.0, 80.0, -20.0, 10.0
                    ),
                    "vertical": _plate_camera(
                        [45.0, 45.0, 12.0], 160.0, 95.0, -20.0, 10.0
                    ),
                },
                action={"type": "mapped_reveal", "phase": "base", "dial": False},
            ),
        ),
    )


_VIDEOS = {
    "meet_pentos": _video_meet_pentos,
    "five_axes": _video_five_axes,
    "tube_divide": _video_tube_divide,
    "tube_continue": _video_tube_continue,
    "nonplanar_guides": _video_nonplanar_guides,
    "nonplanar_map": _video_nonplanar_map,
}


def videos() -> dict[str, Video]:
    from dataclasses import replace

    result = {}
    for video_id, builder in _VIDEOS.items():
        video = builder()
        shots = tuple(replace(shot, scene=video.scene) for shot in video.shots)
        result[video_id] = replace(video, shots=shots)
    return result


# Validation ---------------------------------------------------------------

_ACTION_TYPES = {
    "machine_pose",
    "plane_cut",
    "chunk_flatten",
    "chunks_flat_overview",
    "preview_reveal",
    "guide_field",
    "mapped_reveal",
}

_ACTION_REQUIRED_KEYS = {
    "machine_pose": {"type", "moves", "labels", "arrows", "highlight"},
    "plane_cut": {"type", "phase", "separate"},
    "chunk_flatten": {"type", "chunk", "flat_hold"},
    "chunks_flat_overview": {"type"},
    "preview_reveal": {
        "type",
        "sample",
        "window",
        "dial",
        "transition",
        "transition_label",
    },
    "guide_field": {"type", "phase"},
    "mapped_reveal": {"type", "phase", "dial"},
}

_PLANE_PHASES = {"plane", "separate"}
_GUIDE_PHASES = {"guides", "field", "morph", "flat"}
_MAPPED_PHASES = {"planar", "map", "curved", "base"}


class StoryboardError(ValueError):
    pass


def validate_storyboards(videos: dict[str, Video]) -> None:
    """Validate catalog invariants; raises StoryboardError on any problem."""
    if set(videos) != set(VIDEO_IDS):
        raise StoryboardError(
            f"Video ids {sorted(videos)} do not match catalog {sorted(VIDEO_IDS)}"
        )
    for video_id, video in videos.items():
        if video.id != video_id:
            raise StoryboardError(f"Video id mismatch: {video.id} != {video_id}")
        if video.scene != VIDEO_SCENES[video_id]:
            raise StoryboardError(f"Scene mismatch for {video_id}")
        if not video.shots:
            raise StoryboardError(f"Video {video_id} has no shots")
        shot_ids = set()
        for shot in video.shots:
            if shot.id in shot_ids:
                raise StoryboardError(f"Duplicate shot id {shot.id} in {video_id}")
            shot_ids.add(shot.id)
            if shot.duration_s <= 0 or shot.frames < 1:
                raise StoryboardError(f"Shot {video_id}/{shot.id} has bad duration")
            if not shot.caption.strip():
                raise StoryboardError(f"Shot {video_id}/{shot.id} has empty caption")
            _validate_camera(shot.camera, video_id, shot.id)
            _validate_action(shot.action, video_id, shot.id)


def _validate_camera(camera: dict, video_id: str, shot_id: str) -> None:
    if set(camera) != set(ORIENTATIONS):
        raise StoryboardError(
            f"Camera for {video_id}/{shot_id} must define both orientations"
        )
    for orientation, spec in camera.items():
        orbit = spec.get("orbit")
        if not isinstance(orbit, dict):
            raise StoryboardError(
                f"Camera {video_id}/{shot_id}/{orientation} needs orbit"
            )
        required = {"center", "radius", "height", "start_deg", "end_deg", "ortho_scale"}
        if set(orbit) != required:
            raise StoryboardError(
                f"Camera {video_id}/{shot_id}/{orientation} keys {sorted(orbit)} != {sorted(required)}"
            )
        if len(orbit["center"]) != 3:
            raise StoryboardError(
                f"Camera {video_id}/{shot_id}/{orientation} center needs 3 values"
            )
        if orbit["radius"] <= 0 or orbit["height"] < 0:
            raise StoryboardError(
                f"Camera {video_id}/{shot_id}/{orientation} radius/height invalid"
            )


def _validate_action(action: dict, video_id: str, shot_id: str) -> None:
    action_type = action.get("type")
    if action_type not in _ACTION_TYPES:
        raise StoryboardError(
            f"Unknown action type {action_type!r} in {video_id}/{shot_id}"
        )
    required = _ACTION_REQUIRED_KEYS[action_type]
    if not required.issubset(action):
        raise StoryboardError(
            f"Action {action_type} in {video_id}/{shot_id} missing keys {sorted(required - set(action))}"
        )
    if action_type == "machine_pose":
        for move in action["moves"]:
            if not {"t0", "t1", "joints"}.issubset(move):
                raise StoryboardError(f"Bad move in {video_id}/{shot_id}: {move}")
            if move["t0"] < 0 or move["t1"] < move["t0"]:
                raise StoryboardError(
                    f"Bad move window in {video_id}/{shot_id}: {move}"
                )
            for joint, values in move["joints"].items():
                if len(values) != 2:
                    raise StoryboardError(
                        f"Joint {joint} needs [start, end] in {video_id}/{shot_id}"
                    )
        for label in action["labels"]:
            if (
                not {"text", "at", "t0", "t1", "size"}.issubset(label)
                or len(label["at"]) != 3
            ):
                raise StoryboardError(f"Bad label in {video_id}/{shot_id}: {label}")
        for arrow in action["arrows"]:
            if (
                not {"axis", "at", "t0", "t1", "length"}.issubset(arrow)
                or len(arrow["at"]) != 3
            ):
                raise StoryboardError(f"Bad arrow in {video_id}/{shot_id}: {arrow}")
            if arrow["axis"] not in {"X", "Y", "Z", "A", "B"}:
                raise StoryboardError(
                    f"Bad arrow axis in {video_id}/{shot_id}: {arrow['axis']}"
                )
    elif action_type == "plane_cut":
        if action["phase"] not in _PLANE_PHASES:
            raise StoryboardError(f"Bad plane_cut phase in {video_id}/{shot_id}")
    elif action_type == "chunk_flatten":
        if not isinstance(action["chunk"], int) or action["chunk"] < 0:
            raise StoryboardError(f"Bad chunk index in {video_id}/{shot_id}")
        if not 0.0 <= action["flat_hold"] < 1.0:
            raise StoryboardError(f"Bad flat_hold in {video_id}/{shot_id}")
    elif action_type == "preview_reveal":
        if action["sample"] not in {"tube"}:
            raise StoryboardError(f"Bad preview sample in {video_id}/{shot_id}")
        window = action["window"]
        if window is not None and (
            len(window) != 2 or not 0 <= window[0] < window[1] <= 1
        ):
            raise StoryboardError(f"Bad reveal window in {video_id}/{shot_id}")
    elif action_type == "guide_field":
        if action["phase"] not in _GUIDE_PHASES:
            raise StoryboardError(f"Bad guide_field phase in {video_id}/{shot_id}")
    elif action_type == "mapped_reveal":
        if action["phase"] not in _MAPPED_PHASES:
            raise StoryboardError(f"Bad mapped_reveal phase in {video_id}/{shot_id}")


def shot_frame_ranges(video: Video) -> list[tuple[Shot, int, int]]:
    """Return (shot, first_frame, last_frame) with 1-based global frame numbers."""
    ranges = []
    next_frame = 1
    for shot in video.shots:
        ranges.append((shot, next_frame, next_frame + shot.frames - 1))
        next_frame += shot.frames
    return ranges
