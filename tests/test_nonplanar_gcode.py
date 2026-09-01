from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from gcode_tools import GcodeMove, iter_gcode_moves
from machine import rotation_matrix
from models import DEFAULT_MACHINE_CONFIG
from services.nonplanar_gcode import (
    _ab_angles,
    _commanded_normal,
    _smooth_angles,
    _smooth_normals,
    map_gcode_to_original,
)
from services.volumetric_deformation import TetrahedralVolume


class _SpikedNormalVolume:
    @staticmethod
    def inverse_map_properties(
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = np.where(np.isclose(points[:, 0], 0.3), np.radians(8.0), 0.0)
        normals = np.column_stack(
            (-np.sin(angles), np.zeros(len(angles)), np.cos(angles))
        )
        return points, np.ones(len(points)), normals


def _mapped_ab_moves(text: str) -> list[GcodeMove]:
    return [
        move for move in iter_gcode_moves(text.splitlines()) if "A" in move.parsed.args
    ]


def test_ab_angles_hold_b_near_vertical_and_track_real_tilt() -> None:
    tilt = np.radians(20.0)
    normals = np.array(
        [
            [0.005, 0.005, 1.0],
            [-0.005, 0.005, 1.0],
            [0.0, -np.sin(tilt), np.cos(tilt)],
        ]
    )

    angles = []
    previous_b = 0.0
    for normal in normals:
        angle = _ab_angles(
            normal,
            previous_b,
            DEFAULT_MACHINE_CONFIG,
            DEFAULT_MACHINE_CONFIG.max_normal_error_degrees,
        )
        angles.append(angle)
        previous_b = angle[1]
    angles = np.asarray(angles)

    assert_allclose(angles[:2, 1], 0.0, atol=1e-4)
    assert abs(angles[2, 1]) > 84.0
    assert np.all(np.abs(angles[:, 1]) <= 180.0)
    commanded = np.asarray([_commanded_normal(angle) for angle in angles])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    errors = np.degrees(
        np.arccos(np.clip(np.sum(commanded * normals, axis=1), -1.0, 1.0))
    )
    assert np.all(errors <= DEFAULT_MACHINE_CONFIG.max_normal_error_degrees + 0.01)


def test_smooth_normals_reduces_spike_with_bounded_error() -> None:
    angles = np.radians([0.0, 0.0, 8.0, 0.0, 0.0])
    normals = np.column_stack((-np.sin(angles), np.zeros(len(angles)), np.cos(angles)))

    smoothed = _smooth_normals(
        normals,
        np.full(len(normals), 0.01),
        max_error_degrees=2.0,
    )
    smoothed_angles = np.degrees(np.arctan2(-smoothed[:, 0], smoothed[:, 2]))
    errors = np.degrees(
        np.arccos(np.clip(np.sum(normals * smoothed, axis=1), -1.0, 1.0))
    )

    assert np.max(np.abs(np.diff(smoothed_angles))) < 8.0
    assert np.all(errors <= 2.0 + 1e-8)
    assert_allclose(np.linalg.norm(smoothed, axis=1), 1.0)
    zero_duration = _smooth_normals(normals, np.zeros(len(normals)), 2.0)
    assert np.all(np.isfinite(zero_duration))


def test_smooth_angles_reduces_reversals_with_bounded_error() -> None:
    source_a = np.array([0.0, 0.0, 8.0, 0.0, 0.0])
    angles = np.column_stack((source_a, np.zeros(len(source_a))))
    source_normals = np.asarray([_commanded_normal(angle) for angle in angles])

    smoothed = _smooth_angles(
        angles,
        source_normals,
        np.full(len(angles), 0.01),
        DEFAULT_MACHINE_CONFIG,
    )
    commanded = np.asarray([_commanded_normal(angle) for angle in smoothed])
    errors = np.degrees(
        np.arccos(np.clip(np.sum(source_normals * commanded, axis=1), -1.0, 1.0))
    )

    assert np.max(np.abs(np.diff(smoothed[:, 0]))) < 8.0
    assert np.all(errors <= DEFAULT_MACHINE_CONFIG.max_normal_error_degrees + 1e-6)


def test_smooth_angles_handles_b_wrap() -> None:
    angles = np.array([[10.0, 179.0], [10.0, -179.0], [10.0, -178.0]])
    source_normals = np.asarray([_commanded_normal(angle) for angle in angles])

    smoothed = _smooth_angles(
        angles,
        source_normals,
        np.full(len(angles), 0.01),
        DEFAULT_MACHINE_CONFIG,
    )

    assert np.all(np.abs(smoothed[:, 1]) > 170.0)
    assert np.all(smoothed[:, 1] >= DEFAULT_MACHINE_CONFIG.b_degrees_min)
    assert np.all(smoothed[:, 1] <= DEFAULT_MACHINE_CONFIG.b_degrees_max)


def test_smooth_normals_is_segmentation_invariant() -> None:
    def sampled_normals(times: np.ndarray) -> np.ndarray:
        angles = np.radians(8.0 * np.sin(2.0 * np.pi * times))
        return np.column_stack((-np.sin(angles), np.zeros(len(angles)), np.cos(angles)))

    coarse_times = np.arange(0.01, 0.201, 0.01)
    fine_times = np.arange(0.005, 0.201, 0.005)
    coarse = _smooth_normals(
        sampled_normals(coarse_times), np.full(len(coarse_times), 0.01), 2.0
    )
    fine = _smooth_normals(
        sampled_normals(fine_times), np.full(len(fine_times), 0.005), 2.0
    )

    errors = np.degrees(
        np.arccos(np.clip(np.sum(coarse * fine[1::2], axis=1), -1.0, 1.0))
    )
    assert np.max(errors) < 0.25


def test_map_gcode_uses_source_feedrate_for_smoothing() -> None:
    machine_config = replace(
        DEFAULT_MACHINE_CONFIG,
        max_normal_error_degrees=2.0,
    )
    offset = np.asarray(machine_config.machine_offset)
    start = offset + [0.0, 0.1, 0.1]
    end = offset + [0.5, 0.1, 0.1]
    layers = ";LAYER_CHANGE\n" * 6

    def center_angle(feedrate: float) -> float:
        text = (
            "G90\nM83\n"
            f"G1 X{start[0]} Y{start[1]} Z{start[2]} F600\n"
            f"{layers}"
            f"G1 X{end[0]} Y{end[1]} Z{end[2]} E0.5 F{feedrate}\n"
        )
        mapped = map_gcode_to_original(text, _SpikedNormalVolume(), machine_config, 0.1)
        return float(_mapped_ab_moves(mapped)[-3].end_ab[0])

    assert center_angle(60.0) > center_angle(6000.0) + 0.05


def test_map_gcode_smooths_normals_before_xyz_compensation() -> None:
    offset = np.asarray(DEFAULT_MACHINE_CONFIG.machine_offset)
    start = offset + [0.0, 0.1, 0.1]
    end = offset + [0.5, 0.1, 0.1]
    layers = (f";LAYER_CHANGE\nG1 X{start[0]} Y{start[1]} Z{start[2]}\n") * 6
    text = (
        "G90\n"
        "M83\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]} F600\n"
        f"{layers}"
        f"G1 X{end[0]} Y{end[1]} Z{end[2]} E0.5\n"
    )

    mapped = map_gcode_to_original(
        text,
        _SpikedNormalVolume(),
        DEFAULT_MACHINE_CONFIG,
        max_segment_length=0.1,
    )
    mapped_moves = _mapped_ab_moves(mapped)
    center_move = mapped_moves[-3]

    assert 6.0 <= center_move.end_ab[0] < 8.0
    center = np.asarray(DEFAULT_MACHINE_CONFIG.rotation_center_local_mm)
    expected_local = center + rotation_matrix(center_move.end_ab[0], 0.0) @ (
        np.array([0.3, 0.1, 0.1]) - center
    )
    assert_allclose(center_move.end_xyz, offset + expected_local, atol=1e-5)


def test_map_gcode_smooths_across_extrusion_and_travel() -> None:
    offset = np.asarray(DEFAULT_MACHINE_CONFIG.machine_offset)
    start = offset + [0.0, 0.1, 0.1]
    middle = offset + [0.2, 0.1, 0.1]
    end = offset + [0.5, 0.1, 0.1]
    layers = ";LAYER_CHANGE\n" * 6
    text = (
        "G90\nM83\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]} F600\n"
        f"{layers}"
        f"G1 X{middle[0]} Y{middle[1]} Z{middle[2]} E0.2\n"
        f"G1 X{end[0]} Y{end[1]} Z{end[2]}\n"
    )

    mapped = map_gcode_to_original(
        text, _SpikedNormalVolume(), DEFAULT_MACHINE_CONFIG, 0.1
    )
    assert abs(_mapped_ab_moves(mapped)[-4].end_ab[0]) > 0.5


def test_map_gcode_keeps_combined_orientation_error_bounded() -> None:
    source_normals = np.array(
        [
            [-0.16946263, -0.31162782, 0.93497087],
            [-0.07029540, -0.38629692, 0.91969193],
            [-0.09697608, 0.02327712, 0.99501448],
            [-0.17812250, 0.15858819, 0.97114477],
            [0.13911756, -0.12692026, 0.98210873],
        ]
    )

    class VaryingNormalVolume:
        @staticmethod
        def inverse_map_properties(
            points: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return points, np.ones(len(points)), source_normals

    offset = np.asarray(DEFAULT_MACHINE_CONFIG.machine_offset)
    start = offset + [0.0, 0.1, 0.1]
    end = offset + [0.5, 0.1, 0.1]
    layers = ";LAYER_CHANGE\n" * 6
    text = (
        "G90\nM83\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]} F600\n"
        f"{layers}"
        f"G1 X{end[0]} Y{end[1]} Z{end[2]} E0.5\n"
    )

    mapped = map_gcode_to_original(
        text, VaryingNormalVolume(), DEFAULT_MACHINE_CONFIG, 0.1
    )
    angles = np.asarray([move.end_ab for move in _mapped_ab_moves(mapped)[-5:]])
    commanded = np.asarray([_commanded_normal(angle) for angle in angles])
    errors = np.degrees(
        np.arccos(np.clip(np.sum(source_normals * commanded, axis=1), -1.0, 1.0))
    )
    assert np.all(errors <= DEFAULT_MACHINE_CONFIG.max_normal_error_degrees + 1e-4)


def test_map_gcode_inverse_maps_and_compensates_extrusion() -> None:
    original = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    volume = TetrahedralVolume(
        original_vertices=original,
        deformed_vertices=original * [2.0, 1.0, 1.0],
        tetrahedra=np.array([[0, 1, 2, 3]]),
        boundary_faces=np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
        scalar_values=original[:, 2],
    )
    offset = np.asarray(DEFAULT_MACHINE_CONFIG.machine_offset)
    start = offset + [0.0, 0.1, 0.1]
    end = offset + [0.4, 0.1, 0.1]
    text = (
        "ENABLE_FIVE_AXIS\n"
        "G90\n"
        "M83\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]} F600\n"
        ";LAYER_CHANGE\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]}\n"
        ";LAYER_CHANGE\n"
        f"G1 X{start[0]} Y{start[1]} Z{start[2]}\n"
        ";LAYER_CHANGE\n"
        f"G1 X{end[0]} Y{end[1]} Z{end[2]} E0.9\n"
    )

    mapped = map_gcode_to_original(
        text,
        volume,
        DEFAULT_MACHINE_CONFIG,
        max_segment_length=0.15,
    )
    moves = list(iter_gcode_moves(mapped.splitlines()))

    assert len(moves) == 6
    assert all("A" not in move.parsed.args for move in moves[1:3])
    assert_allclose(moves[-1].end_xyz, offset + [0.35, 0.1, 0.1])
    assert_allclose(
        [move.extrusion_delta for move in moves[-3:]],
        [0.2625, 0.2625, 0.2625],
    )
    assert_allclose([move.feedrate for move in moves[-3:]], 525.0)
    assert_allclose(
        [[move.parsed.args["A"], move.parsed.args["B"]] for move in moves[-3:]],
        0.0,
    )
