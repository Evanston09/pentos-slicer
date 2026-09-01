# Using AI to try to tell if klipper will slow down and also get cool details about gcode
import argparse
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from models import DEFAULT_MACHINE_CONFIG, MachineConfig

from gcode_tools import iter_gcode_moves

SMALL_REVERSAL_DEGREES = 0.25
BROAD_REVERSAL_DEGREES = 2.0


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _cruise_ratio(value: str) -> float:
    parsed = _positive_float(value)
    if parsed >= 1.0:
        raise argparse.ArgumentTypeError("must be less than 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


@dataclass
class KlipperMove:
    gcode_index: int
    line: str
    xyz_displacement: tuple[float, float, float]
    e_displacement: float
    a_displacement: float
    b_displacement: float
    xyz_distance: float
    move_distance: float
    requested_velocity: float
    axis_ratios: tuple[float, float, float, float, float, float]
    max_cruise_velocity_squared: float
    max_start_velocity_squared: float
    acceleration: float
    delta_v2: float
    max_mcr_start_velocity_squared: float
    mcr_delta_v2: float
    is_kinematic_move: bool
    nominal_time: float
    planned_start_velocity: float = 0.0
    planned_cruise_velocity: float = 0.0
    planned_end_velocity: float = 0.0
    acceleration_duration: float = 0.0
    cruise_duration: float = 0.0
    deceleration_duration: float = 0.0
    reasons: set[str] = field(default_factory=set)
    junction_reasons: set[str] = field(default_factory=set)

    @property
    def planned_time(self) -> float:
        return (
            self.acceleration_duration
            + self.cruise_duration
            + self.deceleration_duration
        )


@dataclass(frozen=True)
class KlipperPlan:
    moves: list[KlipperMove]

    @property
    def nominal_time(self) -> float:
        return sum(move.nominal_time for move in self.moves)

    @property
    def planned_time(self) -> float:
        return sum(move.planned_time for move in self.moves)


@dataclass(frozen=True)
class RotaryTrace:
    time_seconds: np.ndarray
    a_degrees: np.ndarray
    b_degrees: np.ndarray


@dataclass(frozen=True)
class AxisAnalysis:
    total_travel_degrees: float
    max_step_degrees: float
    max_velocity_deg_s: float
    velocity_limit_deg_s: float
    max_acceleration_deg_s2: float
    acceleration_limit_deg_s2: float
    small_reversals: int
    medium_reversals: int
    broad_reversals: int


def parse_rotary_trace(text: str) -> RotaryTrace | None:
    """Read commanded A/B positions on the nominal G-code motion timeline."""
    elapsed_seconds = 0.0
    times = [0.0]
    a_degrees = [0.0]
    b_degrees = [0.0]
    has_rotary_commands = False

    for move in iter_gcode_moves(text.splitlines()):
        if "A" in move.parsed.args or "B" in move.parsed.args:
            has_rotary_commands = True

        duration = 0.0
        if (
            move.start_xyz is not None
            and move.end_xyz is not None
            and move.feedrate is not None
            and move.feedrate > 0.0
        ):
            distance = float(np.linalg.norm(move.end_xyz - move.start_xyz))
            duration = distance / move.feedrate * 60.0
        elapsed_seconds += duration

        if duration > 0.0 or not np.array_equal(
            move.end_ab, [a_degrees[-1], b_degrees[-1]]
        ):
            times.append(elapsed_seconds)
            a_degrees.append(float(move.end_ab[0]))
            b_degrees.append(float(move.end_ab[1]))

    if not has_rotary_commands:
        return None
    return RotaryTrace(
        time_seconds=np.asarray(times),
        a_degrees=np.asarray(a_degrees),
        b_degrees=np.asarray(b_degrees),
    )


def analyze_axis(
    times: np.ndarray,
    angles: np.ndarray,
    velocity_limit: float,
    acceleration_limit: float,
) -> AxisAnalysis:
    """Measure commanded rotary motion on the nominal G-code timeline."""
    deltas = np.diff(angles)
    moving_deltas = deltas[~np.isclose(deltas, 0.0)]
    reversal_amplitudes = []
    if len(moving_deltas):
        direction = np.sign(moving_deltas[0])
        run_amplitude = abs(moving_deltas[0])
        for delta in moving_deltas[1:]:
            if np.sign(delta) == direction:
                run_amplitude += abs(delta)
            else:
                reversal_amplitudes.append(run_amplitude)
                direction = np.sign(delta)
                run_amplitude = abs(delta)

    unique_times = []
    unique_angles = []
    for time, angle in zip(times, angles):
        if unique_times and np.isclose(time, unique_times[-1]):
            unique_angles[-1] = angle
        else:
            unique_times.append(time)
            unique_angles.append(angle)

    interval_seconds = np.diff(unique_times)
    interval_angles = np.diff(unique_angles)
    velocities = np.divide(
        interval_angles,
        interval_seconds,
        out=np.zeros_like(interval_angles),
        where=interval_seconds > 0.0,
    )
    acceleration_times = (interval_seconds[:-1] + interval_seconds[1:]) / 2.0
    accelerations = np.divide(
        np.diff(velocities),
        acceleration_times,
        out=np.zeros_like(acceleration_times),
        where=acceleration_times > 0.0,
    )
    reversal_amplitudes = np.asarray(reversal_amplitudes)

    return AxisAnalysis(
        total_travel_degrees=float(np.abs(deltas).sum()),
        max_step_degrees=float(np.max(np.abs(deltas), initial=0.0)),
        max_velocity_deg_s=float(np.max(np.abs(velocities), initial=0.0)),
        velocity_limit_deg_s=velocity_limit,
        max_acceleration_deg_s2=float(np.max(np.abs(accelerations), initial=0.0)),
        acceleration_limit_deg_s2=acceleration_limit,
        small_reversals=int(np.sum(reversal_amplitudes < SMALL_REVERSAL_DEGREES)),
        medium_reversals=int(
            np.sum(
                (reversal_amplitudes >= SMALL_REVERSAL_DEGREES)
                & (reversal_amplitudes < BROAD_REVERSAL_DEGREES)
            )
        ),
        broad_reversals=int(np.sum(reversal_amplitudes >= BROAD_REVERSAL_DEGREES)),
    )


def analyze_gcode(
    text: str,
    machine_config: MachineConfig = DEFAULT_MACHINE_CONFIG,
) -> tuple[AxisAnalysis, AxisAnalysis] | None:
    trace = parse_rotary_trace(text)
    if trace is None:
        return None
    return (
        analyze_axis(
            trace.time_seconds,
            trace.a_degrees,
            machine_config.a_max_velocity_deg_s,
            machine_config.a_max_acceleration_deg_s2,
        ),
        analyze_axis(
            trace.time_seconds,
            trace.b_degrees,
            machine_config.b_max_velocity_deg_s,
            machine_config.b_max_acceleration_deg_s2,
        ),
    )


# Planner equations adapted from Klipper klippy/toolhead.py and
# klippy/extras/manual_stepper.py at f0892d82b0f1c1228454f09eb508eddde2250f4b.
def plan_klipper_moves(
    text: str,
    machine_config: MachineConfig,
    *,
    max_velocity: float,
    max_acceleration: float,
    square_corner_velocity: float,
    minimum_cruise_ratio: float,
    a_instant_corner_velocity: float,
    b_instant_corner_velocity: float,
) -> KlipperPlan:
    """Estimate motion using a complete lookahead queue that stops at both ends."""
    junction_deviation = (
        square_corner_velocity**2 * (math.sqrt(2.0) - 1.0) / max_acceleration
    )
    mcr_pseudo_acceleration = max_acceleration * (1.0 - minimum_cruise_ratio)
    moves: list[KlipperMove] = []

    for gcode_move in iter_gcode_moves(text.splitlines()):
        if gcode_move.feedrate is None or gcode_move.feedrate <= 0.0:
            continue
        if gcode_move.start_xyz is None or gcode_move.end_xyz is None:
            xyz_displacement = np.zeros(3)
        else:
            xyz_displacement = gcode_move.end_xyz - gcode_move.start_xyz
        e_displacement = float(gcode_move.extrusion_delta)
        a_displacement, b_displacement = gcode_move.end_ab - gcode_move.start_ab
        xyz_distance = float(np.linalg.norm(xyz_displacement))
        is_kinematic = xyz_distance >= 1.0e-9
        move_distance = (
            xyz_distance
            if is_kinematic
            else max(abs(e_displacement), abs(a_displacement), abs(b_displacement))
        )
        if move_distance == 0.0:
            continue

        inverse_distance = 1.0 / move_distance
        axis_ratios = tuple(
            float(displacement * inverse_distance)
            for displacement in (
                *xyz_displacement,
                e_displacement,
                a_displacement,
                b_displacement,
            )
        )
        requested_velocity = gcode_move.feedrate / 60.0
        cruise_velocity = min(requested_velocity, max_velocity)
        reasons: set[str] = set()
        if cruise_velocity < requested_velocity:
            reasons.add("global velocity limit")
        acceleration = max_acceleration

        for name, displacement, velocity_limit, acceleration_limit in (
            (
                "A",
                a_displacement,
                machine_config.a_max_velocity_deg_s,
                machine_config.a_max_acceleration_deg_s2,
            ),
            (
                "B",
                b_displacement,
                machine_config.b_max_velocity_deg_s,
                machine_config.b_max_acceleration_deg_s2,
            ),
        ):
            if displacement == 0.0:
                continue
            inverse_axis_ratio = move_distance / abs(displacement)
            axis_velocity_limit = velocity_limit * inverse_axis_ratio
            axis_acceleration_limit = acceleration_limit * inverse_axis_ratio
            if axis_velocity_limit < cruise_velocity:
                cruise_velocity = axis_velocity_limit
                reasons.add(f"{name} velocity limit")
            if axis_acceleration_limit < acceleration:
                acceleration = axis_acceleration_limit
                reasons.add(f"{name} acceleration limit")

        delta_v2 = 2.0 * move_distance * acceleration
        move = KlipperMove(
            gcode_index=gcode_move.index,
            line=gcode_move.line,
            xyz_displacement=tuple(float(value) for value in xyz_displacement),
            e_displacement=e_displacement,
            a_displacement=float(a_displacement),
            b_displacement=float(b_displacement),
            xyz_distance=xyz_distance,
            move_distance=move_distance,
            requested_velocity=requested_velocity,
            axis_ratios=axis_ratios,
            max_cruise_velocity_squared=cruise_velocity**2,
            max_start_velocity_squared=0.0,
            acceleration=acceleration,
            delta_v2=delta_v2,
            max_mcr_start_velocity_squared=0.0,
            mcr_delta_v2=min(2.0 * move_distance * mcr_pseudo_acceleration, delta_v2),
            is_kinematic_move=is_kinematic,
            nominal_time=move_distance / requested_velocity,
            reasons=reasons,
        )
        if moves:
            _calculate_junction(
                moves[-1],
                move,
                junction_deviation,
                a_instant_corner_velocity,
                b_instant_corner_velocity,
            )
        moves.append(move)

    _flush_lookahead(moves)
    for index, move in enumerate(moves):
        if move.planned_time <= move.nominal_time + 1.0e-9:
            continue
        if move.planned_start_velocity < move.planned_cruise_velocity:
            if index == 0:
                move.reasons.add("queue start stop")
            elif move.junction_reasons:
                move.reasons.update(move.junction_reasons)
        if move.planned_end_velocity < move.planned_cruise_velocity:
            if index == len(moves) - 1:
                move.reasons.add("queue end stop")
            elif moves[index + 1].junction_reasons:
                move.reasons.update(moves[index + 1].junction_reasons)
        if not move.reasons:
            move.reasons.add("global acceleration limit")

    return KlipperPlan(moves)


def _calculate_junction(
    previous: KlipperMove,
    move: KlipperMove,
    junction_deviation: float,
    a_instant_corner_velocity: float,
    b_instant_corner_velocity: float,
) -> None:
    if not move.is_kinematic_move or not previous.is_kinematic_move:
        move.junction_reasons.add("non-kinematic continuity break")
        return

    current_velocity_reasons = {
        reason for reason in move.reasons if reason.endswith("velocity limit")
    } or {"current requested velocity"}
    previous_velocity_reasons = {
        f"previous move {reason}"
        for reason in previous.reasons
        if reason.endswith("velocity limit")
    } or {"previous requested velocity"}
    previous_acceleration_reasons = {
        f"previous move {reason}"
        for reason in previous.reasons
        if reason.endswith("acceleration limit")
    } or {"previous acceleration reachability"}
    candidates = [
        *(
            (move.max_cruise_velocity_squared, reason)
            for reason in current_velocity_reasons
        ),
        *(
            (previous.max_cruise_velocity_squared, reason)
            for reason in previous_velocity_reasons
        ),
        *(
            (previous.max_start_velocity_squared + previous.delta_v2, reason)
            for reason in previous_acceleration_reasons
        ),
    ]
    for axis_index, name, instant_velocity in (
        (4, "A", a_instant_corner_velocity),
        (5, "B", b_instant_corner_velocity),
    ):
        ratio_change = move.axis_ratios[axis_index] - previous.axis_ratios[axis_index]
        if ratio_change:
            candidates.append(
                (
                    (instant_velocity / abs(ratio_change)) ** 2,
                    f"{name} junction limit",
                )
            )
        # An unchanged ratio adds the current cruise limit, which is already
        # present in candidates.

    dot_product = sum(
        move.axis_ratios[index] * previous.axis_ratios[index] for index in range(3)
    )
    junction_cos_theta = -dot_product
    sin_theta_d2 = math.sqrt(max(0.5 * (1.0 - junction_cos_theta), 0.0))
    cos_theta_d2 = math.sqrt(max(0.5 * (1.0 + junction_cos_theta), 0.0))
    one_minus_sin_theta_d2 = 1.0 - sin_theta_d2
    if one_minus_sin_theta_d2 > 0.0 and cos_theta_d2 > 0.0:
        radius_factor = sin_theta_d2 / one_minus_sin_theta_d2
        quarter_tan_theta_d2 = 0.25 * sin_theta_d2 / cos_theta_d2
        candidates.extend(
            (
                (
                    radius_factor * junction_deviation * move.acceleration,
                    "XYZ corner limit",
                ),
                (
                    radius_factor * junction_deviation * previous.acceleration,
                    "XYZ corner limit",
                ),
                (move.delta_v2 * quarter_tan_theta_d2, "XYZ corner limit"),
                (previous.delta_v2 * quarter_tan_theta_d2, "XYZ corner limit"),
            )
        )

    maximum_start_v2 = min(value for value, _ in candidates)
    move.max_start_velocity_squared = maximum_start_v2
    move.max_mcr_start_velocity_squared = min(
        maximum_start_v2,
        previous.max_mcr_start_velocity_squared + previous.mcr_delta_v2,
    )
    move.junction_reasons.update(
        reason
        for value, reason in candidates
        if math.isclose(value, maximum_start_v2, rel_tol=1.0e-9, abs_tol=1.0e-12)
    )


def _flush_lookahead(moves: list[KlipperMove]) -> None:
    junction_info: list[tuple[KlipperMove, float, float | None, float]] = [
        (move, 0.0, None, 0.0) for move in moves
    ]
    next_start_v2 = next_mcr_start_v2 = peak_cruise_v2 = 0.0
    pending_cruise_assignments = 0

    for index in range(len(moves) - 1, -1, -1):
        move = moves[index]
        reachable_start_v2 = next_start_v2 + move.delta_v2
        start_v2 = min(move.max_start_velocity_squared, reachable_start_v2)
        cruise_v2 = None
        pending_cruise_assignments += 1
        reachable_mcr_start_v2 = next_mcr_start_v2 + move.mcr_delta_v2
        mcr_start_v2 = min(move.max_mcr_start_velocity_squared, reachable_mcr_start_v2)
        if mcr_start_v2 < reachable_mcr_start_v2:
            if (
                mcr_start_v2 + move.mcr_delta_v2 > next_mcr_start_v2
                or pending_cruise_assignments > 1
            ):
                peak_cruise_v2 = (mcr_start_v2 + reachable_mcr_start_v2) * 0.5
            cruise_v2 = min(
                (start_v2 + reachable_start_v2) * 0.5,
                move.max_cruise_velocity_squared,
                peak_cruise_v2,
            )
            pending_cruise_assignments = 0
        junction_info[index] = (move, start_v2, cruise_v2, next_start_v2)
        next_start_v2 = start_v2
        next_mcr_start_v2 = mcr_start_v2

    previous_cruise_v2 = 0.0
    for move, start_v2, cruise_v2, next_start_v2 in junction_info:
        if cruise_v2 is None:
            cruise_v2 = min(previous_cruise_v2, start_v2)
        _set_junction(
            move,
            min(start_v2, cruise_v2),
            cruise_v2,
            min(next_start_v2, cruise_v2),
        )
        previous_cruise_v2 = cruise_v2


def _set_junction(
    move: KlipperMove, start_v2: float, cruise_v2: float, end_v2: float
) -> None:
    half_inverse_acceleration = 0.5 / move.acceleration
    acceleration_distance = (cruise_v2 - start_v2) * half_inverse_acceleration
    deceleration_distance = (cruise_v2 - end_v2) * half_inverse_acceleration
    cruise_distance = move.move_distance - acceleration_distance - deceleration_distance
    move.planned_start_velocity = start_velocity = math.sqrt(start_v2)
    move.planned_cruise_velocity = cruise_velocity = math.sqrt(cruise_v2)
    move.planned_end_velocity = end_velocity = math.sqrt(end_v2)
    move.acceleration_duration = (
        2.0 * acceleration_distance / (start_velocity + cruise_velocity)
        if acceleration_distance
        else 0.0
    )
    move.cruise_duration = cruise_distance / cruise_velocity
    move.deceleration_duration = (
        2.0 * deceleration_distance / (end_velocity + cruise_velocity)
        if deceleration_distance
        else 0.0
    )


def _format_axis(name: str, analysis: AxisAnalysis) -> str:
    velocity_warning = (
        " EXCEEDS"
        if analysis.max_velocity_deg_s > analysis.velocity_limit_deg_s
        else ""
    )
    return "\n".join(
        (
            f"{name} axis",
            f"  Total travel:       {analysis.total_travel_degrees:.2f}°",
            f"  Maximum step:       {analysis.max_step_degrees:.3f}°",
            f"  Maximum velocity:   {analysis.max_velocity_deg_s:.2f}°/s "
            f"(limit {analysis.velocity_limit_deg_s:.2f}){velocity_warning}",
            f"  Finite-diff accel:  {analysis.max_acceleration_deg_s2:.2f}°/s²",
            f"  Small reversals:    {analysis.small_reversals} (<0.25°)",
            f"  Medium reversals:   {analysis.medium_reversals} (0.25–2°)",
            f"  Broad reversals:    {analysis.broad_reversals} (≥2°)",
        )
    )


def _format_plan(plan: KlipperPlan, top: int) -> str:
    slowed_moves = [
        move for move in plan.moves if move.planned_time > move.nominal_time + 1.0e-9
    ]
    added_time = plan.planned_time - plan.nominal_time
    percentage = 100.0 * added_time / plan.nominal_time if plan.nominal_time else 0.0
    sections = [
        "\n".join(
            (
                "Klipper motion estimate (motion-only)",
                f"  Nominal motion time:  {plan.nominal_time:.1f}s",
                f"  Planned motion time:  {plan.planned_time:.1f}s",
                f"  Added slowdown:        {added_time:.1f}s ({percentage:.1f}%)",
                f"  Slowed moves:          {len(slowed_moves)} / {len(plan.moves)}",
            )
        )
    ]

    largest = sorted(
        slowed_moves,
        key=lambda move: move.planned_time - move.nominal_time,
        reverse=True,
    )[:top]
    if largest:
        lines = ["Largest slowdowns (motion-only)"]
        for move in largest:
            ratio = move.planned_time / move.nominal_time
            lines.extend(
                (
                    f"  Line {move.gcode_index + 1}: {move.nominal_time:.3f}s → "
                    f"{move.planned_time:.3f}s ({ratio:.1f}x)",
                    f"    {', '.join(sorted(move.reasons))}",
                )
            )
        sections.append("\n".join(lines))

    cause_counts = Counter(reason for move in slowed_moves for reason in move.reasons)
    if cause_counts:
        lines = ["Slowdown causes"]
        lines.extend(
            f"  {reason}: {count} moves" for reason, count in cause_counts.most_common()
        )
        sections.append("\n".join(lines))
    sections.append(
        "Times are motion-only estimates; heaters, dwells, macros, and MCU "
        "overhead are excluded. The complete motion queue is flushed to a stop; "
        "Klipper normally flushes incrementally."
    )
    return "\n\n".join(sections)


def format_report(
    path: Path,
    text: str,
    machine_config: MachineConfig,
    *,
    max_velocity: float,
    max_acceleration: float,
    square_corner_velocity: float,
    minimum_cruise_ratio: float,
    a_instant_corner_velocity: float,
    b_instant_corner_velocity: float,
    top: int,
) -> str:
    analysis = analyze_gcode(text, machine_config)
    if analysis is None:
        return f"{path}\n  No A/B commands"
    a_analysis, b_analysis = analysis
    plan = plan_klipper_moves(
        text,
        machine_config,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        square_corner_velocity=square_corner_velocity,
        minimum_cruise_ratio=minimum_cruise_ratio,
        a_instant_corner_velocity=a_instant_corner_velocity,
        b_instant_corner_velocity=b_instant_corner_velocity,
    )
    return "\n\n".join(
        (
            str(path),
            _format_axis("A", a_analysis),
            _format_axis("B", b_analysis),
            _format_plan(plan, top),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze A/B motion in G-code")
    parser.add_argument("gcode", nargs="+", type=Path)
    parser.add_argument("--max-velocity", type=_positive_float, required=True)
    parser.add_argument("--max-acceleration", type=_positive_float, required=True)
    parser.add_argument("--square-corner-velocity", type=_positive_float, required=True)
    parser.add_argument("--minimum-cruise-ratio", type=_cruise_ratio, required=True)
    parser.add_argument("--a-max-velocity", type=_positive_float, required=True)
    parser.add_argument("--a-max-acceleration", type=_positive_float, required=True)
    parser.add_argument(
        "--a-instant-corner-velocity", type=_nonnegative_float, required=True
    )
    parser.add_argument("--b-max-velocity", type=_positive_float, required=True)
    parser.add_argument("--b-max-acceleration", type=_positive_float, required=True)
    parser.add_argument(
        "--b-instant-corner-velocity", type=_nonnegative_float, required=True
    )
    parser.add_argument("--top", type=_positive_int, required=True)
    args = parser.parse_args()
    machine_config = replace(
        DEFAULT_MACHINE_CONFIG,
        a_max_velocity_deg_s=args.a_max_velocity,
        a_max_acceleration_deg_s2=args.a_max_acceleration,
        b_max_velocity_deg_s=args.b_max_velocity,
        b_max_acceleration_deg_s2=args.b_max_acceleration,
    )

    for index, path in enumerate(args.gcode):
        if index:
            print()
        print(
            format_report(
                path,
                path.read_text(),
                machine_config,
                max_velocity=args.max_velocity,
                max_acceleration=args.max_acceleration,
                square_corner_velocity=args.square_corner_velocity,
                minimum_cruise_ratio=args.minimum_cruise_ratio,
                a_instant_corner_velocity=args.a_instant_corner_velocity,
                b_instant_corner_velocity=args.b_instant_corner_velocity,
                top=args.top,
            )
        )


if __name__ == "__main__":
    main()
