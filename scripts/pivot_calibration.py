"""Run: uv run python scripts/pivot_calibration.py."""

import csv
from pathlib import Path
import sys

import inquirer
import numpy as np
from inquirer.errors import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machine import rotation_matrix

SUGGESTED_POSES = [
    (0, 0),
    (0, 90),
    (0, -90),
    (15, 0),
    (-15, 0),
    (10, 45),
    (-10, -45),
    (0, 0),
]
FIELDS = ("a_deg", "b_deg", "x_mm", "y_mm", "z_mm")
CSV_PATH = Path(__file__).with_suffix(".csv")


def save(rows):
    with CSV_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def number(_, value):
    try:
        if np.isfinite(float(value)):
            return True
    except ValueError:
        pass
    raise ValidationError("Enter a finite number.")


def solve(rows):
    if not rows:
        raise ValueError("No fit readings. Include A0/B0, B rotation, and A tilt.")
    design, measured = [], []
    for row in rows:
        rotation = rotation_matrix(row["a_deg"], row["b_deg"])
        # measured = (I - R) pivot + R marker_at_zero
        design.append(np.column_stack((np.eye(3) - rotation, rotation)))
        measured.append([row[key] for key in FIELDS[2:]])
    parameters, _, rank, _ = np.linalg.lstsq(
        np.vstack(design), np.concatenate(measured), rcond=None
    )
    if rank != 6:
        raise ValueError(
            "Poses cannot determine the pivot. Include A0/B0, B rotation, and A tilt."
        )
    return parameters


def predict(a, b, parameters):
    pivot, reference = parameters[:3], parameters[3:]
    return pivot + rotation_matrix(a, b) @ (reference - pivot)


def error_distance(row, parameters):
    predicted = predict(row["a_deg"], row["b_deg"], parameters)
    return float(np.linalg.norm(np.array([row[key] for key in FIELDS[2:]]) - predicted))


def calculate(rows):
    parameters = solve(rows)
    errors = [
        {**row, "error_distance_mm": error_distance(row, parameters)} for row in rows
    ]
    distances = [row["error_distance_mm"] for row in errors]
    return {
        "rotation_center_machine_mm": parameters[:3],
        "fit_rms_position_error_mm": np.sqrt(np.mean(np.square(distances))),
        "fit_max_position_error_mm": max(distances),
        "measurements": errors,
    }


def main():
    print("\nPentos Pivot Calculator")
    print("Measure the same marker point using printer XYZ, not bed-local coordinates.")
    print(
        "Check clearance before moving. Readings auto-save; predictions are not saved.\n"
    )
    rows = []
    if CSV_PATH.exists():
        if inquirer.confirm("Load saved readings?", default=True):
            try:
                with CSV_PATH.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        for key in FIELDS:
                            number(None, row[key])
                            row[key] = float(row[key])
                        rows.append(row)
            except (KeyError, ValueError, ValidationError) as error:
                print(f"Could not load readings: {error}. CSV left unchanged.")
                return
        else:
            CSV_PATH.unlink()

    while True:
        action = inquirer.list_input(
            f"{len(rows)} readings",
            choices=["Add", "Edit", "Delete", "Calculate", "Predict", "Quit"],
        )
        if action == "Quit":
            return
        if action == "Calculate":
            try:
                result = calculate(rows)
            except ValueError as error:
                print(error)
                continue
            x, y, z = result["rotation_center_machine_mm"]
            print(f"\nPivot (machine mm): X {x:.4f}  Y {y:.4f}  Z {z:.4f}")
            print(
                f"Fit RMS: {result['fit_rms_position_error_mm']:.4f} mm | Worst: {result['fit_max_position_error_mm']:.4f} mm"
            )
            for row in result["measurements"]:
                print(
                    f"A{row['a_deg']:7.2f} B{row['b_deg']:7.2f}  error {row['error_distance_mm']:.4f} mm"
                )
            continue

        if action == "Predict":
            try:
                parameters = solve(rows)
            except ValueError as error:
                print(error)
                continue
            a = float(inquirer.text("A (degrees)", default="0", validate=number))
            b = float(inquirer.text("B (degrees)", default="0", validate=number))
            x, y, z = predict(a, b, parameters)
            print(f"Predicted marker (machine mm): X {x:.4f}  Y {y:.4f}  Z {z:.4f}")
            continue

        index = len(rows)
        if action in {"Edit", "Delete"}:
            if not rows:
                print("No readings yet.")
                continue
            index = inquirer.list_input(
                "Reading",
                choices=[
                    (
                        f"{i + 1}. A{r['a_deg']:g} B{r['b_deg']:g} XYZ {r['x_mm']:g}, {r['y_mm']:g}, {r['z_mm']:g}",
                        i,
                    )
                    for i, r in enumerate(rows)
                ]
                + [("Back", -1)],
            )
            if index == -1:
                continue
            if action == "Delete":
                del rows[index]
                save(rows)
                continue

        # Add and edit
        a, b = SUGGESTED_POSES[index] if index < len(SUGGESTED_POSES) else (0, 0)
        defaults = (
            rows[index] if action == "Edit" else dict(zip(FIELDS, (a, b, "", "", "")))
        )
        reading = {}
        for key, label in zip(
            FIELDS,
            ("A (degrees)", "B (degrees)", "X (mm)", "Y (mm)", "Z (mm)"),
        ):
            reading[key] = float(
                inquirer.text(
                    message=label, default=str(defaults[key]), validate=number
                )
            )

        rows[index : index + 1] = [reading]
        save(rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited. Fit readings are saved")
