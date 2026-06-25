def transition(
    initial_xyz: tuple[float, float, float],
    a_degrees: float,
    b_degrees: float,
) -> list[str]:
    gcode = [
        "\n; --- PENTOS A/B TRANSITION ---\n",
    ]

    gcode.extend(
        [
            "G91 ; relative movement for safe lift\n",
            "G1 Z15 F3000\n",
            "G90 ; absolute movement\n",
            f"G1 A{a_degrees} B{b_degrees} F1200\n",
            "; --- PENTOS MOVE TO NEXT CHUNK ---\n",
            f"G1 X{initial_xyz[0]} Y{initial_xyz[1]} F1200\n",
            f"G1 Z{initial_xyz[2]} F1200\n",
        ]
    )

    gcode.append("; --- END PENTOS A/B TRANSITION ---\n")
    return gcode
