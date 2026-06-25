from .commands import GcodeCommand, is_comment_line


def trim_gcode(
    lines: list[str],
    index: int,
    total: int,
) -> list[str]:
    if total == 1:
        return lines
    if index == 0:
        return remove_end(lines)
    if index == total - 1:
        return remove_start(lines)
    return remove_start(remove_end(lines))


def remove_end(lines: list[str]) -> list[str]:
    custom_blocks = [
        i
        for i, line in enumerate(lines)
        if is_comment_line(line, lambda comment: comment == "TYPE:Custom")
    ]
    return lines[: custom_blocks[-1]] if custom_blocks else lines


def remove_start(lines: list[str]) -> list[str]:
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if is_comment_line(line, lambda comment: comment == "LAYER_CHANGE")
        ),
        0,
    )
    return lines[start:]


def remove_leading_retract(lines: list[str]) -> list[str]:
    cleaned = []
    before_print_type = True

    for line in lines:
        if before_print_type and is_comment_line(
            line,
            lambda comment: comment is not None and comment.startswith("TYPE:"),
        ):
            before_print_type = False

        parsed = GcodeCommand.parse(line)
        is_leading_retract = (
            parsed.command in {"G0", "G1"}
            and parsed.args.get("E", 0.0) < 0.0
            and all(key in {"E", "F"} for key in parsed.args)
        )
        if before_print_type and is_leading_retract:
            continue

        cleaned.append(line)

    return cleaned
