from dataclasses import dataclass, field
from typing import Callable, Self


@dataclass
class GcodeCommand:
    command: str = ""
    raw_args: list[str] = field(default_factory=list)
    args: dict[str, float] = field(default_factory=dict)
    comment: str | None = None

    @classmethod
    def parse(cls, line: str) -> Self:
        code, separator, raw_comment = line.strip().partition(";")
        tokens = code.split()
        raw_args = tokens[1:]
        return cls(
            command=tokens[0].upper() if tokens else "",
            raw_args=raw_args,
            args=parse_gcode_args(raw_args),
            comment=(raw_comment.strip() or None) if separator else None,
        )

    def build(self) -> str:
        code = " ".join([self.command, *self.raw_args]) if self.command else ""
        if self.comment is None:
            return code
        if code:
            return f"{code} ;{self.comment}"
        return f";{self.comment}"

    def build_with_updated_args(self, values: dict[str, float]) -> str:
        update_values = {key.upper(): value for key, value in values.items()}
        next_args = []

        for arg in self.raw_args:
            parsed_arg = parse_gcode_arg(arg)
            if parsed_arg is None:
                next_args.append(arg)
                continue

            key, _ = parsed_arg
            if key in update_values:
                next_args.append(f"{key}{update_values[key]}")
            else:
                next_args.append(arg)

        updated = GcodeCommand(
            command=self.command,
            raw_args=next_args,
            args=parse_gcode_args(next_args),
            comment=self.comment,
        )
        return updated.build()


def parse_gcode_arg(arg: str) -> tuple[str, float] | None:
    token = arg.strip()
    if len(token) < 2:
        return None

    key = token[0].upper()
    if not key.isalpha():
        return None

    try:
        return key, float(token[1:])
    except ValueError:
        return None


def parse_gcode_args(args: list[str]) -> dict[str, float]:
    parsed_args = {}
    for arg in args:
        parsed_arg = parse_gcode_arg(arg)
        if parsed_arg is None:
            continue
        key, value = parsed_arg
        parsed_args[key] = value
    return parsed_args


# See if necessary when we introduce time mashing
def is_comment_line(line: str, matches: Callable[[str | None], bool]) -> bool:
    parsed = GcodeCommand.parse(line)
    return not parsed.command and matches(parsed.comment)
