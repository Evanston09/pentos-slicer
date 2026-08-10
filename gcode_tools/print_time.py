_TIME_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
_ESTIMATE_PREFIX = "; estimated printing time (normal mode) = "


def parse_estimated_print_time(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith(_ESTIMATE_PREFIX):
            return (
                sum(
                    int(part[:-1]) * _TIME_UNITS[part[-1]]
                    for part in line.removeprefix(_ESTIMATE_PREFIX).split()
                )
                or None
            )
    return None


def format_print_time(seconds: int) -> str:
    parts = []
    for suffix, size in _TIME_UNITS.items():
        value, seconds = divmod(seconds, size)
        if value:
            parts.append(f"{value}{suffix}")
    return " ".join(parts) or "0s"
