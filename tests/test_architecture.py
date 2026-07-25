import ast
from pathlib import Path


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_mvc_import_boundaries() -> None:
    root = Path(__file__).parents[1]

    for path in (root / "controllers").glob("*.py"):
        assert "viser" not in imported_modules(path)

    for path in (root / "views").glob("*.py"):
        assert not any(
            module == "services" or module.startswith("services.")
            for module in imported_modules(path)
        )

    for path in (root / "models").glob("*.py"):
        assert not any(
            module == blocked or module.startswith(f"{blocked}.")
            for blocked in (
                "controllers",
                "machine",
                "services",
                "views",
                "viser",
            )
            for module in imported_modules(path)
        )
