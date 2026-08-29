"""Regression tests for the execution and replay module seams."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def _imports(path: Path) -> set[str]:
    module_parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts
    package_parts = module_parts[:-1]
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module is not None:
                imports.add(node.module)
            continue
        prefix_parts = package_parts[: len(package_parts) - node.level + 1]
        if node.module is None:
            imports.update(
                ".".join((*prefix_parts, alias.name)) for alias in node.names
            )
        else:
            imports.add(".".join((*prefix_parts, node.module)))
    return imports


def test_execution_timeline_and_stage_execution_do_not_import_runtime() -> None:
    assert "auto_atom.runtime" not in _imports(
        PROJECT_ROOT / "auto_atom" / "execution_timeline.py"
    )
    assert "auto_atom.runtime" not in _imports(
        PROJECT_ROOT / "auto_atom" / "stage_execution.py"
    )


def test_replay_recording_does_not_import_runner_or_data_replay() -> None:
    imports = _imports(PROJECT_ROOT / "auto_atom" / "runner" / "replay_recording.py")
    assert "auto_atom.runner.data_replay" not in imports
    assert "auto_atom.runner" not in imports
