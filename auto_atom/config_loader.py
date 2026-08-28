"""Config-loading utilities that read YAML/Hydra and emit plain Python types.

This module is the single boundary between OmegaConf/Hydra and the rest of
the codebase. Everything outside this module (and the runner entry-point
layer) operates on plain ``dict`` / ``list`` / Pydantic models, never on
``DictConfig`` / ``ListConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from .execution_config import prepare_task_config_for_instantiation
from .framework import AutoAtomConfig, TaskFileConfig


def to_plain(value: Any) -> Any:
    """Recursively convert ``DictConfig``/``ListConfig`` to plain Python types.

    Non-OmegaConf values (already-instantiated objects, primitives, plain
    dicts/lists) are returned unchanged.
    """
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    config = OmegaConf.load(Path(path))
    data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> AutoAtomConfig:
    return load_task_file(path).task


def load_task_file(path: str | Path) -> TaskFileConfig:
    config_path = Path(path)
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"YAML root must be a mapping: {config_path}")

    prepared = prepare_task_config_for_instantiation(config)
    instantiate(prepared)
    raw = OmegaConf.to_container(prepared, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping: {config_path}")
    return TaskFileConfig.model_validate(raw)


def load_task_file_hydra(
    config_name: str,
    config_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> TaskFileConfig:
    """Load a task file using Hydra compose API (supports ``defaults`` merging).

    Unlike :func:`load_task_file` which only reads a single YAML file,
    this function uses Hydra's compose API so that ``defaults`` lists are
    properly resolved and merged.

    Parameters
    ----------
    config_name:
        Name of the config (without ``.yaml`` suffix), e.g. ``"pick_and_place"``.
    config_dir:
        Absolute or relative path to the config directory.
        Defaults to ``<cwd>/aao_configs``.
    overrides:
        Optional Hydra override strings, e.g. ``["task.seed=123"]``.
    """
    from hydra import compose, initialize_config_dir

    resolved_dir = str(Path(config_dir or (Path.cwd() / "aao_configs")).resolve())
    with initialize_config_dir(config_dir=resolved_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    prepared = prepare_task_config_for_instantiation(cfg)
    instantiate(prepared)
    raw = OmegaConf.to_container(prepared, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")
    return TaskFileConfig.model_validate(raw)
