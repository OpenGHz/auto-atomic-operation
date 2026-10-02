"""Config-loading utilities that read YAML/Hydra and emit plain Python types.

This module is the single boundary between OmegaConf/Hydra and the rest of
the codebase. Everything outside this module (and the runner entry-point
layer) operates on plain ``dict`` / ``list`` / Pydantic models, never on
``DictConfig`` / ``ListConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from auto_atom.config.task import AutoAtomConfig, TaskFileConfig

from .execution_config import prepare_task_config_for_instantiation


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


PRIMARY_CONFIG_NAME = "config"
"""The single Hydra entry config in ``aao_configs/``; tasks are a group in it."""


def task_overrides(task: str | None, overrides: list[str] | None = None) -> list[str]:
    """Prefix ``overrides`` with the ``task=<task>`` group selection."""
    selected = [f"task={task}"] if task else []
    return selected + list(overrides or [])


def compose_task_config(
    task: str | None = None,
    overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
    *,
    return_hydra_config: bool = False,
) -> DictConfig:
    """Compose the primary config for one task with Hydra's compose API.

    Parameters
    ----------
    task:
        Option of the ``task`` config group, e.g. ``"pick_and_place"``.
        ``None`` keeps the primary config's default task.
    overrides:
        Further Hydra overrides, e.g. ``["embodiment=xf9600_mocap",
        "render=gs", "task.seed=123"]``.
    config_dir:
        Absolute or relative path to the config directory.
        Defaults to ``<cwd>/aao_configs``.
    """
    from hydra import compose, initialize_config_dir

    resolved_dir = str(Path(config_dir or (Path.cwd() / "aao_configs")).resolve())
    with initialize_config_dir(config_dir=resolved_dir, version_base=None):
        return compose(
            config_name=PRIMARY_CONFIG_NAME,
            overrides=task_overrides(task, overrides),
            return_hydra_config=return_hydra_config,
        )


# Config groups that distinguish one run of a task from another. ``render``
# is only part of the name when it differs from native MuJoCo rendering.
_RUN_NAME_GROUPS = ("task", "embodiment", "render")


def describe_run(choices: Mapping[str, Any]) -> str:
    """Return a stable identifier for a composed run.

    Every runnable config is composed from ``aao_configs/config.yaml``, so the
    Hydra ``config_name`` no longer identifies a run. The selected ``task``,
    ``embodiment`` and non-default ``render`` choices do, e.g.
    ``pick_and_place__xf9600_mocap`` or ``press_blue_button__p7_g2p__gs``.
    Used to name recordings, comparison images and run summaries.
    """
    parts = []
    for group in _RUN_NAME_GROUPS:
        value = choices.get(group)
        if value is None or (group == "render" and value == "mujoco"):
            continue
        parts.append(str(value))
    return "__".join(parts)


def compose_task_run(
    task: str | None = None,
    overrides: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> tuple[DictConfig, str]:
    """Compose a task like :func:`compose_task_config` and name the run.

    Returns the composed config (without the ``hydra`` node) together with its
    :func:`describe_run` name, for scripts that write outputs per run.
    """
    cfg = compose_task_config(task, overrides, config_dir, return_hydra_config=True)
    run_name = describe_run(cfg.hydra.runtime.choices)
    with open_dict(cfg):
        del cfg["hydra"]
    return cfg, run_name


def load_task_file_hydra(
    task: str | None = None,
    config_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> TaskFileConfig:
    """Load a task file using Hydra's compose API (resolves config groups).

    Unlike :func:`load_task_file` which only reads a single YAML file, this
    composes ``aao_configs/config.yaml`` with ``task=<task>`` so every config
    group (embodiment, scene, render, ...) is merged in. See
    :func:`compose_task_config` for the parameters.
    """
    cfg = compose_task_config(task, overrides, config_dir)
    prepared = prepare_task_config_for_instantiation(cfg)
    instantiate(prepared)
    raw = OmegaConf.to_container(prepared, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")
    return TaskFileConfig.model_validate(raw)
