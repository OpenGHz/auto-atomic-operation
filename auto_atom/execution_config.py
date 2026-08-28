"""Hydra-boundary preparation for execution-mode-specific composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


def prepare_task_config_for_instantiation(cfg: DictConfig) -> DictConfig:
    """Return an isolated config tree ready for Hydra instantiation.

    ``object_only`` is resolved before Hydra constructs the environment so
    operator-owned MJCF layers and cameras never enter the simulation model.
    Runtime modules therefore receive one already-consistent object-only
    environment instead of hiding an operator after construction.
    """

    prepared = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if (
        OmegaConf.select(prepared, "execution.mode", default="physical")
        != "object_only"
    ):
        return prepared

    task_operators = OmegaConf.select(prepared, "task_operators", default={})
    env_operators = OmegaConf.select(prepared, "env.operators", default={})
    operator_names = {
        str(name)
        for mapping in (task_operators, env_operators)
        if isinstance(mapping, Mapping)
        for name in mapping
    }
    # Stage declarations are the authoritative consumer of an operator name,
    # even when a task file relies on a backend's implicit/default operator
    # registration and therefore omits that name from ``task_operators``.
    # Include them here so object-only mode also removes randomization entries
    # owned by those operators before the backend is instantiated.
    stages = OmegaConf.select(prepared, "task.stages", default=[])
    if stages is not None and not isinstance(stages, (str, bytes, Mapping)):
        operator_names.update(
            str(operator_name)
            for stage in stages
            for operator_name in [_mapping_value(stage, "operator", "")]
            if operator_name
        )

    layers = OmegaConf.select(prepared, "env.scene.layers", default=[])
    if layers is not None:
        retained_layers = [layer for layer in layers if not _is_operator_layer(layer)]
        OmegaConf.update(
            prepared,
            "env.scene.layers",
            retained_layers,
            merge=False,
            force_add=True,
        )

    cameras = OmegaConf.select(prepared, "env.cameras", default=[])
    removed_camera_names: set[str] = set()
    if cameras is not None:
        retained_cameras = []
        for camera in cameras:
            if _is_operator_camera(camera):
                name = _mapping_value(camera, "name", "")
                if name:
                    removed_camera_names.add(str(name))
                continue
            retained_cameras.append(camera)
        OmegaConf.update(
            prepared,
            "env.cameras",
            retained_cameras,
            merge=False,
            force_add=True,
        )
        enabled_sensors = OmegaConf.select(
            prepared,
            "env.enabled_sensors",
            default=[],
        )
        camera_enabled = "camera" in (enabled_sensors or []) and bool(retained_cameras)
        OmegaConf.update(
            prepared,
            "env.enabled_sensors",
            ["camera"] if camera_enabled else [],
            merge=False,
            force_add=True,
        )

    OmegaConf.update(prepared, "env.operators", {}, merge=False, force_add=True)
    OmegaConf.update(prepared, "task_operators", {}, merge=False, force_add=True)
    _drop_owned_mapping_entries(
        prepared,
        "task.randomization",
        operator_names,
    )
    _drop_owned_mapping_entries(
        prepared,
        "task.camera_randomization",
        removed_camera_names,
    )
    return prepared


def _mapping_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return value[key]
    except (KeyError, TypeError):
        return default


def _is_operator_layer(layer: Any) -> bool:
    """Recognize explicit roles and legacy robot-layer paths."""
    role = _mapping_value(layer, "role", None)
    if role is not None:
        return role == "operator"
    path = str(
        _mapping_value(layer, "path", _mapping_value(layer, "package", ""))
    ).replace("\\", "/")
    return "/robots/" in path or path.startswith("robots/")


def _is_operator_camera(camera: Any) -> bool:
    """Recognize explicit roles and legacy wrist/EEF camera names."""
    role = _mapping_value(camera, "role", None)
    if role is not None:
        return role == "operator"
    name = str(_mapping_value(camera, "name", "")).lower()
    return name == "wrist_cam" or name.startswith("eef_") or "wrist" in name


def _drop_owned_mapping_entries(
    cfg: DictConfig,
    path: str,
    owned_names: set[str],
) -> None:
    value = OmegaConf.select(cfg, path, default=None)
    if not isinstance(value, Mapping):
        return
    retained = {name: item for name, item in value.items() if name not in owned_names}
    OmegaConf.update(cfg, path, retained, merge=False, force_add=True)
