"""Hydra-boundary preparation for execution-mode-specific composition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def prepare_task_config_for_instantiation(cfg: DictConfig) -> DictConfig:
    """Return an isolated config tree ready for Hydra instantiation.

    Keyed ``env.cameras`` / ``env.scene.layers`` collections are first
    flattened into the ordered lists the environment expects (see
    :func:`normalize_env_collections`).

    ``object_only`` is resolved before Hydra constructs the environment so
    operator-owned MJCF layers and cameras never enter the simulation model.
    Runtime modules therefore receive one already-consistent object-only
    environment instead of hiding an operator after construction.

    In physical execution, the grasp-attachment welds that
    ``task_operators.<name>.control.grasp.attach`` needs are declared in
    ``env.grasp_attachments`` (see :func:`declare_grasp_attachments`), since
    they must be compiled into the scene.
    """

    prepared = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    normalize_env_collections(prepared)
    if (
        OmegaConf.select(prepared, "execution.mode", default="physical")
        != "object_only"
    ):
        declare_grasp_attachments(prepared)
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
        "task.randomization.entities",
        operator_names,
    )
    # Camera-owned entries go with their camera: a removed operator camera is no
    # longer in the model, so its randomization and initial-pose overrides would
    # otherwise resolve against a name the object-only scene does not have.
    _drop_owned_mapping_entries(
        prepared,
        "task.randomization.cameras",
        removed_camera_names,
    )
    _drop_owned_mapping_entries(
        prepared,
        "task.camera_initial_pose",
        removed_camera_names,
    )
    return prepared


# Operations whose EEF closes on the stage object.
_CLOSING_OPERATIONS = frozenset({"pick", "pull", "grasp"})


def declare_grasp_attachments(cfg: DictConfig) -> None:
    """Declare the welds that operators with ``grasp.attach`` need, in place.

    One ``env.grasp_attachments`` entry per (operator, stage object) the
    operator picks, pulls or grasps, welded to its EEF site
    (``env.operators.<name>.pose_site``, else ``eef_pose``). Entries the
    config already lists are kept and not repeated.
    """
    task_operators = OmegaConf.select(cfg, "task_operators", default={}) or {}
    env_operators = OmegaConf.select(cfg, "env.operators", default={}) or {}
    names = {
        str(name)
        for mapping in (task_operators, env_operators)
        if isinstance(mapping, Mapping)
        for name in mapping
    }
    stages = OmegaConf.select(cfg, "task.stages", default=[]) or []
    declared = list(OmegaConf.select(cfg, "env.grasp_attachments", default=[]) or [])
    seen = {
        (
            str(_mapping_value(entry, "operator", "")),
            str(_mapping_value(entry, "object", "")),
        )
        for entry in declared
    }
    added = False
    for stage in stages:
        operation = str(_mapping_value(stage, "operation", ""))
        object_name = str(_mapping_value(stage, "object", "") or "")
        operator = str(_mapping_value(stage, "operator", "") or "")
        if not operator and len(names) == 1:
            operator = next(iter(names))
        if operation not in _CLOSING_OPERATIONS or not object_name or not operator:
            continue
        attach = OmegaConf.select(
            cfg, f"task_operators.{operator}.control.grasp.attach", default=False
        )
        if not attach or (operator, object_name) in seen:
            continue
        frame = (
            OmegaConf.select(cfg, f"env.operators.{operator}.pose_site", default="")
            or "eef_pose"
        )
        declared.append({"operator": operator, "object": object_name, "frame": frame})
        seen.add((operator, object_name))
        added = True
    if added:
        OmegaConf.update(
            cfg, "env.grasp_attachments", declared, merge=False, force_add=True
        )


def normalize_env_collections(cfg: DictConfig) -> None:
    """Flatten keyed environment collections in place.

    Config groups declare ``env.scene.layers`` and ``env.cameras`` as mappings
    keyed by a slot name so a later config can replace one entry, or remove it
    with ``null``, without restating the whole list. Both become lists in slot
    insertion order with ``null`` slots dropped; list values pass through.

    Every camera then inherits ``env.camera_defaults`` (resolution and
    modalities) underneath its own fields, and cameras whose role is not in
    ``env.camera_roles`` are dropped. Both helper keys are consumed here.
    """

    env = cfg.get("env")
    if not isinstance(env, DictConfig):
        return

    scene = env.get("scene")
    if isinstance(scene, DictConfig) and "layers" in scene:
        scene.layers = _slots_to_list(scene.layers)

    camera_defaults = env.pop("camera_defaults", None)
    camera_roles = env.pop("camera_roles", None)
    if "cameras" not in env:
        return
    cameras = _slots_to_list(env.cameras)
    if cameras is None:
        return
    defaults = (
        OmegaConf.to_container(camera_defaults, resolve=True)
        if isinstance(camera_defaults, DictConfig)
        else {}
    )
    roles = (
        set(OmegaConf.to_container(camera_roles, resolve=True))
        if isinstance(camera_roles, ListConfig)
        else None
    )
    retained = []
    for camera in cameras:
        if roles is not None and _camera_role(camera) not in roles:
            continue
        retained.append(OmegaConf.merge(defaults, camera) if defaults else camera)
    env.cameras = retained


def _slots_to_list(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return [item for item in value.values() if item is not None]
    if isinstance(value, ListConfig):
        return [item for item in value if item is not None]
    return value


def _camera_role(camera: Any) -> str:
    if _is_operator_camera(camera):
        return "operator"
    return str(_mapping_value(camera, "role", None) or "scene")


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
