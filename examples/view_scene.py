"""Launch the interactive MuJoCo viewer on a composed scene+robot.

Since scene XMLs no longer embed their robot include or keyframe, opening
``demo.xml`` directly in the MuJoCo simulator shows just the empty scene.
This script reads a Hydra task config, compiles the ordered layers declared
under ``env.scene``, applies scene- and operator-owned initial joint positions,
and hands the model to ``mujoco.viewer.launch``.

When the config carries a ``gaussian_render`` section, the script switches
to a passive viewer and opens a second OpenCV window that re-renders the
scene with Gaussian Splatting from the same free-camera pose as the MuJoCo
viewer (synced live as you orbit / pan / zoom).

In GS mode, press ``R`` in either window or click the Reload button in the
GS window to re-read YAML, XML, and PLY files without restarting the Python
process.

Usage::

    python examples/view_scene.py --config-name pick_and_place
    python examples/view_scene.py --config-name open_door_airbot_play_gs
    python examples/view_scene.py --config-name open_door_p7_ik
    python examples/view_scene.py --debug --config-name open_door_p7_ik
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Callable, Mapping

import hydra
import mujoco
import mujoco.viewer
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from auto_atom.basis.mjc.model_initialization import apply_initial_joint_positions
from auto_atom.framework import PoseOverrideConfig, PoseReference
from auto_atom.runner.common import get_config_dir
from auto_atom.scene_composition import SceneConfig, load_composed_scene
from auto_atom.utils.pose import (
    PoseState,
    compose_pose,
    euler_to_quaternion,
    quaternion_from_matrix_3x3,
    resolve_pose_override,
)

_DEBUG = False


def _strip_debug_arg(argv: list[str]) -> bool:
    """Consume this script's --debug flag before Hydra parses argv."""
    debug = False
    stripped: list[str] = []
    hydra_separator_seen = False
    for arg in argv:
        if arg == "--":
            hydra_separator_seen = True
            stripped.append(arg)
        elif not hydra_separator_seen and arg == "--debug":
            debug = True
        else:
            stripped.append(arg)
    argv[:] = stripped
    return debug


def _print_debug_exception(context: str) -> None:
    print(f"[debug] {context} failed; full traceback:", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


def _to_container(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _orientation_to_wxyz(
    orientation: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Accept YAML orientation (4 floats xyzw or 3 floats Euler rpy) and
    return a wxyz quaternion suitable for MuJoCo body_quat / mocap_quat."""
    if len(orientation) == 4:
        x, y, z, w = (float(v) for v in orientation)
    elif len(orientation) == 3:
        x, y, z, w = euler_to_quaternion(tuple(float(v) for v in orientation))
    else:
        raise ValueError(
            f"orientation must be 3 floats (Euler) or 4 floats (xyzw quat), "
            f"got {len(orientation)}"
        )
    return np.array([w, x, y, z], dtype=np.float64)


def _find_freejoint_for_body(model: mujoco.MjModel, body_name: str) -> int:
    """Return the joint id of a freejoint directly attached to ``body_name``,
    or -1 if the body is static (no freejoint child)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return -1
    jnt_start = int(model.body_jntadr[bid])
    jnt_count = int(model.body_jntnum[bid])
    for j in range(jnt_start, jnt_start + jnt_count):
        if int(model.jnt_type[j]) == 0:  # mjJNT_FREE
            return j
    return -1


def _resolve_body_name(model: mujoco.MjModel, requested_name: str) -> str:
    """Resolve a logical object name to the body used by the composed scene.

    Gaussian-rendering scene layers commonly expose a visual ``<name>_gs``
    body alongside the logical object body.  The runtime object handler uses
    that body when it is present, so the standalone viewer follows the same
    convention.  Operator root bodies are passed through unchanged because
    their configured name is already a physical binding, not a logical object
    key.
    """
    gs_name = f"{requested_name}_gs"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs_name) >= 0:
        return gs_name
    return requested_name


def _set_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    position: list[float] | None,
    orientation: list[float] | None,
) -> bool:
    """Apply a world-frame position/orientation override to one body.

    Freejoint qpos is already represented in world coordinates.  Static body
    transforms, however, are stored in the parent body's local frame; convert
    them explicitly so nested scene assets and operator roots behave exactly
    like the backend object/base-pose paths.  ``mj_forward`` is run here so a
    subsequent named-frame override observes the newly applied pose.
    """
    resolved_name = body_name
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, resolved_name)
    if bid < 0:
        print(f"[warn] body '{body_name}' not found; skipping initial_pose")
        return False
    free_jid = _find_freejoint_for_body(model, resolved_name)
    if free_jid >= 0:
        addr = int(model.jnt_qposadr[free_jid])
        if position is not None:
            data.qpos[addr : addr + 3] = [float(v) for v in position[:3]]
        if orientation is not None:
            data.qpos[addr + 3 : addr + 7] = _orientation_to_wxyz(orientation)
        dof_addr = int(model.jnt_dofadr[free_jid])
        data.qvel[dof_addr : dof_addr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return True

    # Static body: body_pos/body_quat are parent-local, while the override
    # contract is world-frame after reference resolution.
    parent_id = int(model.body_parentid[bid])
    parent_pos = np.asarray(data.xpos[parent_id], dtype=np.float64)
    parent_rot = np.asarray(data.xmat[parent_id], dtype=np.float64).reshape(3, 3)
    if position is not None:
        world_pos = np.asarray(position[:3], dtype=np.float64)
        model.body_pos[bid] = parent_rot.T @ (world_pos - parent_pos)
    if orientation is not None:
        world_quat = _orientation_to_wxyz(orientation)
        parent_quat = np.asarray(data.xquat[parent_id], dtype=np.float64)
        inverse_parent_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_negQuat(inverse_parent_quat, parent_quat)
        local_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(local_quat, inverse_parent_quat, world_quat)
        model.body_quat[bid] = local_quat
    mujoco.mj_forward(model, data)
    return True


def _set_camera_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    pose: PoseState,
) -> bool:
    """Apply one world-frame camera pose as parent-local MuJoCo extrinsics."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        print(
            f"[warn] camera '{camera_name}' not found; skipping camera_initial_pose",
            flush=True,
        )
        return False

    mujoco.mj_forward(model, data)
    parent_id = int(model.cam_bodyid[camera_id])
    parent_pos = np.asarray(data.xpos[parent_id], dtype=np.float64)
    parent_rot = np.asarray(data.xmat[parent_id], dtype=np.float64).reshape(3, 3)
    world_pos = np.asarray(pose.position[0], dtype=np.float64)
    model.cam_pos[camera_id] = parent_rot.T @ (world_pos - parent_pos)

    qx, qy, qz, qw = (float(value) for value in pose.orientation[0])
    world_quat_wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    parent_quat_wxyz = np.asarray(data.xquat[parent_id], dtype=np.float64)
    inverse_parent_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_negQuat(inverse_parent_quat, parent_quat_wxyz)
    local_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(local_quat, inverse_parent_quat, world_quat_wxyz)
    model.cam_quat[camera_id] = local_quat
    mujoco.mj_forward(model, data)
    return True


def _initial_pose_order(
    overrides: Mapping[str, PoseOverrideConfig],
) -> list[str]:
    """Return object pose keys in dependency order and reject cycles."""
    names = list(overrides)
    declaration_index = {name: index for index, name in enumerate(names)}
    dependencies: dict[str, set[str]] = {name: set() for name in names}
    for name, config in overrides.items():
        for reference in config.axis_references():
            if (
                isinstance(reference, str)
                and not isinstance(reference, PoseReference)
                and reference in dependencies
            ):
                dependencies[name].add(reference)

    order: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"Circular initial pose reference involving {name!r}")
        visiting.add(name)
        for dependency in sorted(dependencies[name], key=declaration_index.__getitem__):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in names:
        visit(name)
    return order


def _element_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> PoseState:
    """Return a world pose for a MuJoCo site/body/geom/joint."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        return PoseState(
            position=data.site_xpos[sid],
            orientation=quaternion_from_matrix_3x3(data.site_xmat[sid].reshape(3, 3)),
        )
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        return PoseState(
            position=data.xpos[bid],
            orientation=quaternion_from_matrix_3x3(data.xmat[bid].reshape(3, 3)),
        )
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid >= 0:
        return PoseState(
            position=data.geom_xpos[gid],
            orientation=quaternion_from_matrix_3x3(data.geom_xmat[gid].reshape(3, 3)),
        )
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        # xanchor/xmat are the articulated joint frame after ``mj_forward``;
        # parent-local body data would describe only the XML default pose.
        body_id = int(model.jnt_bodyid[jid])
        return PoseState(
            position=data.xanchor[jid],
            orientation=quaternion_from_matrix_3x3(data.xmat[body_id].reshape(3, 3)),
        )
    raise KeyError(
        f"No site, body, geom, or joint named '{name}' found in the MuJoCo model"
    )


def _resolve_viewer_override(
    raw_config: (
        PoseOverrideConfig | Mapping[str, object] | list[float] | tuple[float, ...]
    ),
    fallback: PoseState,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    context: str,
    operator_frames: Mapping[str, Mapping[str, str]] | None = None,
    allow_operator_aliases: bool = False,
) -> PoseState:
    """Resolve a shared pose override for the standalone viewer.

    The resolver is intentionally the same ``resolve_pose_override`` used by
    the runtime backend.  This function only supplies the MuJoCo-specific
    reference-frame lookup (site/body/geom/joint and ``operator.base/eef``).
    ``operator_frames`` is optional so existing callers and small unit-test
    doubles can continue to resolve plain scene references.
    """
    if isinstance(raw_config, PoseOverrideConfig):
        config = raw_config
    elif isinstance(raw_config, (list, tuple)):
        # The only legacy list form is the operator EEF pose.  Keep accepting
        # tuples here because OmegaConf and test doubles may materialize them
        # differently; the shared resolver consumes the canonical list form.
        config = list(raw_config)
    else:
        config = PoseOverrideConfig.model_validate(raw_config)
    reference = PoseReference.WORLD if isinstance(config, list) else config.reference

    def resolve_reference(reference_value: PoseReference | str) -> PoseState:
        if isinstance(reference_value, PoseReference):
            if reference_value != PoseReference.WORLD:
                raise ValueError(
                    f"{context} reference {reference_value.value!r} is not supported by "
                    "the standalone viewer; use 'world' or a named scene element"
                )
            return PoseState()
        try:
            operator_reference = None
            if (
                allow_operator_aliases
                and operator_frames is not None
                and "." in reference_value
            ):
                operator_name, attribute = reference_value.rsplit(".", 1)
                frame = operator_frames.get(operator_name)
                if frame is not None and attribute in {"base", "eef"}:
                    element_name = frame.get(attribute)
                    if element_name:
                        operator_reference = _element_pose(model, data, element_name)
            if operator_reference is not None:
                reference_pose = operator_reference
            else:
                try:
                    reference_pose = _element_pose(model, data, reference_value)
                except KeyError:
                    # Match the runtime object's logical-name resolution for
                    # Gaussian scenes, where ``name_gs`` is the physical
                    # visual body backing a logical ``name`` key.
                    resolved_reference = _resolve_body_name(model, reference_value)
                    reference_pose = _element_pose(model, data, resolved_reference)
        except KeyError as exc:
            raise ValueError(
                f"{context} reference {reference_value!r} is not a scene element"
            ) from exc
        return reference_pose

    if isinstance(config, list):
        reference_poses = {PoseReference.WORLD: PoseState()}
    else:
        reference_poses = {
            reference_value: resolve_reference(reference_value)
            for reference_value in config.axis_references()
        }
    return resolve_pose_override(
        config,
        fallback,
        reference_poses.get(reference),
        reference_poses,
    )


def _apply_home_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_joint_positions: dict,
    actuator_names: list[str] | None = None,
) -> None:
    """Apply the runtime home-state contract before launching the viewer."""
    actuator_ids: list[int] = []
    for actuator_name in actuator_names or ():
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if actuator_id < 0:
            print(f"[warn] actuator '{actuator_name}' not found; skipping")
            continue
        actuator_ids.append(int(actuator_id))

    missing_joint_names = apply_initial_joint_positions(
        model,
        data,
        initial_joint_positions,
        actuator_ids,
    )
    for joint_name in missing_joint_names:
        print(f"[warn] joint '{joint_name}' not found; skipping")

    _sync_mocap_bodies(model, data)


def _sync_mocap_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Align mocap weld targets with their physical bodies.

    The helper is called both after the joint home state and after any later
    root-body pose override.  Keeping it separate prevents a base relocation
    from being pulled back toward the stale mocap target on the first viewer
    step.
    """
    for eq in range(model.neq):
        if int(model.eq_type[eq]) != 1:  # mjEQ_WELD
            continue
        b1, b2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        m1, m2 = int(model.body_mocapid[b1]), int(model.body_mocapid[b2])
        if m1 >= 0 and m2 < 0:
            mocap_id, phys_id = m1, b2
        elif m2 >= 0 and m1 < 0:
            mocap_id, phys_id = m2, b1
        else:
            continue
        data.mocap_pos[mocap_id] = data.xpos[phys_id].copy()
        data.mocap_quat[mocap_id] = data.xquat[phys_id].copy()


def _eef_joint_home(
    model: mujoco.MjModel,
    operator_name: str,
    actuator_name: str,
    value: float,
) -> tuple[str, float] | None:
    """Resolve one scalar EEF home command to its joint qpos target."""

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        print(
            f"[warn] task_operators.{operator_name}.initial_state.eef actuator "
            f"{actuator_name!r} is not present in the model; skipping",
            flush=True,
        )
        return None
    transmission = int(model.actuator_trntype[actuator_id])
    if transmission not in {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    }:
        print(
            f"[warn] task_operators.{operator_name}.initial_state.eef uses "
            f"non-joint actuator {actuator_name!r}; view_scene cannot apply it "
            "without the runtime operator mapper",
            flush=True,
        )
        return None
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is None:
        return None
    return joint_name, float(value)


def _extract_overrides(cfg: DictConfig) -> dict:
    """Extract viewer-relevant overrides from a freshly composed Hydra cfg.

    Pose values are validated into the shared :class:`PoseOverrideConfig`
    model here, rather than maintaining a viewer-only schema.  Operator frame
    metadata is retained for resolving ``<operator>.base`` / ``.eef`` named
    references at build time.  EEF initial poses are reported explicitly but
    not applied: a standalone viewer has no backend IK/controller to turn a
    world pose into fixed-arm joint qpos.  Camera initial poses are applied
    directly because they are model-level extrinsics.
    """
    env_cfg = cfg.env
    scene_data = _to_container(env_cfg.get("scene")) or {}
    operators_cfg = _to_container(env_cfg.get("operators")) or {}
    actuator_names = list(
        dict.fromkeys(
            str(actuator_name)
            for operator in operators_cfg.values()
            for field in ("arm_actuators", "eef_actuators")
            for actuator_name in (operator.get(field) or [])
        )
    )
    sim_freq_raw = env_cfg.get("sim_freq")
    raw_initial_poses = _to_container(cfg.get("task", {}).get("initial_pose")) or {}
    initial_poses: dict[str, PoseOverrideConfig] = {}
    for body_name, raw_override in raw_initial_poses.items():
        if raw_override is None:
            continue
        initial_poses[str(body_name)] = PoseOverrideConfig.model_validate(raw_override)

    raw_camera_initial_poses = (
        _to_container(cfg.get("task", {}).get("camera_initial_pose")) or {}
    )
    camera_initial_poses: dict[str, PoseOverrideConfig] = {}
    for camera_name, raw_override in raw_camera_initial_poses.items():
        if raw_override is None:
            continue
        camera_initial_poses[str(camera_name)] = PoseOverrideConfig.model_validate(
            raw_override
        )

    operator_frames: dict[str, dict[str, str]] = {}
    for op_name, operator in operators_cfg.items():
        if not isinstance(operator, Mapping):
            continue
        frame: dict[str, str] = {}
        root_body = operator.get("root_body")
        pose_site = operator.get("pose_site")
        if root_body:
            frame["base"] = str(root_body)
        if pose_site:
            frame["eef"] = str(pose_site)
        if frame:
            operator_frames[str(op_name)] = frame

    out: dict = {
        "scene": scene_data,
        "scene_base": str(scene_data["base"]),
        "mjcf_paths": [
            str(layer.get("path"))
            for layer in scene_data.get("layers", [])
            if isinstance(layer, dict) and layer.get("kind") == "mjcf"
        ],
        "sim_freq": float(sim_freq_raw) if sim_freq_raw is not None else None,
        "actuator_names": actuator_names,
        "ijp": _to_container(env_cfg.get("initial_joint_positions")) or {},
        "initial_pose": initial_poses,
        "camera_initial_pose": camera_initial_poses,
        "op_bases": [],
        "op_joint_homes": [],
        "op_eef_homes": [],
        "op_eef_poses": [],
        "operator_frames": operator_frames,
    }

    task_operators_cfg = cfg.get("task_operators") or {}
    items = list(task_operators_cfg.items()) if task_operators_cfg else []
    for op_name, op_node in items:
        initial_state = op_node.get("initial_state") or {}
        joint_positions = _to_container(initial_state.get("joint_positions")) or {}
        if joint_positions:
            duplicates = set(out["ijp"]) & set(joint_positions)
            if duplicates:
                raise ValueError(
                    "Initial joint positions are declared in both "
                    "env.initial_joint_positions and "
                    f"task_operators.{op_name}.initial_state.joint_positions: "
                    f"{sorted(duplicates)}"
                )
            operator_cfg = operators_cfg.get(op_name) or {}
            arm_actuators = [
                str(name) for name in operator_cfg.get("arm_actuators") or ()
            ]
            out["op_joint_homes"].append(
                (str(op_name), dict(joint_positions), arm_actuators)
            )
        eef_value = initial_state.get("eef")
        if eef_value is not None:
            operator_cfg = operators_cfg.get(op_name) or {}
            eef_actuators = [
                str(name) for name in operator_cfg.get("eef_actuators") or ()
            ]
            if not eef_actuators:
                print(
                    f"[warn] task_operators.{op_name}.initial_state.eef is set "
                    "but the operator has no eef_actuators; skipping",
                    flush=True,
                )
            else:
                out["op_eef_homes"].append(
                    (str(op_name), eef_actuators[0], float(eef_value))
                )
        bp_raw = _to_container(initial_state.get("base_pose"))
        if bp_raw is not None:
            bp_cfg = PoseOverrideConfig.model_validate(bp_raw)
        else:
            bp_cfg = None
        if bp_cfg is None:
            continue
        root_body = (operators_cfg.get(op_name) or {}).get("root_body")
        if not root_body:
            print(
                f"[warn] task_operators.{op_name}.initial_state.base_pose set "
                f"but env.operators.{op_name}.root_body is empty; skipping"
            )
            continue
        out["op_bases"].append((root_body, bp_cfg))

    # The runtime backend can solve an initial EEF pose (and may use a
    # reference such as ``base``), but this script only has a raw MjModel/Data
    # pair.  Keep the configured values visible in diagnostics and make the
    # limitation explicit instead of silently showing a different home pose.
    for op_name, op_node in items:
        initial_state = op_node.get("initial_state") or {}
        eef_raw = _to_container(initial_state.get("eef_pose"))
        if eef_raw is None:
            continue
        out["op_eef_poses"].append((str(op_name), eef_raw))
        print(
            f"[warn] task_operators.{op_name}.initial_state.eef_pose is "
            "configured but view_scene cannot apply EEF poses without the "
            "runtime backend IK/controller; showing the model/keyframe EEF "
            "pose instead",
            flush=True,
        )
    return out


def _build(overrides: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = load_composed_scene(SceneConfig.model_validate(overrides["scene"]))
    sim_freq = overrides.get("sim_freq")
    if sim_freq is not None:
        if sim_freq <= 0:
            raise ValueError(f"env.sim_freq must be positive, got {sim_freq}")
        m.opt.timestep = 1.0 / sim_freq
    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)
    # ``mj_resetData`` initializes qpos/qvel but does not guarantee that the
    # derived world transforms (xpos/xmat/site_xpos/...) are current.  Pose
    # overrides use those transforms as fallbacks and named references, so
    # establish a valid snapshot before resolving the first entry.
    mujoco.mj_forward(m, d)

    # 1) env.initial_joint_positions — mirror ``MujocoBasis.reset()`` before
    # resolving any pose override.  In particular, mocap operators expose a
    # freejoint in this map; applying it after a base override would silently
    # undo the requested root pose.
    _apply_home_pose(m, d, overrides["ijp"], overrides.get("actuator_names", []))

    # 2) task.initial_pose — resolve and apply object poses.  Named
    # references are sampled against the current scene state, so each update
    # is forwarded before the next override (matching backend reset order).
    initial_poses = overrides["initial_pose"]
    for requested_name in _initial_pose_order(initial_poses):
        override = initial_poses[requested_name]
        if override is None:
            continue
        body_name = _resolve_body_name(m, requested_name)
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
            print(
                f"[warn] body '{requested_name}' not found; skipping initial_pose",
                flush=True,
            )
            continue
        fallback = _element_pose(m, d, body_name)
        resolved = _resolve_viewer_override(
            override,
            fallback,
            m,
            d,
            context=f"initial_pose[{requested_name!r}]",
            operator_frames=None,
        )
        _set_body_pose(
            m,
            d,
            body_name,
            resolved.position[0],
            resolved.orientation[0],
        )

    # 3) task_operators.*.initial_state.base_pose — resolve only after all
    # object initial poses have been applied, so a reference such as a handle
    # site observes the final object placement.
    for root_body, base_override in overrides["op_bases"]:
        if base_override is None:
            continue
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_body) < 0:
            print(
                f"[warn] operator root body '{root_body}' not found; "
                "skipping initial_state.base_pose",
                flush=True,
            )
            continue
        fallback = _element_pose(m, d, root_body)
        resolved = _resolve_viewer_override(
            base_override,
            fallback,
            m,
            d,
            context=f"operator root body {root_body!r} base_pose",
            operator_frames=overrides.get("operator_frames"),
            allow_operator_aliases=True,
        )
        _set_body_pose(
            m,
            d,
            root_body,
            resolved.position[0],
            resolved.orientation[0],
        )

    # 4) task_operators.*.initial_state.joint_positions — mirror the backend's
    # operator-owned home pass after base-pose resolution.  Keeping this
    # separate from env.initial_joint_positions preserves the same ownership
    # and reset order used by MujocoTaskBackend.
    operator_joint_positions: dict[str, object] = {}
    operator_actuator_names: list[str] = []
    for _op_name, joint_positions, actuator_names in overrides.get(
        "op_joint_homes", []
    ):
        operator_joint_positions.update(joint_positions)
        operator_actuator_names.extend(actuator_names)
    for op_name, actuator_name, value in overrides.get("op_eef_homes", []):
        resolved = _eef_joint_home(m, op_name, actuator_name, value)
        if resolved is not None:
            joint_name, joint_value = resolved
            operator_joint_positions[joint_name] = joint_value
            operator_actuator_names.append(actuator_name)
    if operator_joint_positions:
        _apply_home_pose(
            m,
            d,
            operator_joint_positions,
            operator_actuator_names,
        )

    # A base override may have moved a physical body that is welded to a
    # mocap target.  Re-sync after the relocation so the first viewer step
    # preserves the configured base rather than applying a corrective weld
    # impulse toward the pre-override home pose.
    _sync_mocap_bodies(m, d)

    # 5) task.camera_initial_pose — cameras are model-level extrinsics, so the
    # standalone viewer can apply the same world-frame contract directly.
    # Resolve after object/base overrides so named scene references observe the
    # final composed placement.
    for camera_name, camera_override in overrides.get(
        "camera_initial_pose", {}
    ).items():
        if camera_override is None:
            continue
        camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            print(
                f"[warn] camera '{camera_name}' not found; skipping camera_initial_pose",
                flush=True,
            )
            continue
        fallback = PoseState(
            position=d.cam_xpos[camera_id],
            orientation=quaternion_from_matrix_3x3(d.cam_xmat[camera_id].reshape(3, 3)),
        )
        resolved = _resolve_viewer_override(
            camera_override,
            fallback,
            m,
            d,
            context=f"camera_initial_pose[{camera_name!r}]",
            operator_frames=None,
        )
        _set_camera_pose(m, d, camera_name, resolved)

    return m, d


def _compose_config_from_disk(
    config_dir: str,
    config_name: str,
    cli_overrides: list[str] | None = None,
) -> DictConfig:
    """Re-compose the Hydra config from disk for viewer reloads."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=cli_overrides or [])


def _load_reloaded_scene(
    config_dir: str,
    config_name: str,
    cli_overrides: list[str] | None = None,
) -> tuple[DictConfig, dict, mujoco.MjModel, mujoco.MjData]:
    cfg_now = _compose_config_from_disk(config_dir, config_name, cli_overrides)
    overrides = _extract_overrides(cfg_now)
    m, d = _build(overrides)
    return cfg_now, overrides, m, d


def _print_model_summary(model: mujoco.MjModel, overrides: dict) -> None:
    print(
        f"[info] model  : nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nbody={model.nbody} ngeom={model.ngeom}  "
        f"(mjcf={overrides['mjcf_paths']}, ijp={len(overrides['ijp'])}, "
        f"body_pose={len(overrides['initial_pose'])}, "
        f"op_base={len(overrides['op_bases'])}, "
        f"op_joint_home={len(overrides.get('op_joint_homes', []))}, "
        f"op_eef_home={len(overrides.get('op_eef_homes', []))}, "
        f"camera_pose={len(overrides.get('camera_initial_pose', {}))})"
    )


def _print_gs_summary(gs_cfg) -> None:
    print(
        f"[info] gs     : {len(gs_cfg.body_gaussians)} body gaussian(s), "
        f"background_ply={'list/glob' if gs_cfg.is_multi_background() else gs_cfg.background_ply!r}"
    )


def _maybe_gs_config(cfg: DictConfig):
    """Return a ``GaussianRenderConfig`` if the task config requests GS rendering.

    Detection is content-based (looks for ``env.gaussian_render`` with at least
    one body gaussian or a background ply), so it works whether the task uses
    the GS env target directly or composes GS into a non-GS env.
    """
    env_cfg = cfg.get("env", {})
    gs_node = env_cfg.get("gaussian_render", None)
    if gs_node is None:
        return None
    gs_dict = OmegaConf.to_container(gs_node, resolve=True) or {}
    if not (gs_dict.get("body_gaussians") or gs_dict.get("background_ply")):
        return None
    from auto_atom.basis.mjc.gs_mujoco_env import GaussianRenderConfig

    return GaussianRenderConfig.model_validate(gs_dict)


def _build_gs_renderer(gs_cfg, model: mujoco.MjModel):
    """Build a ``GSRendererMuJoCo`` covering body PLYs + a single background.

    Multi-background configs (list / glob) only get their first entry here —
    the viewer is for previewing geometry alignment, not for sweeping bgs.
    """
    from gaussian_renderer import GSRendererMuJoCo

    combined = dict(gs_cfg.resolved_body_gaussians())
    if gs_cfg.is_multi_background():
        bgs = gs_cfg.resolved_background_plys()
        if bgs:
            combined["background"] = bgs[0]
    else:
        bg = gs_cfg.resolved_background_ply()
        if bg:
            combined["background"] = bg
    return GSRendererMuJoCo(combined, model)


def _run_gs_synced_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gs_cfg,
    reload_callback: Callable[
        [], tuple[DictConfig, dict, mujoco.MjModel, mujoco.MjData]
    ]
    | None = None,
    width: int = 640,
    height: int = 480,
) -> None:
    """Passive MuJoCo viewer + cv2 window showing the GS render of the same
    free-camera pose, refreshed every step."""
    import cv2
    import torch

    win = "GS view (synced with MuJoCo viewer)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, width, height)

    def reload_button_rect() -> tuple[int, int, int, int]:
        button_width = 142
        button_height = 36
        x1 = width - 18
        y0 = 18
        return max(18, x1 - button_width), y0, x1, y0 + button_height

    def draw_reload_button(frame: np.ndarray) -> None:
        if reload_callback is None:
            return
        x0, y0, x1, y1 = reload_button_rect()
        cv2.rectangle(frame, (x0, y0), (x1, y1), (42, 112, 170), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 210, 255), 1)
        cv2.putText(
            frame,
            "Reload (R)",
            (x0 + 16, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    def show_status(title: str, lines: list[str]) -> None:
        frame = np.full((height, width, 3), 24, dtype=np.uint8)
        accent = (80, 180, 255)
        text = (235, 235, 235)
        muted = (170, 170, 170)
        cv2.putText(
            frame,
            title,
            (28, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            accent,
            2,
            cv2.LINE_AA,
        )
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (28, 104 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                text if idx == 0 else muted,
                1,
                cv2.LINE_AA,
            )
        draw_reload_button(frame)
        try:
            cv2.imshow(win, frame)
            cv2.waitKey(1)
        except cv2.error:
            # Window was closed (X) — silently skip; outer loop will detect
            # via getWindowProperty and exit cleanly.
            pass

    reload_event = threading.Event()
    startup_start = time.perf_counter()
    first_visible_frame = False
    render_attempts = 0
    last_black_notice = 0.0

    def reset_warmup_state() -> None:
        nonlocal startup_start, first_visible_frame, render_attempts, last_black_notice
        startup_start = time.perf_counter()
        first_visible_frame = False
        render_attempts = 0
        last_black_notice = 0.0

    def on_mouse(event, x: int, y: int, _flags, _param) -> None:
        if reload_callback is None or event != cv2.EVENT_LBUTTONUP:
            return
        x0, y0, x1, y1 = reload_button_rect()
        if x0 <= x <= x1 and y0 <= y <= y1:
            print("[reload] requested from GS window button", flush=True)
            reload_event.set()

    cv2.setMouseCallback(win, on_mouse)

    def build_renderer_with_status(current_gs_cfg, current_model: mujoco.MjModel):
        show_status(
            "Loading Gaussian renderer...",
            [
                "Reading Gaussian PLY files and preparing GPU resources.",
                "The first render can take a few seconds; this is expected.",
                "Terminal logs will report when GS is ready.",
            ],
        )
        print(
            "[info] GS renderer: loading Gaussian PLYs; first render may take "
            "a few seconds...",
            flush=True,
        )
        load_start = time.perf_counter()
        renderer = _build_gs_renderer(current_gs_cfg, current_model)
        print(
            f"[info] GS renderer: loaded in {time.perf_counter() - load_start:.1f}s; "
            "warming up first frame...",
            flush=True,
        )
        show_status(
            "Warming up GS render...",
            [
                "Waiting for the first visible GS frame.",
                "If the renderer returns a black warmup frame, it is hidden here.",
                "Press ESC to exit; click Reload or press R after edits.",
            ],
        )
        reset_warmup_state()
        return renderer

    def key_callback(key: int) -> None:
        if reload_callback is not None and key in (ord("R"), ord("r")):
            reload_event.set()

    gs_renderer = build_renderer_with_status(gs_cfg, model)
    print(
        "[info] GS sync: orbit/pan/zoom in the MuJoCo viewer to drive the GS"
        f" view (size {width}x{height}). ESC in the GS window to close it;"
        " close the MuJoCo viewer to exit."
        + (
            " Press R in either window or click Reload in the GS window "
            "to re-read YAML/XML/PLY. The MuJoCo viewer window will reopen "
            "after reload."
            if reload_callback is not None
            else ""
        )
    )

    def clone_camera(camera) -> mujoco.MjvCamera:
        cloned = mujoco.MjvCamera()
        cloned.type = camera.type
        cloned.fixedcamid = camera.fixedcamid
        cloned.trackbodyid = camera.trackbodyid
        cloned.lookat[:] = camera.lookat
        cloned.distance = camera.distance
        cloned.azimuth = camera.azimuth
        cloned.elevation = camera.elevation
        return cloned

    def restore_camera(target, source: mujoco.MjvCamera) -> None:
        target.type = source.type
        target.fixedcamid = source.fixedcamid
        target.trackbodyid = source.trackbodyid
        target.lookat[:] = source.lookat
        target.distance = source.distance
        target.azimuth = source.azimuth
        target.elevation = source.elevation

    def wait_for_viewer_exit(v) -> None:
        deadline = time.perf_counter() + 2.0
        sim_ref = getattr(v, "_sim", None)
        while time.perf_counter() < deadline:
            if sim_ref is None or sim_ref() is None:
                return
            time.sleep(0.01)

    def load_reloaded_scene_with_status():
        if reload_callback is None:
            raise RuntimeError("GS reload is not enabled")
        show_status(
            "Reloading GS scene...",
            [
                "Re-reading YAML, XML, and Gaussian PLY files.",
                "The MuJoCo viewer is restarted to avoid passive reload races.",
                "Keep this GS window open; errors will appear here.",
            ],
        )
        cfg_now, overrides_now, new_model, new_data = reload_callback()
        new_gs_cfg = _maybe_gs_config(cfg_now)
        if new_gs_cfg is None:
            raise RuntimeError("reloaded config no longer defines env.gaussian_render")
        new_renderer = build_renderer_with_status(new_gs_cfg, new_model)
        return cfg_now, overrides_now, new_model, new_data, new_gs_cfg, new_renderer

    camera_state: mujoco.MjvCamera | None = None
    while True:
        restart_requested = False
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback if reload_callback is not None else None,
        ) as v:
            if camera_state is not None:
                with v.lock():
                    restore_camera(v.cam, camera_state)
                v.sync(state_only=True)

            while v.is_running():
                step_start = time.time()
                if reload_event.is_set():
                    reload_event.clear()
                    if reload_callback is not None:
                        with v.lock():
                            camera_state = clone_camera(v.cam)
                        restart_requested = True
                        print(
                            "[reload] closing MuJoCo viewer before rebuilding scene...",
                            flush=True,
                        )
                        show_status(
                            "Reloading GS scene...",
                            [
                                "Closing the MuJoCo viewer before scene rebuild.",
                                "This avoids passive-viewer scene/frustum races.",
                                "The viewer will reopen automatically.",
                            ],
                        )
                        v.close()
                        break
                with v.lock():
                    mujoco.mj_step(model, data)
                    camera_for_render = clone_camera(v.cam)
                v.sync()
                try:
                    render_attempts += 1
                    gs_renderer.update_gaussians(data)
                    results = gs_renderer.render(
                        model, data, [-1], width, height, free_camera=camera_for_render
                    )
                    rgb_t, _depth = results[-1]
                    rgb = rgb_t.clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if not first_visible_frame and int(rgb.max(initial=0)) <= 1:
                        now = time.perf_counter()
                        if now - last_black_notice >= 2.0:
                            last_black_notice = now
                            print(
                                "[info] GS warmup: renderer returned a black frame; "
                                "keeping the loading screen visible...",
                                flush=True,
                            )
                        show_status(
                            "Warming up GS render...",
                            [
                                f"Black warmup frame hidden (attempt {render_attempts}).",
                                "This can happen while CUDA kernels or PLY data settle.",
                                "If it persists, check GS paths/camera alignment or press R.",
                            ],
                        )
                    else:
                        if not first_visible_frame:
                            first_visible_frame = True
                            print(
                                "[info] GS ready: first visible frame after "
                                f"{time.perf_counter() - startup_start:.1f}s "
                                f"({render_attempts} render attempt(s)).",
                                flush=True,
                            )
                        draw_reload_button(bgr)
                        cv2.imshow(win, bgr)
                except Exception as e:
                    if _DEBUG:
                        _print_debug_exception("GS render")
                    else:
                        print(f"[warn] GS render error: {e}")
                    show_status(
                        "GS render error",
                        [
                            str(e)[:78],
                            "The viewer will retry on the next frame.",
                            "Use --debug for a full traceback.",
                        ],
                    )
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    v.close()
                    break  # ESC exits both windows
                if reload_callback is not None and key in (ord("R"), ord("r")):
                    reload_event.set()
                # Detect the user clicking the X on the cv2 window. After the
                # Qt backend destroys the window, ``getWindowProperty`` itself
                # raises ``NULL guiReceiver`` instead of returning <1, so treat
                # any error here as "window gone".
                try:
                    win_visible = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    win_visible = 0.0
                if win_visible < 1:
                    v.close()
                    break  # GS window closed via X
                elapsed = time.time() - step_start
                sleep_for = float(model.opt.timestep) - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        wait_for_viewer_exit(v)
        if not restart_requested:
            break
        try:
            print("[reload] re-reading YAML/XML/PLY...", flush=True)
            (
                _cfg_now,
                overrides_now,
                new_model,
                new_data,
                new_gs_cfg,
                new_renderer,
            ) = load_reloaded_scene_with_status()
            model, data, gs_cfg, gs_renderer = (
                new_model,
                new_data,
                new_gs_cfg,
                new_renderer,
            )
            _print_model_summary(model, overrides_now)
            _print_gs_summary(gs_cfg)
            print("[reload] done; reopening MuJoCo viewer", flush=True)
        except Exception as e:
            if _DEBUG:
                _print_debug_exception("GS reload")
            else:
                print(f"[warn] GS reload error: {e}")
            show_status(
                "GS reload failed",
                [
                    str(e)[:78],
                    "Old scene will reopen; fix the error, then reload again.",
                    "Use --debug for a full traceback.",
                ],
            )
            print("[reload] keeping previous scene", flush=True)
        reload_event.clear()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Recover the absolute config directory + config name from HydraConfig so
    # the loader can re-compose the same task on every reload click. (Hydra
    # changes cwd inside @hydra.main, so we can't rely on get_config_dir().)
    hc = HydraConfig.get()
    config_name = hc.job.config_name
    cli_overrides = list(hc.overrides.task)
    config_dir = next(
        (s.path for s in hc.runtime.config_sources if s.provider == "main"),
        None,
    )
    if config_dir is None:
        raise RuntimeError("Could not resolve absolute config dir from HydraConfig.")

    overrides = _extract_overrides(cfg)
    print(f"[info] scene  : {overrides['scene_base']}")
    print(f"[info] mjcf   : {overrides['mjcf_paths'] or '(none)'}")
    print(
        f"[info] home   : {len(overrides['ijp'])} joint override(s), "
        f"{len(overrides['initial_pose'])} body pose(s), "
        f"{len(overrides['op_bases'])} operator base(s), "
        f"{len(overrides.get('op_joint_homes', []))} operator joint home(s), "
        f"{len(overrides.get('op_eef_homes', []))} operator EEF home(s), "
        f"{len(overrides.get('camera_initial_pose', {}))} camera pose(s)"
    )

    if _DEBUG:
        print("[debug] preflight build before launching viewer...", flush=True)
        try:
            m, d = _build(overrides)
            print(
                f"[debug] preflight ok: nq={m.nq} nv={m.nv} nu={m.nu} "
                f"nbody={m.nbody} ngeom={m.ngeom}",
                flush=True,
            )
            del m, d
        except Exception:
            _print_debug_exception("preflight build")
            raise

    gs_cfg = _maybe_gs_config(cfg)
    if gs_cfg is not None:
        # GS path: passive viewer whose free-camera state drives a synced GS
        # render in a cv2 window. Passive viewer has no loader hook, so reload
        # is wired through R / viewer reload requests in the render loop below.
        m, d = _build(overrides)
        _print_model_summary(m, overrides)
        _print_gs_summary(gs_cfg)

        def gs_reload_loader() -> tuple[
            DictConfig, dict, mujoco.MjModel, mujoco.MjData
        ]:
            return _load_reloaded_scene(config_dir, config_name, cli_overrides)

        _run_gs_synced_viewer(m, d, gs_cfg, reload_callback=gs_reload_loader)
        return

    def loader() -> tuple[mujoco.MjModel, mujoco.MjData]:
        try:
            # Re-compose the Hydra config from disk so the reload button picks up
            # YAML edits (scene layers, initial_pose, base_pose, joint positions,
            # ...), then rebuild the model through the same generic composer used
            # by the environment.
            _cfg_now, ov, m, d = _load_reloaded_scene(
                config_dir, config_name, cli_overrides
            )
            _print_model_summary(m, ov)
            return m, d
        except Exception:
            if _DEBUG:
                _print_debug_exception("viewer loader")
            raise

    print(
        "[info] launching viewer (close the window to exit; reload button"
        " re-reads YAML+XML and re-applies overrides)..."
    )
    mujoco.viewer.launch(loader=loader)


if __name__ == "__main__":
    _DEBUG = _strip_debug_arg(sys.argv)
    main()
