"""Launch the interactive MuJoCo viewer on a composed scene+robot.

Since scene XMLs no longer embed their robot include or keyframe, opening
``demo.xml`` directly in the MuJoCo simulator shows just the empty scene.
This script reads a Hydra task config, composes the scene with the robot(s)
declared under ``env.robot_paths``, applies ``env.initial_joint_positions``
as the home pose, and hands the model to ``mujoco.viewer.launch``.

Usage::

    python examples/view_scene.py --config-name pick_and_place
    python examples/view_scene.py --config-name open_door_airbot_play_back_gs
    python examples/view_scene.py --config-name open_door_p7_ik
"""

from __future__ import annotations

import hydra
import mujoco
import mujoco.viewer
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from auto_atom.runner.common import get_config_dir
from auto_atom.utils.pose import euler_to_quaternion
from auto_atom.utils.scene_loader import load_scene

# Joint qpos widths by mjtJoint enum value: free=7, ball=4, slide=1, hinge=1.
_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}


def _to_container(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _orientation_to_wxyz(orientation: list[float]) -> np.ndarray:
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


def _set_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    position: list[float] | None,
    orientation: list[float] | None,
) -> bool:
    """Apply a position+orientation override to a body. Uses the body's
    freejoint qpos when available, otherwise mutates ``model.body_pos`` /
    ``model.body_quat`` (treating it as world-relative — the parent must be
    world for this to be an exact override). Returns True on success."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        print(f"[warn] body '{body_name}' not found; skipping initial_pose")
        return False
    free_jid = _find_freejoint_for_body(model, body_name)
    if free_jid >= 0:
        addr = int(model.jnt_qposadr[free_jid])
        if position is not None:
            data.qpos[addr : addr + 3] = [float(v) for v in position[:3]]
        if orientation is not None:
            data.qpos[addr + 3 : addr + 7] = _orientation_to_wxyz(orientation)
        return True
    # Static body: write into the model's body_pos / body_quat slot.
    if position is not None:
        model.body_pos[bid] = [float(v) for v in position[:3]]
    if orientation is not None:
        model.body_quat[bid] = _orientation_to_wxyz(orientation)
    return True


def _apply_home_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_joint_positions: dict,
) -> None:
    """Mirror ``MujocoBasis.reset()``: write scalars + multi-DOF entries and
    let parallel-linkage equality constraints settle under zero gravity."""
    multi_dof: list[tuple[int, np.ndarray]] = []
    pin_addrs: list[int] = []
    for joint_name, value in initial_joint_positions.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            print(f"[warn] joint '{joint_name}' not found; skipping")
            continue
        addr = int(model.jnt_qposadr[jid])
        width = _QPOS_WIDTH[int(model.jnt_type[jid])]
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=float)
            if arr.size != width:
                raise ValueError(
                    f"initial_joint_positions['{joint_name}']: got {arr.size} "
                    f"values, joint expects {width}"
                )
            multi_dof.append((addr, arr))
        else:
            if width != 1:
                raise ValueError(
                    f"initial_joint_positions['{joint_name}']: scalar given for "
                    f"a {width}-DOF joint; pass a list"
                )
            data.qpos[addr] = float(value)
            pin_addrs.append(addr)

    if pin_addrs and model.neq > 0:
        # Settle equality-constrained passive joints (e.g. gripper coupler /
        # follower) by stepping in zero gravity while pinning all freejoint
        # bodies and the configured scalar joints.
        free_addrs: list[int] = []
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 0:  # free
                a = int(model.jnt_qposadr[j])
                free_addrs.extend(range(a, a + 7))
        free_snap = data.qpos[free_addrs].copy() if free_addrs else None

        saved_g = model.opt.gravity.copy()
        model.opt.gravity[:] = 0
        target = data.qpos[pin_addrs].copy()
        for _ in range(500):
            mujoco.mj_step(model, data)
            data.qpos[pin_addrs] = target
            if free_snap is not None:
                data.qpos[free_addrs] = free_snap
        data.qvel[:] = 0.0
        model.opt.gravity[:] = saved_g

    for addr, arr in multi_dof:
        data.qpos[addr : addr + arr.size] = arr

    mujoco.mj_forward(model, data)
    # Sync mocap bodies (if any) onto the freejoint they're welded to so the
    # arm doesn't snap back when the viewer takes its first step.
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


def _extract_overrides(cfg: DictConfig) -> dict:
    """Pull the four override surfaces out of a freshly-composed Hydra cfg
    so reload can re-read them without restarting the process."""
    env_cfg = cfg.env
    out: dict = {
        "model_path": str(env_cfg.model_path),
        "robot_paths": [str(p) for p in (env_cfg.get("robot_paths") or [])],
        "ijp": _to_container(env_cfg.get("initial_joint_positions")) or {},
        "initial_pose": _to_container(cfg.get("task", {}).get("initial_pose")) or {},
        "op_bases": [],
    }

    operators_cfg = env_cfg.get("operators") or {}
    task_operators_cfg = cfg.get("task_operators") or {}
    items = task_operators_cfg.items() if task_operators_cfg else []
    for op_name, op_node in items:
        bp_cfg = _to_container((op_node.get("initial_state") or {}).get("base_pose"))
        if not bp_cfg:
            continue
        root_body = (operators_cfg.get(op_name) or {}).get("root_body")
        if not root_body:
            print(
                f"[warn] task_operators.{op_name}.initial_state.base_pose set "
                f"but env.operators.{op_name}.root_body is empty; skipping"
            )
            continue
        out["op_bases"].append((root_body, bp_cfg))
    return out


def _build(overrides: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = load_scene(overrides["model_path"], overrides["robot_paths"])
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # 1) task_operators.*.initial_state.base_pose — relocates the arm root
    #    body so the arm sits at the right world pose.
    for root_body, bp in overrides["op_bases"]:
        _set_body_pose(m, d, root_body, bp.get("position"), bp.get("orientation"))

    # 2) task.initial_pose — per-object overrides (freejoint qpos or static body_pos).
    for body_name, override in overrides["initial_pose"].items():
        override = override or {}
        _set_body_pose(
            m,
            d,
            body_name,
            override.get("position"),
            override.get("orientation"),
        )

    # 3) env.initial_joint_positions — joint-level home pose.
    _apply_home_pose(m, d, overrides["ijp"])
    return m, d


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
    config_dir = next(
        (s.path for s in hc.runtime.config_sources if s.provider == "main"),
        None,
    )
    if config_dir is None:
        raise RuntimeError("Could not resolve absolute config dir from HydraConfig.")

    overrides = _extract_overrides(cfg)
    print(f"[info] scene  : {overrides['model_path']}")
    print(f"[info] robots : {overrides['robot_paths'] or '(none)'}")
    print(
        f"[info] home   : {len(overrides['ijp'])} joint override(s), "
        f"{len(overrides['initial_pose'])} body pose(s), "
        f"{len(overrides['op_bases'])} operator base(s)"
    )

    def loader() -> tuple[mujoco.MjModel, mujoco.MjData]:
        # Re-compose the Hydra config from disk so the reload button picks up
        # YAML edits (initial_pose, base_pose, robot_paths, joint_positions, ...),
        # then rebuild the model. ``load_scene`` already re-reads the scene/robot
        # XML files from disk so XML edits also flow through.
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg_now = compose(config_name=config_name)
        ov = _extract_overrides(cfg_now)
        m, d = _build(ov)
        print(
            f"[info] model  : nq={m.nq} nv={m.nv} nu={m.nu} "
            f"nbody={m.nbody} ngeom={m.ngeom}  "
            f"(robots={ov['robot_paths']}, ijp={len(ov['ijp'])}, "
            f"body_pose={len(ov['initial_pose'])}, op_base={len(ov['op_bases'])})"
        )
        return m, d

    print(
        "[info] launching viewer (close the window to exit; reload button"
        " re-reads YAML+XML and re-applies overrides)..."
    )
    mujoco.viewer.launch(loader=loader)


if __name__ == "__main__":
    main()
