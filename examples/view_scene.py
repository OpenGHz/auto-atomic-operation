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
from omegaconf import DictConfig, ListConfig, OmegaConf

from auto_atom.runner.common import get_config_dir
from auto_atom.utils.scene_loader import load_scene

# Joint qpos widths by mjtJoint enum value: free=7, ball=4, slide=1, hinge=1.
_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}


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


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    env_cfg = cfg.env
    model_path = str(env_cfg.model_path)
    robot_paths = [str(p) for p in (env_cfg.get("robot_paths") or [])]
    ijp_raw = env_cfg.get("initial_joint_positions")
    if ijp_raw is None:
        ijp = {}
    elif isinstance(ijp_raw, (DictConfig, ListConfig)):
        ijp = OmegaConf.to_container(ijp_raw, resolve=True) or {}
    else:
        ijp = dict(ijp_raw)

    print(f"[info] scene  : {model_path}")
    print(f"[info] robots : {robot_paths or '(none)'}")
    print(f"[info] home   : {len(ijp)} joint override(s)")

    def loader() -> tuple[mujoco.MjModel, mujoco.MjData]:
        # Called once at startup and again every time the user clicks the
        # viewer's reload button. Re-runs the full compose + home-pose
        # pipeline so reload picks up edits to the scene/robot XML AND
        # restores the configured home pose (instead of snapping to qpos0).
        m = load_scene(model_path, robot_paths)
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        _apply_home_pose(m, d, ijp)
        print(
            f"[info] model  : nq={m.nq} nv={m.nv} nu={m.nu} "
            f"nbody={m.nbody} ngeom={m.ngeom}"
        )
        return m, d

    print(
        "[info] launching viewer (close the window to exit; reload button"
        " re-runs compose+home)..."
    )
    mujoco.viewer.launch(loader=loader)


if __name__ == "__main__":
    main()
