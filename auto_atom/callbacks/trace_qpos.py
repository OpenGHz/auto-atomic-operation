"""Debug pre-step callback: prints every N steps handle/door qpos + EEF pose.

Configure in YAML::

    env:
      pre_step_callbacks:
        - _target_: auto_atom.callbacks.trace_qpos.TraceQposCallback
          joints: [handle_hinge, door_hinge]
          eef_site: eef_pose
          every: 10
"""

from __future__ import annotations

import mujoco
import numpy as np


class TraceQposCallback:
    def __init__(
        self,
        joints: list[str],
        eef_site: str = "",
        every: int = 10,
    ) -> None:
        self._joint_names = list(joints)
        self._eef_site = eef_site
        self._every = int(every)
        self._joint_qidx: list[int] = []
        self._eef_site_id: int = -1
        self._step: int = 0

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._joint_qidx = []
        for name in self._joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qidx.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)
        self._eef_site_id = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._eef_site)
            if self._eef_site
            else -1
        )
        self._step = 0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._step += 1
        if self._step % self._every != 0:
            return
        parts = [f"step={self._step:4d}"]
        for name, idx in zip(self._joint_names, self._joint_qidx):
            if idx >= 0:
                parts.append(f"{name}={float(data.qpos[idx]):+.4f}")
        if self._eef_site_id >= 0:
            pos = np.asarray(data.site_xpos[self._eef_site_id]).round(3).tolist()
            parts.append(f"eef={pos}")
        print("[trace]", "  ".join(parts), flush=True)
