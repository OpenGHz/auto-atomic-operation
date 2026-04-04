"""Door latch callback: conditional spring-damper that locks the door hinge."""

from __future__ import annotations

import mujoco


class DoorLatchCallback:
    """Per-step callback that simulates a door latch mechanism.

    Applies a spring-damper restoring force to ``door_joint`` only when
    ``handle_joint`` is below ``unlock_threshold`` **and** the door angle
    is within ``lock_zone`` of zero.  Once the handle is pressed past the
    threshold, the latch releases and the door swings freely.

    Configure in YAML via Hydra ``_target_``::

        env:
          pre_step_callbacks:
            - _target_: auto_atom.callbacks.door_latch.DoorLatchCallback
              door_joint: door_hinge
              handle_joint: handle_hinge
              kp: 80.0
              kd: 8.0
    """

    def __init__(
        self,
        door_joint: str,
        handle_joint: str,
        kp: float = 80.0,
        kd: float = 8.0,
        unlock_threshold: float = 0.20,
        lock_zone: float = 0.05,
    ) -> None:
        self._door_joint = door_joint
        self._handle_joint = handle_joint
        self._kp = kp
        self._kd = kd
        self._unlock_threshold = unlock_threshold
        self._lock_zone = lock_zone
        self._door_qpos_idx: int = -1
        self._door_dof_idx: int = -1
        self._handle_qpos_idx: int = -1

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Resolve joint indices from the loaded MuJoCo model."""
        jid_door = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._door_joint)
        if jid_door < 0:
            raise ValueError(
                f"DoorLatchCallback: joint '{self._door_joint}' not found in model."
            )
        self._door_qpos_idx = int(model.jnt_qposadr[jid_door])
        self._door_dof_idx = int(model.jnt_dofadr[jid_door])

        jid_handle = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, self._handle_joint
        )
        if jid_handle < 0:
            raise ValueError(
                f"DoorLatchCallback: joint '{self._handle_joint}' not found in model."
            )
        self._handle_qpos_idx = int(model.jnt_qposadr[jid_handle])

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        handle_angle = float(data.qpos[self._handle_qpos_idx])
        door_angle = float(data.qpos[self._door_qpos_idx])
        door_vel = float(data.qvel[self._door_dof_idx])
        if handle_angle < self._unlock_threshold and abs(door_angle) < self._lock_zone:
            data.qfrc_applied[self._door_dof_idx] += (
                -self._kp * door_angle - self._kd * door_vel
            )
