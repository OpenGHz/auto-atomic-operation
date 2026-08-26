"""Door latch callback backed by a switchable MuJoCo joint constraint."""

from __future__ import annotations

from typing import Literal

import mujoco
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
)


class DoorLatchConfig(BaseModel, frozen=True):
    """Configuration for a handle-released door latch."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    lock_constraint: str
    """Name of an active-by-default MuJoCo joint equality that holds the door closed."""
    door_joint: str
    """Door hinge constrained by ``lock_constraint``."""
    handle_joint: str
    """Handle hinge whose travel releases the latch."""
    handle_direction: Literal[-1, 1] = 1
    """Sign that maps handle joint displacement onto positive unlatching travel."""
    unlock_travel: PositiveFloat = 0.20
    """Handle travel in radians required to release the latch."""
    relock_travel: NonNegativeFloat = 0.15
    """Handle travel below which an open latch may re-engage."""
    relock_zone: PositiveFloat = 0.02
    """Maximum door-angle error in radians at which the latch may re-engage."""

    @model_validator(mode="after")
    def validate_hysteresis(self) -> DoorLatchConfig:
        """Require a non-empty hysteresis band between relock and unlock."""
        if self.relock_travel >= self.unlock_travel:
            raise ValueError("relock_travel must be smaller than unlock_travel")
        return self


class DoorLatchCallback:
    """Toggle a door-hinge equality from measured handle travel.

    The equality constraint is the physical lock. It starts active, becomes
    inactive after the handle reaches ``unlock_travel``, and can re-engage only
    after the handle returns below ``relock_travel`` while the door remains in
    the configured closed-door capture zone.

    Configure in Hydra YAML with a validated nested config::

        env:
          pre_step_callbacks:
            - _target_: auto_atom.callbacks.door_latch.DoorLatchCallback
              config:
                _target_: auto_atom.callbacks.door_latch.DoorLatchConfig
                lock_constraint: door_latch_lock
                door_joint: door_hinge
                handle_joint: handle_hinge
                handle_direction: 1
                unlock_travel: 0.12
                relock_travel: 0.08
                relock_zone: 0.02
    """

    def __init__(self, config: DoorLatchConfig) -> None:
        self.config = config
        self._constraint_id = -1
        self._door_qpos_idx = -1
        self._handle_qpos_idx = -1
        self._handle_rest_angle = 0.0
        self._lock_angle = 0.0

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Resolve and validate the latch's joints and equality constraint."""
        del data
        door_id = self._joint_id(model, self.config.door_joint)
        handle_id = self._joint_id(model, self.config.handle_joint)
        for joint_id, name in (
            (door_id, self.config.door_joint),
            (handle_id, self.config.handle_joint),
        ):
            joint_type = model.jnt_type[joint_id]
            if joint_type not in {
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            }:
                raise ValueError(
                    f"DoorLatchCallback: joint '{name}' must be hinge or slide."
                )

        constraint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            self.config.lock_constraint,
        )
        if constraint_id < 0:
            raise ValueError(
                "DoorLatchCallback: equality constraint "
                f"'{self.config.lock_constraint}' not found."
            )
        if model.eq_type[constraint_id] != mujoco.mjtEq.mjEQ_JOINT:
            raise ValueError(
                "DoorLatchCallback: lock_constraint must be a joint equality."
            )
        if (
            int(model.eq_obj1id[constraint_id]) != door_id
            or int(model.eq_obj2id[constraint_id]) != -1
        ):
            raise ValueError(
                "DoorLatchCallback: lock_constraint must constrain door_joint "
                "to a constant angle."
            )
        if any(
            abs(float(value)) > 1e-12 for value in model.eq_data[constraint_id, 1:5]
        ):
            raise ValueError(
                "DoorLatchCallback: lock_constraint must use a constant polycoef."
            )
        if not bool(model.eq_active0[constraint_id]):
            raise ValueError(
                "DoorLatchCallback: lock_constraint must be active by default."
            )

        self._constraint_id = int(constraint_id)
        self._door_qpos_idx = int(model.jnt_qposadr[door_id])
        self._handle_qpos_idx = int(model.jnt_qposadr[handle_id])
        self._handle_rest_angle = float(model.qpos_spring[self._handle_qpos_idx])
        self._lock_angle = float(model.eq_data[constraint_id, 0])

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Release or re-engage the equality before one physics substep."""
        del model
        handle_angle = float(data.qpos[self._handle_qpos_idx])
        handle_travel = self.config.handle_direction * (
            handle_angle - self._handle_rest_angle
        )
        locked = bool(data.eq_active[self._constraint_id])
        if locked and handle_travel >= self.config.unlock_travel:
            data.eq_active[self._constraint_id] = 0
            return
        door_error = float(data.qpos[self._door_qpos_idx]) - self._lock_angle
        if (
            not locked
            and handle_travel <= self.config.relock_travel
            and abs(door_error) <= self.config.relock_zone
        ):
            data.eq_active[self._constraint_id] = 1

    @staticmethod
    def _joint_id(model: mujoco.MjModel, name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"DoorLatchCallback: joint '{name}' not found in model.")
        return int(joint_id)
