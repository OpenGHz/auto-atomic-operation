"""Shared MuJoCo model-state initialization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import mujoco
import numpy as np

# Joint qpos widths by mjtJoint enum value: free=7, ball=4, slide=1, hinge=1.
_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}


def _is_joint_position_actuator(model: mujoco.MjModel, actuator_id: int) -> bool:
    """Return whether ``actuator_id`` accepts a scalar joint-position target."""

    if int(model.actuator_dyntype[actuator_id]) != int(mujoco.mjtDyn.mjDYN_NONE):
        return False
    transmission = int(model.actuator_trntype[actuator_id])
    joint_transmissions = {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    }
    if transmission not in joint_transmissions:
        return False
    if int(model.actuator_biastype[actuator_id]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
        return False
    if int(model.actuator_gaintype[actuator_id]) != int(mujoco.mjtGain.mjGAIN_FIXED):
        return False

    gain = float(model.actuator_gainprm[actuator_id, 0])
    bias = model.actuator_biasprm[actuator_id]
    # MuJoCo's <position> shortcut compiles to
    # ``gain * ctrl - gain * length - kv * velocity``.  Requiring that full
    # equilibrium signature avoids treating arbitrary affine <general>
    # actuators as joint-position targets.
    return (
        not np.isclose(gain, 0.0)
        and np.isclose(bias[0], 0.0)
        and np.isclose(bias[1], -gain)
    )


def _hold_position_actuators(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: Iterable[int],
) -> None:
    """Set configured position-actuator targets to their current length."""

    # Actuator controls target transmission length, which equals raw qpos only
    # for the common scalar joint transmission with unit gear.  Forward first
    # so non-unit gears and other supported joint transmissions are respected.
    mujoco.mj_forward(model, data)
    for actuator_id_raw in actuator_ids:
        actuator_id = int(actuator_id_raw)
        if not _is_joint_position_actuator(model, actuator_id):
            continue
        data.ctrl[actuator_id] = data.actuator_length[actuator_id]


def apply_initial_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_joint_positions: Mapping[str, object],
    actuator_ids: Iterable[int] = (),
) -> tuple[str, ...]:
    """Apply configured home joints and settle constrained passive linkages.

    ``actuator_ids`` identifies the operator actuators whose position targets
    must hold the initialized state.  Motor, velocity, tendon, and site
    actuators are deliberately ignored because their controls are not joint
    positions.

    Returns the configured joint names that were not present in ``model``.
    """

    actuator_ids = tuple(dict.fromkeys(int(value) for value in actuator_ids))
    multi_dof: list[tuple[int, np.ndarray]] = []
    pin_addrs: list[int] = []
    missing_joint_names: list[str] = []

    for joint_name, value in initial_joint_positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing_joint_names.append(joint_name)
            continue

        address = int(model.jnt_qposadr[joint_id])
        width = _QPOS_WIDTH[int(model.jnt_type[joint_id])]
        if isinstance(value, (list, tuple)):
            values = np.asarray(value, dtype=float)
            if values.size != width:
                raise ValueError(
                    f"initial_joint_positions['{joint_name}'] has {values.size} "
                    f"values but joint has {width} qpos slots"
                )
            multi_dof.append((address, values))
        else:
            if width != 1:
                raise ValueError(
                    f"initial_joint_positions['{joint_name}'] is scalar but joint "
                    f"has {width} qpos slots; use a list to set all slots"
                )
            data.qpos[address] = float(value)
            pin_addrs.append(address)

    if pin_addrs or multi_dof:
        # Position actuators otherwise retain mj_resetData's zero controls and
        # immediately drive a non-zero home pose back toward zero.
        _hold_position_actuators(model, data, actuator_ids)

    if pin_addrs and model.neq > 0:
        # Equality constraints (parallel-linkage grippers, etc.) are resolved
        # during mj_step.  Pin configured scalar joints and all freejoint
        # bodies while passive joints settle under zero gravity.
        free_addrs: list[int] = []
        for joint_id in range(model.njnt):
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                address = int(model.jnt_qposadr[joint_id])
                free_addrs.extend(range(address, address + 7))

        free_snapshot = data.qpos[free_addrs].copy() if free_addrs else None
        target = data.qpos[pin_addrs].copy()
        saved_gravity = model.opt.gravity.copy()
        model.opt.gravity[:] = 0.0
        try:
            for _ in range(500):
                mujoco.mj_step(model, data)
                data.qpos[pin_addrs] = target
                if free_snapshot is not None:
                    data.qpos[free_addrs] = free_snapshot
        finally:
            model.opt.gravity[:] = saved_gravity
        data.qvel[:] = 0.0

    for address, values in multi_dof:
        data.qpos[address : address + values.size] = values

    if multi_dof:
        _hold_position_actuators(model, data, actuator_ids)
    mujoco.mj_forward(model, data)
    return tuple(missing_joint_names)
