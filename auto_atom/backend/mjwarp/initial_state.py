"""Applying an operator's ``initial_state`` on the MJWarp backend.

The end-to-end physical run (design doc 5.2) found that nothing applied
``task_operators.<name>.initial_state``: the arm sat at its raw MJCF pose, so
targets were converted against the wrong base frame and the arm drove away from
every goal while IK reported success. This is the fix.

Scope is deliberately the forms the target config uses -- a full ``base_pose``
(world frame), a full ``eef_pose`` (base frame), and an ``eef`` gripper value.
The partial/structured/Euler ``PoseOverrideConfig`` forms and per-joint
``joint_positions`` are refused with a clear message rather than silently
half-applied: a mis-resolved home pose is exactly the failure this round exists
to remove, so guessing is worse than stopping.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from auto_atom.backend.mjwarp.operator_state import override_base_pose


def _full_pose(
    override: Any, *, field: str, operator: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``(position_xyz, orientation_xyzw)`` from a pose override.

    Only the full-quaternion form is accepted: a length-3 position and a
    length-4 xyzw orientation, both present. Anything else -- an omitted
    component, a structured x/y/z mapping, or Euler angles -- is refused, because
    resolving those correctly means reproducing the native pose resolver, and a
    silently wrong home pose is the very bug being fixed.
    """
    position = getattr(override, "position", None)
    orientation = getattr(override, "orientation", None)

    def _reject(reason: str) -> None:
        raise NotImplementedError(
            f"Operator '{operator}' {field}: the MJWarp backend applies only the "
            f"full-pose form (3-value position + 4-value xyzw orientation). {reason} "
            "Structured/partial/Euler overrides and named scene references are not "
            "implemented; add them deliberately rather than resolving them wrong."
        )

    if position is None or orientation is None:
        _reject("A component was omitted.")
    try:
        position_array = np.asarray(position, dtype=np.float64).reshape(-1)
        orientation_array = np.asarray(orientation, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        _reject("A component was not a plain numeric sequence.")
    if position_array.shape != (3,):
        _reject(f"Position had shape {position_array.shape}, not (3,).")
    if orientation_array.shape != (4,):
        _reject(
            f"Orientation had shape {orientation_array.shape}, not (4,) -- Euler "
            "angles (3 values) are not accepted here."
        )
    return position_array, orientation_array


def _reference(override: Any) -> str:
    """The override's reference frame as a plain string (default ``world``)."""
    reference = getattr(override, "reference", "world")
    return getattr(reference, "value", reference)


def apply_initial_state(
    handler: Any,
    initial_state: Any,
    env_mask: Optional[np.ndarray] = None,
) -> None:
    """Apply one operator's configured initial state, then home the arm.

    Order matters and follows native: the base frame is resolved first, because
    an ``eef_pose`` given in ``base`` reference must see the new base; then the
    home arm pose is solved from the eef target; then the gripper; then the arm
    is physically homed to the solved joints.
    """
    if initial_state is None:
        return

    state = handler.state
    operator = handler.operator

    if getattr(initial_state, "joint_positions", None):
        raise NotImplementedError(
            f"Operator '{operator.name}': initial_state.joint_positions is not "
            "implemented on the MJWarp backend. The target config uses eef_pose "
            "instead; add joint homing deliberately if a config needs it."
        )

    if getattr(initial_state, "base_pose", None) is not None:
        _apply_base_pose(handler, initial_state.base_pose, env_mask)

    if getattr(initial_state, "eef_pose", None) is not None:
        _apply_eef_home(handler, initial_state.eef_pose, env_mask)

    # Physically realise the home arm pose for the selected worlds. home()
    # writes qpos, zeroes velocity, mirrors ctrl and clears controller progress.
    # The gripper is homed separately below by writing its ctrl directly, since
    # it is actuator-commanded rather than kinematically pinned.
    handler.home(env_mask=env_mask)
    _home_gripper(handler, initial_state, env_mask)


def _apply_base_pose(
    handler: Any, base_pose: Any, env_mask: Optional[np.ndarray]
) -> None:
    """Relocate the operator's base frame, physically and virtually.

    The root body is relocated in the scene *and* the stored base frame is
    updated, because both must agree: world<->base conversion reads the stored
    frame, while IK and collisions read the physical body. Native does the same
    for a joint-mode operator (``override_operator_base_pose``); doing only one
    is what left the arm converting against a frame the body was not at.
    """
    operator = handler.operator
    reference = _reference(base_pose)
    if reference != "world":
        raise NotImplementedError(
            f"Operator '{operator.name}' base_pose reference '{reference}' is not "
            "implemented; only world-frame base poses are applied."
        )
    position, orientation = _full_pose(
        base_pose, field="base_pose", operator=operator.name
    )

    # The root body carries no joint (a static mount), so it is placed by the
    # static-body write -- which also refreshes its geom frames (design doc 3.9).
    handler.state.set_object_pose(
        operator.root_body_name,
        position,
        orientation,
        world_mask=None if env_mask is None else np.asarray(env_mask, dtype=bool),
    )
    override_base_pose(handler.state, operator, position, orientation)


def _apply_eef_home(
    handler: Any, eef_pose: Any, env_mask: Optional[np.ndarray]
) -> None:
    """Solve IK for the configured home eef pose and record it as home qpos.

    The eef target is resolved to world (a ``base`` reference is composed through
    the operator's freshly-set base), then IK is solved per world from the
    current arm qpos as seed. A world whose IK fails keeps its previous home
    rather than adopting a bad solution -- the failure is already logged by the
    IK caller.
    """
    operator = handler.operator
    reference = _reference(eef_pose)
    position, orientation = _full_pose(
        eef_pose, field="eef_pose", operator=operator.name
    )

    home = operator.home_arm_qpos.copy()
    seeds = handler.state.get_joint_positions(operator.arm_qpos_indices)
    mask = (
        np.ones(handler.state.nworld, dtype=bool)
        if env_mask is None
        else np.asarray(env_mask, dtype=bool)
    )
    for world in np.flatnonzero(mask):
        if reference == "base":
            pos_b, quat_b = position, orientation
        elif reference == "world":
            # Express the world target in this world's base frame for IK.
            from auto_atom.backend.mjwarp.frames import world_to_base

            pos_b, quat_b = world_to_base(
                position,
                orientation,
                operator.base_position[world],
                operator.base_orientation[world],
            )
        else:
            raise NotImplementedError(
                f"Operator '{operator.name}' eef_pose reference '{reference}' is "
                "not implemented; use 'base' or 'world'."
            )
        solution = handler.arm.ik.solve(
            world, pos_b, quat_b, seeds[world], context="initial_state"
        )
        if solution is not None:
            home[world, : len(solution)] = np.asarray(solution, dtype=np.float64)
    operator.home_arm_qpos = home


def _home_gripper(
    handler: Any, initial_state: Any, env_mask: Optional[np.ndarray]
) -> None:
    """Write the gripper's home command for the selected worlds, if configured."""
    if getattr(initial_state, "eef", None) is None or handler.eef is None:
        return
    eef_ids = handler.operator.eef_actuator_ids
    if eef_ids.size == 0:
        return
    value = float(initial_state.eef)
    handler.state.set_ctrl(
        eef_ids,
        np.full(eef_ids.size, value, dtype=np.float64),
        world_mask=None if env_mask is None else np.asarray(env_mask, dtype=bool),
    )
