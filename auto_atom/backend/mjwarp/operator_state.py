"""Per-operator control state for the MJWarp backend.

Registration snapshots everything the control path needs to address an operator
without re-deriving it every tick: the base frame each world's targets are
expressed in, the fixed tool offset from that base to the end-effector, and the
home joint/actuator state a reset restores.

Two differences from the native ``_OperatorState`` are structural rather than
incidental:

* **Base pose is per world.** Native stores one ``base_pos``/``base_quat`` per
  replica because it owns one ``MjModel`` per replica. MJWarp has a single
  device model with batched ``body_pos``/``body_quat``, so the base pose is
  ``(nworld, 3)`` / ``(nworld, 4)`` and each world converts against its own --
  which is what lets randomization place the base per environment.
* **Mocap uses a virtual base.** Its world-origin base is independent of the
  moving physical body. Tool offsets include the authored weld transform, so
  the controller can command an EEF pose through the configured mocap target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from auto_atom.backend.mjwarp.frames import world_to_base_batch
from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.utils.transformations import (
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)


@dataclass
class MjWarpOperatorState:
    """What the control path needs to drive one operator.

    All pose arrays carry a leading world axis, so a caller never has to ask
    whether a value is shared or per-world: it is always per-world.
    """

    name: str
    root_body_name: str
    eef_site_name: str

    # Actuator and joint addressing, resolved once at registration.
    arm_actuator_ids: np.ndarray
    eef_actuator_ids: np.ndarray
    arm_qpos_indices: np.ndarray
    arm_dof_indices: np.ndarray
    eef_qpos_indices: np.ndarray
    eef_dof_indices: np.ndarray

    # Base frame in world, per world.
    base_position: np.ndarray  # (nworld, 3)
    base_orientation: np.ndarray  # (nworld, 4) xyzw

    # Fixed base -> eef offset, per world.
    tool_offset_position: np.ndarray  # (nworld, 3)
    tool_offset_orientation: np.ndarray  # (nworld, 4) xyzw

    # Home state a reset restores.
    home_arm_qpos: np.ndarray  # (nworld, n_arm)
    home_ctrl: np.ndarray  # (nworld, nu)

    mocap_body_name: str = ""
    freejoint_name: str = ""
    home_mocap_position: Optional[np.ndarray] = None
    home_mocap_orientation: Optional[np.ndarray] = None
    mocap_to_root_position: Optional[np.ndarray] = None
    mocap_to_root_orientation: Optional[np.ndarray] = None

    joint_control_mode: str = "per_step_ik"
    max_joint_delta: float = 0.35
    joint_interp_speed: float = 0.05
    ik_solver: Optional[object] = field(default=None, repr=False)
    _baseline: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def restore_baseline(self, world_mask: np.ndarray) -> None:
        """Restore frame and home snapshots before applying a fresh reset."""
        for name, baseline in self._baseline.items():
            getattr(self, name)[world_mask] = baseline[world_mask]

    @property
    def joint_mode(self) -> bool:
        """True for an actuator-driven arm; false for a mocap-driven body."""
        return self.arm_actuator_ids.size > 0


_SUPPORTED_CONTROL_MODES = frozenset({"per_step_ik", "solve_once_interpolate"})


def register_operator(
    state: MjWarpSceneState,
    *,
    name: str,
    root_body: str,
    eef_site: str,
    arm_actuators: Tuple[str, ...],
    eef_actuators: Tuple[str, ...] = (),
    ik_solver: Optional[object] = None,
    joint_control_mode: str = "per_step_ik",
    joint_interp_speed: float = 0.05,
    max_joint_delta: float = 0.35,
    mocap_body: str = "",
    freejoint: str = "",
) -> MjWarpOperatorState:
    """Resolve an operator's addressing and snapshot its home state.

    The base pose is read from the root body's *current* world pose, per world,
    and the tool offset from the eef site's current pose expressed in that base.
    Both are therefore taken against whatever the scene looks like at
    registration time, matching the native path's registration-time snapshot.

    Rejects the cases that cannot be driven rather than producing a state that
    fails later at the first control tick: an arm with no IK solver, an
    unsupported control mode, or a non-positive interpolation speed -- the same
    three checks native makes, with the same reasons.
    """
    if not arm_actuators and (not mocap_body or not freejoint):
        raise ValueError(
            f"Operator '{name}' has no arm_actuators; mocap control requires "
            "both mocap_body and freejoint."
        )
    if arm_actuators and ik_solver is None:
        raise ValueError(
            f"Operator '{name}' has arm_actuators but no ik_solver was provided."
        )
    if joint_control_mode not in _SUPPORTED_CONTROL_MODES:
        raise ValueError(
            f"Unsupported joint_control_mode '{joint_control_mode}' for "
            f"operator '{name}'. Supported: {sorted(_SUPPORTED_CONTROL_MODES)}."
        )
    if joint_interp_speed <= 0.0:
        raise ValueError(
            f"joint_interp_speed must be > 0 for operator '{name}', "
            f"got {joint_interp_speed}."
        )

    arm_ids = state.actuator_ids(arm_actuators)
    eef_ids = (
        state.actuator_ids(eef_actuators)
        if eef_actuators
        else np.empty(0, dtype=np.int32)
    )
    arm_qidx, arm_vidx = state.actuator_joint_indices(arm_ids)
    eef_qidx, eef_vidx = state.actuator_joint_indices(eef_ids)

    base_position, base_orientation = state.get_body_pose_batch(root_body)
    base_position = np.asarray(base_position, dtype=np.float64)
    base_orientation = np.asarray(base_orientation, dtype=np.float64)

    eef_position, eef_orientation = state.get_site_pose_batch(eef_site)
    tool_position, tool_orientation = world_to_base_batch(
        eef_position, eef_orientation, base_position, base_orientation
    )

    home_mocap_position = home_mocap_orientation = None
    mocap_to_root_position = mocap_to_root_orientation = None
    if not arm_actuators:
        home_mocap_position, home_mocap_orientation = state.get_mocap_pose(mocap_body)
        # Validate the free-joint binding at registration, before any control write.
        import mujoco

        joint = state._id(mujoco.mjtObj.mjOBJ_JOINT, freejoint, "Joint")
        if int(state.host_model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"Joint '{freejoint}' is not a free joint.")
        from auto_atom.basis.mjc.model_initialization import mocap_weld_transform

        mocap_to_root_position, mocap_to_root_orientation = mocap_weld_transform(
            state.host_model,
            state.body_id(mocap_body),
            int(state.host_model.jnt_bodyid[joint]),
        )
        tool_position, tool_orientation = world_to_base_batch(
            eef_position, eef_orientation, home_mocap_position, home_mocap_orientation
        )
        base_position = np.zeros((state.nworld, 3))
        base_orientation = np.tile([0.0, 0.0, 0.0, 1.0], (state.nworld, 1))

    operator = MjWarpOperatorState(
        name=name,
        root_body_name=root_body,
        eef_site_name=eef_site,
        arm_actuator_ids=arm_ids,
        eef_actuator_ids=eef_ids,
        arm_qpos_indices=arm_qidx,
        arm_dof_indices=arm_vidx,
        eef_qpos_indices=eef_qidx,
        eef_dof_indices=eef_vidx,
        base_position=base_position,
        base_orientation=base_orientation,
        tool_offset_position=tool_position,
        tool_offset_orientation=tool_orientation,
        home_arm_qpos=state.get_joint_positions(arm_qidx),
        home_ctrl=state.get_ctrl(),
        joint_control_mode=joint_control_mode,
        max_joint_delta=float(max_joint_delta),
        joint_interp_speed=float(joint_interp_speed),
        ik_solver=ik_solver,
        mocap_body_name=mocap_body,
        freejoint_name=freejoint,
        home_mocap_position=home_mocap_position,
        home_mocap_orientation=home_mocap_orientation,
        mocap_to_root_position=mocap_to_root_position,
        mocap_to_root_orientation=mocap_to_root_orientation,
    )
    for field_name in (
        "base_position",
        "base_orientation",
        "tool_offset_position",
        "tool_offset_orientation",
        "home_arm_qpos",
        "home_ctrl",
        "home_mocap_position",
        "home_mocap_orientation",
    ):
        value = getattr(operator, field_name)
        if value is not None:
            operator._baseline[field_name] = value.copy()
    return operator


def eef_to_root_pose(
    operator: MjWarpOperatorState,
    world: int,
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a world EEF goal to the mocap control frame, including its weld."""
    root_quat = quaternion_multiply(
        orientation_xyzw, quaternion_inverse(operator.tool_offset_orientation[world])
    )
    root_pos = np.asarray(position) - (
        quaternion_matrix(root_quat)[:3, :3] @ operator.tool_offset_position[world]
    )
    return root_pos, root_quat


def override_base_pose(
    state: MjWarpSceneState,
    operator: MjWarpOperatorState,
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
) -> None:
    """Replace the stored base frame and refresh the tool offset against it.

    ``initial_state.base_pose`` in a task config is applied through here. The
    tool offset has to be recomputed rather than kept: it is defined relative to
    the base, so moving the base without refreshing it would leave the control
    path converting targets against one frame and measuring the tool in another.

    The physical root body is **not** relocated here. Native does relocate it
    for joint-mode operators so the IK solver and the virtual base agree; doing
    that needs the arm-placement write that lands with the physical env, so this
    function is the frame half only and the caller places the body.
    """
    nworld = state.nworld
    position = np.atleast_2d(np.asarray(position, dtype=np.float64))
    orientation_xyzw = np.atleast_2d(np.asarray(orientation_xyzw, dtype=np.float64))
    if position.shape[0] == 1:
        position = np.repeat(position, nworld, axis=0)
    if orientation_xyzw.shape[0] == 1:
        orientation_xyzw = np.repeat(orientation_xyzw, nworld, axis=0)
    if position.shape != (nworld, 3) or orientation_xyzw.shape != (nworld, 4):
        raise ValueError(
            f"Base pose must be broadcastable to ({nworld}, 3) / ({nworld}, 4); "
            f"got {position.shape} and {orientation_xyzw.shape}."
        )

    operator.base_position = position
    operator.base_orientation = orientation_xyzw

    eef_position, eef_orientation = state.get_site_pose_batch(operator.eef_site_name)
    tool_position, tool_orientation = world_to_base_batch(
        eef_position, eef_orientation, position, orientation_xyzw
    )
    operator.tool_offset_position = tool_position
    operator.tool_offset_orientation = tool_orientation


def get_eef_pose_in_world(
    state: MjWarpSceneState,
    operator: MjWarpOperatorState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Current end-effector pose per world, read from the eef site."""
    return state.get_site_pose_batch(operator.eef_site_name)


def get_eef_pose_in_base(
    state: MjWarpSceneState,
    operator: MjWarpOperatorState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Current end-effector pose per world, in that world's base frame."""
    position, orientation = state.get_site_pose_batch(operator.eef_site_name)
    return world_to_base_batch(
        position, orientation, operator.base_position, operator.base_orientation
    )


def get_base_pose(operator: MjWarpOperatorState) -> Tuple[np.ndarray, np.ndarray]:
    """Stored base pose per world.

    Read from the snapshot rather than from the body every call, matching
    native: the base is fixed for the duration of a reset, and re-reading it
    would make a mid-episode body write silently redefine every target frame.
    """
    return operator.base_position, operator.base_orientation
