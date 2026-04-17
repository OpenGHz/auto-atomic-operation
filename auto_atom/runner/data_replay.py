"""DataReplayRunner – replays recorded demonstration data through PolicyEvaluator.

This module extracts the core replay logic from ``examples/replay_demo.py`` into
a runner class that conforms to the :class:`RunnerBase` interface, so it can be
driven by an external manager via the standard ``reset`` / ``update`` / ``close``
lifecycle.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_atom.framework import TaskFileConfig
from auto_atom.policy_eval import PolicyEvaluator
from auto_atom.runtime import ExecutionContext, TaskUpdate

from .base import RunnerBase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SimEntityRef(BaseModel):
    """Reference to a simulation entity whose world pose can be queried
    (and, for kinds that support it, updated).

    Supported kinds:
      - ``site``: MuJoCo site, queried via ``env.get_site_pose``.
      - ``body``: MuJoCo body, queried via ``env.get_body_pose``.
      - ``operator_base``: operator virtual base frame, queried via
        ``env.get_operator_base_pose`` and updated via
        ``env.override_operator_base_pose``.
    """

    model_config = ConfigDict(validate_assignment=True)

    kind: Literal["site", "body", "operator_base"]
    name: str
    """site/body name in the MuJoCo model, or operator name when
    ``kind == 'operator_base'``."""


class PoseOffset(BaseModel):
    """Small calibration tweak added to a computed pose.

    ``position`` is an additive offset in the world frame. ``orientation``
    is right-multiplied onto the computed rotation (i.e. applied in the
    movable entity's local frame). Defaults are the identity transform,
    so an unset offset is a no-op.
    """

    model_config = ConfigDict(validate_assignment=True)

    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    """World-frame translation added to the computed position (x, y, z)."""

    orientation: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    """Rotation offset right-multiplied onto the computed quaternion.
    Accepts either 4 floats (xyzw quaternion) or 3 floats (intrinsic XYZ
    euler, radians).  Default is identity."""


class TransformResetConfig(BaseModel):
    """Generic scene-reset rule driven by a recorded ``TransformStamped``.

    Reads the ``message_index``-th ``geometry_msgs/TransformStamped`` on
    ``topic`` from the MCAP. The message is interpreted as
    ``T_parent->child`` (translation of the child origin expressed in the
    parent frame, plus child-in-parent rotation). At reset time the
    runner queries the ``move``-side entity's current pose together with
    the *other* (fixed) side's world pose, then repositions the
    ``move``-side entity so that the simulated relative pose matches the
    recording.  The optional ``offset`` is applied afterwards for fine
    calibration.
    """

    model_config = ConfigDict(validate_assignment=True)

    topic: str
    parent: SimEntityRef
    child: SimEntityRef
    move: Literal["parent", "child"] = "parent"
    """Which side to reposition. The other side is treated as fixed."""
    message_index: int = 0
    use_orientation: bool = False
    """If False (default) keep the movable entity's current world
    orientation and only adjust translation. If True, compose the full
    transform (may rotate the movable entity)."""
    offset: PoseOffset = Field(default_factory=PoseOffset)
    """Post-hoc calibration offset applied to the computed movable-side
    pose.  Default is identity (no adjustment)."""


class JointClipBounds(BaseModel):
    """Optional ``[min, max]`` clamp applied to one actuator's recorded
    trajectory.  Either bound may be omitted to leave that side
    unclipped.  Values are in the actuator's user-facing units (e.g.
    finger distance in metres for an EEF behind a
    :class:`FingerDistanceMapper`)."""

    model_config = ConfigDict(validate_assignment=True)

    min: Optional[float] = None
    max: Optional[float] = None


class DataReplayConfig(BaseModel):
    """Replay-specific settings (subset of the original ``ReplayConfig``)."""

    model_config = ConfigDict(validate_assignment=True)

    demo_name: str | None = Field(default=None)
    mode: str = Field(default="pose")
    reset_from_first_frame: bool = Field(default=True)
    steps_per_action: int = Field(default=1)
    """Number of physics steps to repeat each action before advancing to the
    next one.  Set to >1 when the simulation timestep is smaller than the
    demo recording interval (e.g. 10 for MuJoCo sub-stepping)."""
    mcap_path: str | None = Field(default=None)
    arm_topic: str = Field(default="/robot/right_arm/joint_state")
    gripper_topic: str = Field(default="/robot/right_gripper/joint_state")
    joint_name_mapping: Dict[str, str] = {"gripper": "xfg_claw_joint"}
    joint_axis_scale: List[float] = Field(default_factory=list)
    """Per-joint multipliers applied to recorded joint/ctrl data after
    column reordering. Only the first ``N`` columns are scaled, where
    ``N = len(joint_axis_scale)``; trailing columns keep a factor of
    ``1.0``. Useful for mirroring a replay by negating selected revolute
    axes, e.g. ``[1, 1, -1, 1, 1, 1]``."""
    kinematic: bool = Field(default=False)
    """If ``True`` the replay sets joint positions directly (no physics);
    if ``False`` the replay drives actuators through the physics engine."""
    demo_dir: str | None = Field(default=None)
    """Directory containing npz demo files. Defaults to ``outputs/records/demos``."""
    done_on_success: bool = Field(default=False)
    """If ``True``, report ``done=True`` as soon as all stages succeed.
    If ``False`` (default), defer ``done=True`` until all replay data has
    been played back, even if stages already succeeded."""
    transform_resets: List[TransformResetConfig] = Field(default_factory=list)
    """Generic scene-reset rules driven by MCAP ``TransformStamped`` topics.
    Applied after ``evaluator.reset()`` and before the first-frame action
    reset.  Empty by default (no-op)."""
    joint_clip: Dict[str, JointClipBounds] = Field(default_factory=dict)
    """Per-actuator value clamping applied to the recorded joint trajectory
    once at demo-load time, after column reordering.  Keyed by actuator
    name (after ``joint_name_mapping``).  Useful in kinematic replay
    where the recorded command can drive joints into geometry that
    real-world contact would have stopped — e.g. a gripper closing
    further than the grasped object's thickness."""


class DataReplayTaskFileConfig(TaskFileConfig):
    """TaskFileConfig with an embedded :class:`DataReplayConfig`.

    Object/operator randomization is automatically disabled for exact
    trajectory reproduction.  Camera randomization is preserved.
    """

    replay: DataReplayConfig = DataReplayConfig()

    @model_validator(mode="after")
    def _disable_object_randomization(self):
        # Clear object/operator randomization for exact trajectory reproduction.
        # camera_randomization is a separate AutoAtomConfig field and is preserved.
        self.task.randomization.clear()
        return self


# ---------------------------------------------------------------------------
# Demo loading
# ---------------------------------------------------------------------------


def _load_low_dim_map(demo_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    if "low_dim_keys" not in demo_data:
        raise KeyError("NPZ missing 'low_dim_keys'.")
    low_dim_keys = [str(key) for key in np.asarray(demo_data["low_dim_keys"])]
    low_dim_map: Dict[str, np.ndarray] = {}
    for idx, key in enumerate(low_dim_keys):
        data_key = f"low_dim_data__{idx}"
        if data_key not in demo_data:
            raise KeyError(f"NPZ missing '{data_key}' for low-dimensional key '{key}'.")
        low_dim_map[key] = np.asarray(demo_data[data_key], dtype=np.float32)
    return low_dim_map


def _load_pose_demo(demo_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    """Load pose + gripper arrays for pose-mode replay."""
    low_dim = _load_low_dim_map(demo_data)
    result: Dict[str, np.ndarray] = {
        "position": low_dim["action/arm/pose/position"],
        "orientation": low_dim["action/arm/pose/orientation"],
    }
    gripper_key = (
        "action/gripper/joint_state/position"
        if "action/gripper/joint_state/position" in low_dim
        else "action/eef/joint_state/position"
    )
    if gripper_key in low_dim:
        result["gripper"] = low_dim[gripper_key]
    return result


def _load_ctrl_demo(demo_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    """Load joint ctrl arrays for ctrl-mode replay."""
    low_dim = _load_low_dim_map(demo_data)
    arm = low_dim["action/arm/joint_state/position"]
    eef_key = "action/eef/joint_state/position"
    if eef_key in low_dim:
        ctrl = np.concatenate([arm, low_dim[eef_key]], axis=-1)
    else:
        ctrl = arm
    return {"ctrl": ctrl}


def _get_operator_actuator_names(op_cfg: Any) -> list[str]:
    """Return ``arm_actuators + eef_actuators`` from an operator config-like object."""
    if op_cfg is None:
        return []
    arm_actuators = list(getattr(op_cfg, "arm_actuators", []) or [])
    eef_actuators = list(getattr(op_cfg, "eef_actuators", []) or [])
    return arm_actuators + eef_actuators


# ---------------------------------------------------------------------------
# McapDemo
# ---------------------------------------------------------------------------


class McapDemo:
    """Container for data loaded from a ROS2 mcap file."""

    joint: np.ndarray  # (T, n_arm + n_grip)
    joint_names: list[str]

    def __init__(self, joint: np.ndarray, joint_names: list[str]) -> None:
        self.joint = joint
        self.joint_names = joint_names

    def first_frame_joint_positions(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.joint_names, self.joint[0])}

    def align_to_actuators(
        self,
        actuator_names: list[str],
        name_mapping: Dict[str, str] | None = None,
    ) -> None:
        mapping = name_mapping or {}
        mapped_names = [mapping.get(n, n) for n in self.joint_names]
        reorder: list[int] = []
        for act_name in actuator_names:
            if act_name not in mapped_names:
                raise ValueError(
                    f"Actuator '{act_name}' not found in mcap joint names "
                    f"{self.joint_names} (after mapping: {mapped_names})"
                )
            reorder.append(mapped_names.index(act_name))
        self.joint = self.joint[:, reorder]
        self.joint_names = [actuator_names[i] for i in range(len(actuator_names))]


def _load_mcap_demo(mcap_path: str, arm_topic: str, gripper_topic: str) -> McapDemo:
    """Load arm joint + gripper arrays from a ROS2 mcap file."""
    from mcap.reader import make_reader
    from mcap_ros2idl_support import Ros2DecodeFactory

    factory = Ros2DecodeFactory()
    arm_names: list[str] | None = None
    gripper_names: list[str] | None = None
    arm_positions: list[list[float]] = []
    gripper_positions: list[list[float]] = []
    arm_times: list[int] = []
    gripper_times: list[int] = []

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[factory])
        for decoded in reader.iter_decoded_messages(topics=[arm_topic, gripper_topic]):
            topic = decoded.channel.topic
            msg = decoded.decoded_message
            t = decoded.message.log_time
            if topic == arm_topic:
                if arm_names is None:
                    arm_names = list(msg["name"])
                arm_positions.append(list(msg["position"]))
                arm_times.append(t)
            elif topic == gripper_topic:
                if gripper_names is None:
                    gripper_names = list(msg["name"])
                gripper_positions.append(list(msg["position"]))
                gripper_times.append(t)

    if not arm_positions:
        raise ValueError(f"No arm messages found on topic '{arm_topic}' in {mcap_path}")
    if not gripper_positions:
        raise ValueError(
            f"No gripper messages found on topic '{gripper_topic}' in {mcap_path}"
        )

    arm = np.array(arm_positions, dtype=np.float32)
    grip = np.array(gripper_positions, dtype=np.float32)
    arm_t = np.array(arm_times, dtype=np.int64)
    grip_t = np.array(gripper_times, dtype=np.int64)

    if len(arm_t) != len(grip_t) or np.any(np.abs(arm_t - grip_t) > 50_000_000):
        indices = np.searchsorted(grip_t, arm_t, side="left")
        indices = np.clip(indices, 0, len(grip_t) - 1)
        grip = grip[indices]

    joint = np.concatenate([arm, grip], axis=-1)
    joint_names = (arm_names or []) + (gripper_names or [])
    return McapDemo(joint=joint, joint_names=joint_names)


def _load_mcap_transform_at(
    mcap_path: str, topic: str, index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(translation xyz, rotation xyzw)`` of the ``index``-th
    ``geometry_msgs/TransformStamped`` message on ``topic`` in
    ``mcap_path``."""
    from mcap.reader import make_reader
    from mcap_ros2idl_support import Ros2DecodeFactory

    factory = Ros2DecodeFactory()
    seen = 0
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[factory])
        for decoded in reader.iter_decoded_messages(topics=[topic]):
            if seen < index:
                seen += 1
                continue
            msg = decoded.decoded_message
            t = msg["transform"]["translation"]
            r = msg["transform"]["rotation"]
            translation = np.array(
                [float(t["x"]), float(t["y"]), float(t["z"])], dtype=np.float32
            )
            rotation = np.array(
                [float(r["x"]), float(r["y"]), float(r["z"]), float(r["w"])],
                dtype=np.float32,
            )
            return translation, rotation
    raise ValueError(
        f"Fewer than {index + 1} TransformStamped message(s) on topic "
        f"'{topic}' in {mcap_path}"
    )


# ---------------------------------------------------------------------------
# Quaternion / transform helpers (xyzw convention, batched)
# ---------------------------------------------------------------------------


def _quat_inv_xyzw(q: np.ndarray) -> np.ndarray:
    """Inverse (conjugate for unit quaternions) of an xyzw quaternion."""
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (supports batched inputs)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float64)
    out[..., 0] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 1] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 2] = aw * bz + ax * by - ay * bx + az * bw
    out[..., 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _rotate_vec_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` (xyzw). Supports batched inputs."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


# ---------------------------------------------------------------------------
# Batch normalisation
# ---------------------------------------------------------------------------


def normalize_demo_for_batch(
    demo: Dict[str, np.ndarray],
    batch_size: int,
    mode: str,
) -> Dict[str, np.ndarray]:
    """Slice a (T, B_rec, dim) demo to match the replay *batch_size*."""
    if mode == "pose":
        position = demo["position"]
        orientation = demo["orientation"]
        rec_bs = position.shape[1]
        if batch_size > rec_bs:
            raise ValueError(
                f"Demo recorded with batch_size={rec_bs}, "
                f"but replay requires batch_size={batch_size}."
            )
        position = position[:, :batch_size, :]
        orientation = orientation[:, :batch_size, :]
        result: Dict[str, np.ndarray] = {
            "position": position[:, 0, :] if batch_size == 1 else position,
            "orientation": orientation[:, 0, :] if batch_size == 1 else orientation,
        }
        if "gripper" in demo:
            gripper = demo["gripper"][:, :batch_size, :]
            result["gripper"] = gripper[:, 0, :] if batch_size == 1 else gripper
        return result

    if mode == "joint":
        joint = demo["joint"]
        if joint.ndim == 2:
            return {"joint": joint}
        rec_bs = joint.shape[1]
        if batch_size > rec_bs:
            raise ValueError(
                f"Demo recorded with batch_size={rec_bs}, "
                f"but replay requires batch_size={batch_size}."
            )
        joint = joint[:, :batch_size, :]
        return {"joint": joint[:, 0, :] if batch_size == 1 else joint}

    ctrl = demo["ctrl"]
    rec_bs = ctrl.shape[1]
    if batch_size > rec_bs:
        raise ValueError(
            f"Demo recorded with batch_size={rec_bs}, "
            f"but replay requires batch_size={batch_size}."
        )
    ctrl = ctrl[:, :batch_size, :]
    return {"ctrl": ctrl[:, 0, :] if batch_size == 1 else ctrl}


# ---------------------------------------------------------------------------
# ReplayPolicy
# ---------------------------------------------------------------------------


class ReplayPolicy:
    """Replays recorded actions step by step (index-based, no observation)."""

    def __init__(self, demo: Dict[str, np.ndarray], mode: str) -> None:
        self._demo = demo
        self._mode = mode
        if mode == "pose":
            self._max = len(demo["position"]) - 1
        elif mode == "joint":
            self._max = len(demo["joint"]) - 1
        else:
            self._max = len(demo["ctrl"]) - 1
        self._step = 0

    def reset(self, start_step: int = 0) -> None:
        self._step = max(0, int(start_step))

    @property
    def num_steps(self) -> int:
        return self._max + 1

    @property
    def remaining_steps(self) -> int:
        return max(self._max - self._step + 1, 0)

    def _action_at(self, index: int) -> Dict[str, Any]:
        i = min(max(0, index), self._max)
        if self._mode == "pose":
            action: Dict[str, Any] = {
                "position": self._demo["position"][i],
                "orientation": self._demo["orientation"][i],
            }
            if "gripper" in self._demo:
                action["gripper"] = self._demo["gripper"][i]
            return action
        if self._mode == "joint":
            return {"joint": self._demo["joint"][i]}
        return {"ctrl": self._demo["ctrl"][i]}

    def apply_first_frame_as_reset(self) -> Dict[str, Any] | None:
        if self.num_steps <= 0:
            return None
        action = self._action_at(0)
        self._step = min(1, self.num_steps)
        return action

    def act(self) -> Dict[str, Any]:
        action = self._action_at(self._step)
        self._step += 1
        return action


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _apply_reset_action(context: ExecutionContext, action: Any) -> None:
    """Apply a recorded action as an exact reset state when possible."""
    if action is None:
        return
    env = context.backend.env
    if "joint" in action:
        env.apply_joint_action("arm", action["joint"], kinematic=True)
    elif "ctrl" in action:
        ctrl = np.asarray(action["ctrl"], dtype=np.float64)
        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(1, -1).repeat(env.batch_size, axis=0)
        env.step(ctrl)
    else:
        env.apply_pose_action(
            "arm",
            action["position"],
            action["orientation"],
            action.get("gripper"),
            kinematic=True,
        )


def _make_replay_action_applier(kinematic: bool = False):
    """Return an action applier closure with the configured *kinematic* flag."""

    print(f"Replay action applier created with kinematic={kinematic}")

    def replay_action_applier(
        context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
    ) -> None:
        if action is None:
            return
        env = context.backend.env
        if "joint" in action:
            env.apply_joint_action(
                "arm", action["joint"], env_mask=env_mask, kinematic=kinematic
            )
        elif "ctrl" in action:
            ctrl = np.asarray(action["ctrl"], dtype=np.float64)
            if ctrl.ndim == 1:
                ctrl = ctrl.reshape(1, -1).repeat(env.batch_size, axis=0)
            env.step(ctrl, env_mask=env_mask)
        else:
            env.apply_pose_action(
                "arm",
                action["position"],
                action["orientation"],
                action.get("gripper"),
                env_mask=env_mask,
                kinematic=kinematic,
            )

    return replay_action_applier


def _query_entity_world(env: Any, ref: SimEntityRef) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(B, 3), (B, 4)`` world pose of the referenced entity."""
    if ref.kind == "site":
        return env.get_site_pose(ref.name)
    if ref.kind == "body":
        return env.get_body_pose(ref.name)
    if ref.kind == "operator_base":
        return env.get_operator_base_pose(ref.name)
    raise ValueError(f"Unsupported SimEntityRef.kind: {ref.kind!r}")


def _apply_entity_world(
    env: Any,
    ref: SimEntityRef,
    pos: np.ndarray,
    quat: np.ndarray,
    env_mask: Optional[np.ndarray],
) -> None:
    """Set the referenced entity's world pose (only kinds that support
    relocation can be used as a ``move`` target)."""
    if ref.kind == "operator_base":
        env.override_operator_base_pose(ref.name, pos, quat, env_mask=env_mask)
        return
    raise ValueError(
        f"SimEntityRef.kind={ref.kind!r} is not supported as a move target "
        f"(currently supported: operator_base)."
    )


def _invert_transform_xyzw(
    translation: np.ndarray, rotation_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a rigid transform expressed as (translation, xyzw quat)."""
    inv_q = _quat_inv_xyzw(rotation_xyzw)
    inv_t = -_rotate_vec_xyzw(inv_q, translation)
    return inv_t, inv_q


def _apply_joint_axis_scale(
    joint_data: np.ndarray,
    joint_axis_scale: List[float],
    *,
    joint_names: Optional[List[str]] = None,
    label: str = "joint data",
) -> np.ndarray:
    """Return a copy of ``joint_data`` with per-column scale factors applied.

    Scaling is applied along the last dimension. If fewer scale factors than
    columns are provided, only the first ``N`` columns are modified and the
    remainder keep a factor of ``1.0``.
    """
    if not joint_axis_scale:
        return joint_data

    data = np.asarray(joint_data)
    if data.ndim == 0:
        raise ValueError(f"{label} must have at least 1 dimension, got {data.shape}")

    scale = np.asarray(joint_axis_scale, dtype=data.dtype)
    num_scaled = int(scale.shape[0])
    if data.shape[-1] < num_scaled:
        raise ValueError(
            f"joint_axis_scale has length {num_scaled}, but {label} only has "
            f"{data.shape[-1]} column(s)."
        )

    scaled = data.copy()
    scaled[..., :num_scaled] *= scale

    name_parts = []
    for idx, factor in enumerate(scale.tolist()):
        if joint_names is not None and idx < len(joint_names):
            name_parts.append(f"{joint_names[idx]} x {factor:g}")
        else:
            name_parts.append(f"col{idx} x {factor:g}")
    trailing = ""
    if num_scaled < data.shape[-1]:
        trailing = (
            f"; remaining {data.shape[-1] - num_scaled} trailing column(s) keep x 1.0"
        )
    print(f"[joint_axis_scale] {label}: " + ", ".join(name_parts) + trailing)
    return scaled


def _prepare_mcap_demo_for_replay(
    mcap_demo: McapDemo,
    actuator_names: list[str],
    replay_cfg: DataReplayConfig,
) -> None:
    """Align and transform MCAP joint data into replay actuator order."""
    if actuator_names:
        mcap_demo.align_to_actuators(
            actuator_names, replay_cfg.joint_name_mapping or None
        )
    mcap_demo.joint = _apply_joint_axis_scale(
        mcap_demo.joint,
        replay_cfg.joint_axis_scale,
        joint_names=mcap_demo.joint_names,
        label="mcap joint replay",
    )
    _apply_joint_clip_to_mcap_demo(mcap_demo, replay_cfg.joint_clip)


def _apply_joint_clip_to_mcap_demo(
    mcap_demo: McapDemo,
    joint_clip: Dict[str, JointClipBounds],
) -> None:
    """Clamp ``mcap_demo.joint`` columns in-place per ``joint_clip`` rules.

    Must be called *after* :meth:`McapDemo.align_to_actuators` so that
    ``mcap_demo.joint_names`` already matches the actuator name order.
    """
    if not joint_clip:
        return
    for idx, name in enumerate(mcap_demo.joint_names):
        bounds = joint_clip.get(name)
        if bounds is None:
            continue
        col = mcap_demo.joint[:, idx]
        before_min = float(np.min(col))
        before_max = float(np.max(col))
        if bounds.min is not None:
            col = np.maximum(col, bounds.min)
        if bounds.max is not None:
            col = np.minimum(col, bounds.max)
        mcap_demo.joint[:, idx] = col
        print(
            f"[joint_clip] '{name}': "
            f"raw range=[{before_min:.4f}, {before_max:.4f}] -> "
            f"clipped range=[{float(np.min(col)):.4f}, {float(np.max(col)):.4f}] "
            f"(min={bounds.min}, max={bounds.max})"
        )


def _resolve_pose_offset(offset: "PoseOffset") -> tuple[np.ndarray, np.ndarray]:
    """Convert a :class:`PoseOffset` into ``(position (3,), quat xyzw (4,))``.

    Accepts either 4-element (xyzw quat) or 3-element (intrinsic XYZ
    euler radians) ``orientation`` lists.
    """
    from auto_atom.utils.pose import euler_to_quaternion

    pos = np.asarray(offset.position, dtype=np.float64)
    if pos.shape != (3,):
        raise ValueError(
            f"PoseOffset.position must be 3 floats, got {offset.position!r}"
        )
    ori = list(offset.orientation)
    if len(ori) == 4:
        quat = np.asarray(ori, dtype=np.float64)
    elif len(ori) == 3:
        quat = np.asarray(
            euler_to_quaternion((ori[0], ori[1], ori[2])), dtype=np.float64
        )
    else:
        raise ValueError(
            f"PoseOffset.orientation must be 3 (euler) or 4 (xyzw quat) floats, "
            f"got {offset.orientation!r}"
        )
    return pos, quat


def _apply_transform_resets(
    evaluator: PolicyEvaluator,
    replay_cfg: "DataReplayConfig",
    env_mask: Optional[np.ndarray],
) -> None:
    """Apply every configured :class:`TransformResetConfig` to the scene.

    Reads the referenced MCAP transform once per rule, queries the fixed
    side's current world pose, then repositions the ``move``-side entity
    so that the simulated relative pose matches the recording.
    """
    if replay_cfg.mcap_path is None or not replay_cfg.transform_resets:
        return
    mcap_path = replay_cfg.mcap_path
    if not os.path.isabs(mcap_path):
        mcap_path = os.path.join(os.getcwd(), mcap_path)
    if not os.path.exists(mcap_path):
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    context = evaluator._context
    if context is None:
        raise RuntimeError(
            "PolicyEvaluator must be initialized before applying transform resets."
        )
    env = context.backend.env

    for tr in replay_cfg.transform_resets:
        t_pc, q_pc = _load_mcap_transform_at(mcap_path, tr.topic, tr.message_index)
        p_wp, q_wp = _query_entity_world(env, tr.parent)
        p_wc, q_wc = _query_entity_world(env, tr.child)
        # p_wp/q_wp and p_wc/q_wc are batched (B, 3) / (B, 4)

        if tr.move == "parent":
            move_ref = tr.parent
            cur_pos, cur_quat = np.asarray(p_wp), np.asarray(q_wp)
            fixed_pos, fixed_quat = np.asarray(p_wc), np.asarray(q_wc)
            # Desired T_W_parent = T_W_child * T_parent_child^-1
            inv_t, inv_q = _invert_transform_xyzw(t_pc, q_pc)
            if tr.use_orientation:
                new_quat = _quat_mul_xyzw(fixed_quat, inv_q).astype(np.float32)
                new_pos = (fixed_pos + _rotate_vec_xyzw(fixed_quat, inv_t)).astype(
                    np.float32
                )
            else:
                new_quat = cur_quat.astype(np.float32)
                # Keep current orientation → p_W_parent = p_W_child - R(q_W_parent) * t_pc
                new_pos = (fixed_pos - _rotate_vec_xyzw(new_quat, t_pc)).astype(
                    np.float32
                )
        elif tr.move == "child":
            move_ref = tr.child
            cur_pos, cur_quat = np.asarray(p_wc), np.asarray(q_wc)
            fixed_pos, fixed_quat = np.asarray(p_wp), np.asarray(q_wp)
            # Desired T_W_child = T_W_parent * T_parent_child
            if tr.use_orientation:
                new_quat = _quat_mul_xyzw(fixed_quat, q_pc).astype(np.float32)
            else:
                new_quat = cur_quat.astype(np.float32)
            new_pos = (fixed_pos + _rotate_vec_xyzw(fixed_quat, t_pc)).astype(
                np.float32
            )
        else:
            raise ValueError(f"Unknown TransformResetConfig.move: {tr.move!r}")

        # Apply the optional calibration offset.
        #   position: additive in world frame
        #   orientation: right-multiplied (movable entity's local frame)
        off_pos, off_quat = _resolve_pose_offset(tr.offset)
        off_identity = np.allclose(off_pos, 0.0) and np.allclose(
            off_quat, np.array([0.0, 0.0, 0.0, 1.0])
        )
        if not off_identity:
            new_pos = (new_pos + off_pos).astype(np.float32)
            new_quat = _quat_mul_xyzw(new_quat, off_quat).astype(np.float32)

        with evaluator.sim_lock:
            _apply_entity_world(env, move_ref, new_pos, new_quat, env_mask)
        offset_str = (
            ""
            if off_identity
            else (f" (+offset pos={off_pos.tolist()} quat={off_quat.tolist()})")
        )
        print(
            f"[transform_reset] topic={tr.topic!r} "
            f"moved {move_ref.kind}:{move_ref.name} "
            f"from pos={cur_pos[0].tolist()} to pos={new_pos[0].tolist()}"
            f"{offset_str}"
        )


def _apply_first_frame_reset(
    evaluator: PolicyEvaluator,
    policy: ReplayPolicy,
) -> Dict[str, Any] | None:
    """Apply the first recorded action as the post-reset initial state."""
    reset_action = policy.apply_first_frame_as_reset()
    if reset_action is None:
        return None
    context = evaluator._context
    if context is None:
        raise RuntimeError("PolicyEvaluator must be initialized before applying reset.")
    with evaluator.sim_lock:
        # print(f"{reset_action=}")
        _apply_reset_action(context, reset_action)
    return reset_action


# ---------------------------------------------------------------------------
# DataReplayRunner
# ---------------------------------------------------------------------------


def preprocess_replay_dictconfig(
    cfg: Any,
    replay_cfg: DataReplayConfig,
) -> None:
    """Pre-process a Hydra DictConfig for mcap replay **before** ``prepare_task_file``.

    This injects ``initial_joint_positions`` into ``cfg.env`` so the backend
    resets the robot at the recorded starting configuration.

    Call this *before* ``prepare_task_file(cfg)`` when ``replay_cfg.mcap_path``
    is set.  For npz demos no pre-processing is needed.
    """
    from omegaconf import open_dict

    mcap_path = replay_cfg.mcap_path
    if mcap_path is None:
        return

    if not os.path.isabs(mcap_path):
        mcap_path = os.path.join(os.getcwd(), mcap_path)
    if not os.path.exists(mcap_path):
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    mcap_demo = _load_mcap_demo(
        mcap_path, replay_cfg.arm_topic, replay_cfg.gripper_topic
    )
    replay_cfg.mode = "joint"

    op_cfg = cfg.env.operators.arm
    actuator_names = _get_operator_actuator_names(op_cfg)
    _prepare_mcap_demo_for_replay(mcap_demo, actuator_names, replay_cfg)

    init_jpos = mcap_demo.first_frame_joint_positions()
    # When eef_mapper is configured, the mcap eef values are in user-space
    # (e.g. finger distance), not raw joint values.  Exclude them from
    # initial_joint_positions (which writes directly to qpos) — the reset
    # action path will apply them through the mapper instead.
    has_eef_mapper = getattr(op_cfg, "eef_mapper", None) is not None
    if has_eef_mapper:
        eef_names = set(op_cfg.eef_actuators)
        init_jpos = {k: v for k, v in init_jpos.items() if k not in eef_names}
    with open_dict(cfg):
        if "initial_joint_positions" not in cfg.env:
            cfg.env.initial_joint_positions = {}
        cfg.env.initial_joint_positions.update(init_jpos)
    print(f"Injected initial_joint_positions: {init_jpos}")


class DataReplayRunner(RunnerBase):
    """Runner that replays recorded demonstration data.

    Wraps :class:`PolicyEvaluator` and :class:`ReplayPolicy` behind the
    standard ``RunnerBase`` interface so it can be driven by a manager in the
    same way as :class:`TaskRunner`.
    """

    def __init__(
        self,
        observation_getter: Optional[Any] = None,
    ) -> None:
        self._observation_getter = observation_getter
        self._evaluator: Optional[PolicyEvaluator] = None
        self._policy: Optional[ReplayPolicy] = None
        self._replay_cfg: Optional[DataReplayConfig] = None
        self._config: Optional[DataReplayTaskFileConfig] = None
        self._current_action: Optional[Dict[str, Any]] = None
        self._action_step: int = 0

    def from_config(self, cfg) -> DataReplayRunner:
        from omegaconf import DictConfig

        if isinstance(cfg, DictConfig):
            from omegaconf import OmegaConf, open_dict

            from .common import prepare_task_file

            replay_raw = OmegaConf.to_container(cfg.get("replay", {}), resolve=True)
            replay_cfg = DataReplayConfig.model_validate(replay_raw or {})
            preprocess_replay_dictconfig(cfg, replay_cfg)
            with open_dict(cfg):
                cfg.replay = OmegaConf.create(replay_cfg.model_dump())
            config = prepare_task_file(cfg, config_cls=DataReplayTaskFileConfig)
        else:
            config = cfg

        replay_raw = config.replay
        if isinstance(replay_raw, DataReplayConfig):
            self._replay_cfg = replay_raw
        else:
            self._replay_cfg = DataReplayConfig.model_validate(replay_raw)
        self._config = config

        # --- Build evaluator ---
        self._evaluator = PolicyEvaluator(
            action_applier=_make_replay_action_applier(self._replay_cfg.kinematic),
            observation_getter=self._observation_getter,
        ).from_config(config)

        # --- Load initial demo data ---
        self._load_demo()
        return self

    def set_demo_path(
        self,
        *,
        demo_name: Optional[str] = None,
        demo_dir: Optional[str] = None,
        mcap_path: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        """Change the demo data source.  Takes effect on the next ``reset()``.

        Pass *mcap_path* for mcap replay, or *demo_name* / *demo_dir* for npz
        replay.  Only the provided arguments are updated; the rest keep their
        current values.
        """
        rcfg = self._require_replay_cfg()
        if mcap_path is not None:
            rcfg.mcap_path = mcap_path
        if demo_name is not None:
            rcfg.demo_name = demo_name
        if demo_dir is not None:
            rcfg.demo_dir = demo_dir
        if mode is not None:
            rcfg.mode = mode
        # Mark policy as stale so _load_demo() runs on next reset().
        self._policy = None

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        evaluator = self._require_evaluator()
        # Reload demo data if set_demo_path() invalidated the policy.
        if self._policy is None:
            self._load_demo()
        policy = self._require_policy()
        policy.reset()
        self._current_action = None
        self._action_step = 0
        update = evaluator.reset(env_mask)
        if self._replay_cfg is not None:
            _apply_transform_resets(evaluator, self._replay_cfg, env_mask)
            if self._replay_cfg.reset_from_first_frame:
                _apply_first_frame_reset(evaluator, policy)
        return update

    def update(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        evaluator = self._require_evaluator()
        policy = self._require_policy()
        steps_per_action = self._replay_cfg.steps_per_action if self._replay_cfg else 1

        # Advance to the next recorded action when the sub-step budget for
        # the current action is exhausted.
        if self._current_action is None or self._action_step >= steps_per_action:
            if policy.remaining_steps > 0:
                self._current_action = policy.act()
            else:
                self._current_action = None
            self._action_step = 0

        self._action_step += 1
        print(f"action={self._current_action}")
        input("Press Enter to continue to the next step...")
        task_update = evaluator.update(self._current_action, env_mask)

        # Defer done for successful envs until replay data is exhausted.
        done_on_success = self._replay_cfg.done_on_success if self._replay_cfg else True
        if not done_on_success and policy.remaining_steps > 0:
            task_update.done[task_update.success] = False

        return task_update

    def close(self) -> None:
        if self._evaluator is not None:
            self._evaluator.close()
            self._evaluator = None
        self._policy = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._require_evaluator().batch_size

    @property
    def remaining_steps(self) -> int:
        return self._require_policy().remaining_steps

    def get_observation(self) -> Any:
        return self._require_evaluator().get_observation()

    def summarize(
        self,
        update: Optional[TaskUpdate] = None,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
        elapsed_time_sec: float = 0.0,
    ) -> Any:
        return self._require_evaluator().summarize(
            update,
            max_updates=max_updates,
            updates_used=updates_used,
            elapsed_time_sec=elapsed_time_sec,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_demo(self) -> None:
        """(Re)load demo data from the current replay config and build a new
        :class:`ReplayPolicy`."""
        rcfg = self._require_replay_cfg()
        config = self._config

        if rcfg.mcap_path is not None:
            mcap_path = rcfg.mcap_path
            if not os.path.isabs(mcap_path):
                mcap_path = os.path.join(os.getcwd(), mcap_path)
            if not os.path.exists(mcap_path):
                raise FileNotFoundError(f"MCAP file not found: {mcap_path}")
            mcap_demo = _load_mcap_demo(mcap_path, rcfg.arm_topic, rcfg.gripper_topic)
            rcfg.mode = "joint"

            # Align mcap column order to the YAML actuator declaration order.
            from auto_atom import ComponentRegistry

            env = ComponentRegistry.get_env(config.task.env_name)
            op_binding = env.config.operators.get("arm")
            if op_binding is not None:
                actuator_names = _get_operator_actuator_names(op_binding)
            else:
                actuator_names = []
            _prepare_mcap_demo_for_replay(mcap_demo, actuator_names, rcfg)

            demo: Dict[str, np.ndarray] = {"joint": mcap_demo.joint}
        else:
            demo_name = rcfg.demo_name or "demo"
            demo_dir = rcfg.demo_dir or os.path.join(
                os.getcwd(), "outputs", "records", "demos"
            )
            demo_npz_path = os.path.join(demo_dir, f"{demo_name}.npz")
            if not os.path.exists(demo_npz_path):
                raise FileNotFoundError(f"Demo file not found: {demo_npz_path}")
            demo_data = np.load(demo_npz_path)
            if rcfg.mode == "pose":
                demo = _load_pose_demo(demo_data)
            elif rcfg.mode == "ctrl":
                demo = _load_ctrl_demo(demo_data)
                actuator_names: list[str] = []
                from auto_atom import ComponentRegistry

                env = ComponentRegistry.get_env(config.task.env_name)
                op_binding = env.config.operators.get("arm")
                if op_binding is not None:
                    actuator_names = _get_operator_actuator_names(op_binding)
                demo["ctrl"] = _apply_joint_axis_scale(
                    demo["ctrl"],
                    rcfg.joint_axis_scale,
                    joint_names=actuator_names or None,
                    label="npz ctrl replay",
                )
            else:
                raise ValueError(
                    f"Unknown replay mode: {rcfg.mode!r} "
                    f"(expected 'pose', 'ctrl', or set mcap_path)"
                )

        batch_size = self._require_evaluator().batch_size
        demo = normalize_demo_for_batch(demo, batch_size=batch_size, mode=rcfg.mode)
        self._policy = ReplayPolicy(demo, rcfg.mode)

    def _require_replay_cfg(self) -> DataReplayConfig:
        if self._replay_cfg is None:
            raise RuntimeError(
                "DataReplayRunner is not initialized. Call from_config() first."
            )
        return self._replay_cfg

    def _require_evaluator(self) -> PolicyEvaluator:
        if self._evaluator is None:
            raise RuntimeError(
                "DataReplayRunner is not initialized. Call from_config() first."
            )
        return self._evaluator

    def _require_policy(self) -> ReplayPolicy:
        if self._policy is None:
            raise RuntimeError(
                "DataReplayRunner has no demo loaded. "
                "Call from_config() or set_demo_path() + reset() first."
            )
        return self._policy
