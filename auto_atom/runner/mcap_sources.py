"""MCAP source adapters independent of the replay runner lifecycle.

This module owns format detection, decoding, timestamp alignment, and the
legacy ``McapDemo`` value object.  It depends only on the canonical trajectory
model; the runner can therefore consume recordings without a reverse import.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .replay_model import ReplayTrajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# McapDemo
# ---------------------------------------------------------------------------


class McapDemo:
    """Container for data loaded from a ROS2 or Foxglove flatbuffer mcap file."""

    joint: np.ndarray  # (T, n_arm + n_grip)
    joint_names: list[str]
    joint_times: np.ndarray | None
    base_position: np.ndarray | None
    base_orientation: np.ndarray | None
    scene_joint: np.ndarray | None
    scene_joint_names: list[str]

    def __init__(
        self,
        joint: np.ndarray,
        joint_names: list[str],
        *,
        joint_times: np.ndarray | None = None,
        base_position: np.ndarray | None = None,
        base_orientation: np.ndarray | None = None,
        scene_joint: np.ndarray | None = None,
        scene_joint_names: list[str] | None = None,
    ) -> None:
        self.joint = joint
        self.joint_names = joint_names
        self.joint_times = joint_times
        self.base_position = base_position
        self.base_orientation = base_orientation
        self.scene_joint = scene_joint
        self.scene_joint_names = list(scene_joint_names or [])

    def first_frame_joint_positions(self) -> Dict[str, float]:
        result = {n: float(v) for n, v in zip(self.joint_names, self.joint[0])}
        if self.scene_joint is not None:
            result.update(
                {
                    n: float(v)
                    for n, v in zip(self.scene_joint_names, self.scene_joint[0])
                }
            )
        return result

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

    def to_trajectory(self) -> ReplayTrajectory:
        """Return the canonical recording view used by the replay runner."""
        return ReplayTrajectory.from_mcap_demo(self)


def _nearest_sample_indices(
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Return indices of ``source_times`` nearest to each ``target_times`` item."""

    source_times = np.asarray(source_times, dtype=np.int64).reshape(-1)
    target_times = np.asarray(target_times, dtype=np.int64).reshape(-1)
    if source_times.size == 0:
        raise ValueError("Cannot align against an empty timestamp array.")
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.clip(right - 1, 0, source_times.size - 1)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right)


def _align_samples_to_times(
    samples: np.ndarray,
    sample_times: np.ndarray,
    target_times: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Nearest-neighbour align sample rows to target timestamps."""

    samples = np.asarray(samples)
    sample_times = np.asarray(sample_times, dtype=np.int64).reshape(-1)
    if samples.shape[0] != sample_times.shape[0]:
        raise ValueError(
            f"{label} sample count ({samples.shape[0]}) does not match timestamp "
            f"count ({sample_times.shape[0]})."
        )
    order = np.argsort(sample_times)
    sorted_times = sample_times[order]
    sorted_samples = samples[order]
    return sorted_samples[_nearest_sample_indices(sorted_times, target_times)]


def _extract_pose_stamped_xyzw(
    msg: Any, *, topic: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(position xyz, orientation xyzw)`` from a PoseStamped-like dict."""

    if not isinstance(msg, dict):
        raise TypeError(
            f"Expected PoseStamped message on topic '{topic}' to decode as dict, "
            f"got {type(msg).__name__}."
        )
    pose = msg.get("pose")
    if not isinstance(pose, dict):
        raise ValueError(
            f"Expected PoseStamped message on topic '{topic}' to contain a 'pose' dict."
        )
    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        raise ValueError(
            f"Expected PoseStamped message on topic '{topic}' to contain "
            "pose.position and pose.orientation dicts."
        )
    pos = np.array(
        [float(position["x"]), float(position["y"]), float(position["z"])],
        dtype=np.float32,
    )
    quat = np.array(
        [
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        ],
        dtype=np.float32,
    )
    return pos, quat


def _extract_joint_state_positions(
    msg: Any,
    *,
    topic: str,
    expected_names: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Extract ``(joint_names, positions)`` from a JointState-like dict."""

    if not isinstance(msg, dict):
        raise TypeError(
            f"Expected JointState message on topic '{topic}' to decode as dict, "
            f"got {type(msg).__name__}."
        )
    names_raw = msg.get("name")
    pos_raw = msg.get("position")
    if names_raw is None or pos_raw is None:
        raise ValueError(
            f"Expected JointState message on topic '{topic}' to contain "
            "'name' and 'position' fields."
        )
    names = [str(name) for name in names_raw]
    positions = np.asarray(pos_raw, dtype=np.float32).reshape(-1)
    if len(names) != positions.shape[0]:
        raise ValueError(
            f"JointState message on topic '{topic}' has {len(names)} names but "
            f"{positions.shape[0]} positions."
        )
    if expected_names is None:
        return names, positions
    index_by_name = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in expected_names if name not in index_by_name]
    if missing:
        raise ValueError(
            f"JointState message on topic '{topic}' is missing joint(s) {missing}; "
            f"available names: {names}"
        )
    reordered = np.asarray(
        [positions[index_by_name[name]] for name in expected_names], dtype=np.float32
    )
    return list(expected_names), reordered


def _align_optional_scene_joint(
    scene_joint_positions: list[np.ndarray],
    scene_joint_times: list[int],
    arm_times: np.ndarray,
    *,
    scene_joint_topic: str,
    mcap_path: str,
) -> np.ndarray | None:
    """Align optional scene-joint samples or skip cleanly when absent."""

    if not scene_joint_positions:
        logger.warning(
            f"[scene_joint_topic] No JointState messages found on "
            f"'{scene_joint_topic}' in {mcap_path}; skipping scene joint replay."
        )
        return None

    scene_joint_t = np.array(scene_joint_times, dtype=np.int64)
    return _align_samples_to_times(
        np.asarray(scene_joint_positions, dtype=np.float32),
        scene_joint_t,
        arm_times,
        label=f"scene joint topic '{scene_joint_topic}'",
    )


# ---------------------------------------------------------------------------
# Foxglove flatbuffer demos (new recording format)
# ---------------------------------------------------------------------------
#
# Newer recordings (e.g. ``data/replay/george.mcap``) are written with the
# Foxglove flatbuffer message encoding instead of ROS2 CDR.  Their schemas are
# ``foxglove.JointStates`` (a ``timestamp`` + a vector of ``JointState``
# {name, position, velocity, acceleration, effort}) and ``foxglove.PoseInFrame``
# ({timestamp, frame_id, pose{position, orientation}}).  Arm and gripper live on
# separate action topics, each with locally-indexed joint names (``j0..jN``), so
# the columns are canonicalised to the target operator's actuator names by
# position (arm topic -> arm_actuators, gripper topic -> eef_actuators).

_FLB_JOINT_STATES_SCHEMA = "foxglove.JointStates"
_FLB_POSE_IN_FRAME_SCHEMA = "foxglove.PoseInFrame"
_FLB_MESSAGE_ENCODING = "flatbuffer"

# Default Foxglove (airbot_play) replay topics, used when the configured ROS2
# topics are absent from a flatbuffer recording.  These are the *commanded*
# ``/action/`` streams (not the measured ``/observation/`` state): in a
# position-controlled replay the actuator force comes from the gap between the
# commanded target and the achieved joint position, so the command is what
# reproduces contact/grasp forces (pushing a door, closing onto a handle).
# The measured state has already absorbed that compliance and would replay with
# ~0 tracking error, losing the interaction forces.
_FLB_DEFAULT_ARM_TOPIC = "/action/airbot_play/arm/joint_position"
_FLB_DEFAULT_GRIPPER_TOPIC = "/action/airbot_play/g2t/parallel_position"


def _read_mcap_channel_meta(mcap_path: str) -> Dict[str, tuple[str, str]]:
    """Return ``{topic: (schema_name, message_encoding)}`` for every channel."""
    from mcap.reader import make_reader

    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
        meta: Dict[str, tuple[str, str]] = {}
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            meta[channel.topic] = (
                schema.name if schema is not None else "",
                channel.message_encoding,
            )
    return meta


def _mcap_is_foxglove(channel_meta: Dict[str, tuple[str, str]]) -> bool:
    """True when the recording carries Foxglove flatbuffer JointStates."""
    return any(
        schema_name == _FLB_JOINT_STATES_SCHEMA
        for schema_name, _encoding in channel_meta.values()
    )


def _decode_foxglove_joint_states(data: bytes) -> tuple[list[str], np.ndarray]:
    """Decode a ``foxglove.JointStates`` flatbuffer into ``(names, positions)``.

    ``positions`` is a ``float32`` array with one entry per joint.  The
    embedded ``timestamp`` is ignored — replay timing uses the MCAP log time.
    """
    import flatbuffers
    from flatbuffers import number_types as N
    from flatbuffers.table import Table

    root = Table(data, flatbuffers.encode.Get(N.UOffsetTFlags.packer_type, data, 0))
    joints_off = root.Offset(6)  # JointStates.joints (field id=1)
    names: list[str] = []
    positions: list[float] = []
    if joints_off:
        vector_start = root.Vector(joints_off)
        for i in range(root.VectorLen(joints_off)):
            joint = Table(data, root.Indirect(vector_start + i * 4))
            name_off = joint.Offset(4)  # JointState.name (field id=0)
            pos_off = joint.Offset(6)  # JointState.position (field id=1)
            names.append(
                joint.String(name_off + joint.Pos).decode() if name_off else ""
            )
            positions.append(
                joint.Get(N.Float64Flags, pos_off + joint.Pos) if pos_off else 0.0
            )
    return names, np.asarray(positions, dtype=np.float32)


def _decode_foxglove_pose_in_frame(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode a ``foxglove.PoseInFrame`` flatbuffer into ``(position, xyzw)``."""
    import flatbuffers
    from flatbuffers import number_types as N
    from flatbuffers.table import Table

    def _f64(tbl: Table, vtable_off: int) -> float:
        off = tbl.Offset(vtable_off)
        return tbl.Get(N.Float64Flags, off + tbl.Pos) if off else 0.0

    def _subtable(tbl: Table, vtable_off: int) -> Table | None:
        off = tbl.Offset(vtable_off)
        return Table(tbl.Bytes, tbl.Indirect(off + tbl.Pos)) if off else None

    root = Table(data, flatbuffers.encode.Get(N.UOffsetTFlags.packer_type, data, 0))
    pose = _subtable(root, 8)  # PoseInFrame.pose (field id=2)
    if pose is None:
        raise ValueError("foxglove.PoseInFrame message is missing its 'pose' field.")
    vec = _subtable(pose, 4)  # Pose.position (field id=0) -> Vector3
    quat = _subtable(pose, 6)  # Pose.orientation (field id=1) -> Quaternion
    position = np.array(
        [_f64(vec, 4), _f64(vec, 6), _f64(vec, 8)] if vec else [0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    orientation = np.array(
        [_f64(quat, 4), _f64(quat, 6), _f64(quat, 8), _f64(quat, 10)]
        if quat
        else [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return position, orientation


def _resolve_foxglove_joint_topic(
    channel_meta: Dict[str, tuple[str, str]],
    configured: str | None,
    default_topic: str,
    role_keywords: tuple[str, ...],
    role: str,
) -> str:
    """Pick the JointStates topic for ``role`` in a flatbuffer recording.

    Honors an explicitly-configured topic when it exists in the file; otherwise
    falls back to the airbot default, then to auto-discovery by keyword. Raises
    a clear error listing candidates when nothing matches.
    """

    def _is_joint_states(topic: str) -> bool:
        return channel_meta.get(topic, ("", ""))[0] == _FLB_JOINT_STATES_SCHEMA

    if configured and _is_joint_states(configured):
        return configured
    if _is_joint_states(default_topic):
        return default_topic

    candidates = [
        topic
        for topic, (schema_name, _enc) in channel_meta.items()
        if schema_name == _FLB_JOINT_STATES_SCHEMA
    ]
    matched = [
        topic
        for topic in candidates
        if any(keyword in topic for keyword in role_keywords)
    ]
    if matched:
        # Prefer commanded ``/action/`` streams over ``/observation/`` ones:
        # the command carries the interaction force (see the default-topic note).
        matched.sort(key=lambda t: (not t.startswith("/action/"), t))
        return matched[0]
    raise ValueError(
        f"Could not resolve a Foxglove {role} JointStates topic "
        f"(configured={configured!r}, default={default_topic!r}). "
        f"Available JointStates topics: {sorted(candidates)}"
    )


def _stack_joint_positions(positions: list[np.ndarray], *, topic: str) -> np.ndarray:
    """Stack per-message joint rows, guarding against a ragged joint count."""
    widths = {row.shape[0] for row in positions}
    if len(widths) != 1:
        raise ValueError(
            f"JointStates topic '{topic}' has inconsistent joint counts {widths}."
        )
    return np.asarray(positions, dtype=np.float32)


def _canonicalize_foxglove_names(
    recorded_names: list[str],
    actuator_names: list[str] | None,
    *,
    role: str,
    topic: str,
    fallback_prefix: str,
) -> list[str]:
    """Rename a JointStates block's columns to the operator's actuator names.

    Foxglove recordings index joints locally per topic (``j0..jN``), so the arm
    and gripper blocks both start at ``j0``.  When the operator's actuator names
    are known they are assigned positionally (recordings list joints in actuator
    order); otherwise the recorded names are kept but prefixed to stay unique
    across blocks.
    """
    if actuator_names:
        if len(recorded_names) != len(actuator_names):
            raise ValueError(
                f"Foxglove {role} topic '{topic}' has {len(recorded_names)} joint(s) "
                f"{recorded_names}, but the operator declares {len(actuator_names)} "
                f"{role} actuator(s) {actuator_names}. Provide matching actuators or "
                f"an explicit joint_name_mapping."
            )
        return list(actuator_names)
    return [f"{fallback_prefix}{name}" for name in recorded_names]


# ---------------------------------------------------------------------------
# Demo loading (dispatches on recording format)
# ---------------------------------------------------------------------------


def _load_mcap_demo_ros2(
    mcap_path: str,
    arm_topic: str,
    gripper_topic: str,
    base_topic: str | None = None,
    scene_joint_topic: str | None = None,
) -> McapDemo:
    """Load arm, gripper, and optional base/scene arrays from a ROS2 mcap file."""
    from mcap.reader import make_reader
    from mcap_ros2idl_support import Ros2DecodeFactory

    factory = Ros2DecodeFactory()
    arm_names: list[str] | None = None
    gripper_names: list[str] | None = None
    arm_positions: list[list[float]] = []
    gripper_positions: list[list[float]] = []
    arm_times: list[int] = []
    gripper_times: list[int] = []
    base_positions: list[np.ndarray] = []
    base_orientations: list[np.ndarray] = []
    base_times: list[int] = []
    scene_joint_names: list[str] | None = None
    scene_joint_positions: list[np.ndarray] = []
    scene_joint_times: list[int] = []
    topics = [arm_topic, gripper_topic]
    if base_topic is not None:
        topics.append(base_topic)
    if scene_joint_topic is not None:
        topics.append(scene_joint_topic)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[factory])
        for decoded in reader.iter_decoded_messages(topics=topics):
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
            elif base_topic is not None and topic == base_topic:
                pos, quat = _extract_pose_stamped_xyzw(msg, topic=base_topic)
                base_positions.append(pos)
                base_orientations.append(quat)
                base_times.append(t)
            elif scene_joint_topic is not None and topic == scene_joint_topic:
                scene_joint_names, scene_pos = _extract_joint_state_positions(
                    msg,
                    topic=scene_joint_topic,
                    expected_names=scene_joint_names,
                )
                scene_joint_positions.append(scene_pos)
                scene_joint_times.append(t)

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
        grip = _align_samples_to_times(
            grip,
            grip_t,
            arm_t,
            label=f"gripper topic '{gripper_topic}'",
        )

    base_position: np.ndarray | None = None
    base_orientation: np.ndarray | None = None
    if base_topic is not None:
        if not base_positions:
            raise ValueError(
                f"No PoseStamped messages found on base topic '{base_topic}' "
                f"in {mcap_path}"
            )
        base_t = np.array(base_times, dtype=np.int64)
        base_position = _align_samples_to_times(
            np.asarray(base_positions, dtype=np.float32),
            base_t,
            arm_t,
            label=f"base topic '{base_topic}' position",
        )
        base_orientation = _align_samples_to_times(
            np.asarray(base_orientations, dtype=np.float32),
            base_t,
            arm_t,
            label=f"base topic '{base_topic}' orientation",
        )

    scene_joint: np.ndarray | None = None
    if scene_joint_topic is not None:
        scene_joint = _align_optional_scene_joint(
            scene_joint_positions,
            scene_joint_times,
            arm_t,
            scene_joint_topic=scene_joint_topic,
            mcap_path=mcap_path,
        )

    joint = np.concatenate([arm, grip], axis=-1)
    joint_names = (arm_names or []) + (gripper_names or [])
    # Zero-anchor the MCAP times so the resulting video starts at t=0 instead
    # of the absolute ROS epoch (which would produce a hours-long initial gap
    # in the encoder's PTS).
    joint_times = arm_t - int(arm_t[0]) if arm_t.size > 0 else None
    return McapDemo(
        joint=joint,
        joint_names=joint_names,
        joint_times=joint_times,
        base_position=base_position,
        base_orientation=base_orientation,
        scene_joint=scene_joint,
        scene_joint_names=scene_joint_names,
    )


def _load_mcap_demo_foxglove(
    mcap_path: str,
    arm_topic: str | None,
    gripper_topic: str | None,
    base_topic: str | None,
    scene_joint_topic: str | None,
    channel_meta: Dict[str, tuple[str, str]],
    *,
    arm_actuator_names: list[str] | None = None,
    eef_actuator_names: list[str] | None = None,
) -> McapDemo:
    """Load arm, gripper, and optional base/scene arrays from a Foxglove
    flatbuffer mcap file (``foxglove.JointStates`` / ``foxglove.PoseInFrame``)."""
    from mcap.reader import make_reader

    arm_topic = _resolve_foxglove_joint_topic(
        channel_meta, arm_topic, _FLB_DEFAULT_ARM_TOPIC, ("/arm/",), "arm"
    )
    gripper_topic = _resolve_foxglove_joint_topic(
        channel_meta,
        gripper_topic,
        _FLB_DEFAULT_GRIPPER_TOPIC,
        ("parallel", "/g2t/", "gripper"),
        "gripper",
    )

    topics = [arm_topic, gripper_topic]
    if base_topic is not None:
        topics.append(base_topic)
    if scene_joint_topic is not None:
        topics.append(scene_joint_topic)

    arm_names: list[str] | None = None
    grip_names: list[str] | None = None
    scene_joint_names: list[str] | None = None
    arm_positions: list[np.ndarray] = []
    grip_positions: list[np.ndarray] = []
    scene_positions: list[np.ndarray] = []
    arm_times: list[int] = []
    grip_times: list[int] = []
    scene_times: list[int] = []
    base_positions: list[np.ndarray] = []
    base_orientations: list[np.ndarray] = []
    base_times: list[int] = []

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(topics=topics):
            topic = channel.topic
            t = message.log_time
            if topic == arm_topic:
                names, values = _decode_foxglove_joint_states(message.data)
                if arm_names is None:
                    arm_names = names
                arm_positions.append(values)
                arm_times.append(t)
            elif topic == gripper_topic:
                names, values = _decode_foxglove_joint_states(message.data)
                if grip_names is None:
                    grip_names = names
                grip_positions.append(values)
                grip_times.append(t)
            elif base_topic is not None and topic == base_topic:
                pos, quat = _decode_foxglove_pose_in_frame(message.data)
                base_positions.append(pos)
                base_orientations.append(quat)
                base_times.append(t)
            elif scene_joint_topic is not None and topic == scene_joint_topic:
                names, values = _decode_foxglove_joint_states(message.data)
                if scene_joint_names is None:
                    scene_joint_names = names
                scene_positions.append(values)
                scene_times.append(t)

    if not arm_positions:
        raise ValueError(f"No arm messages found on topic '{arm_topic}' in {mcap_path}")
    if not grip_positions:
        raise ValueError(
            f"No gripper messages found on topic '{gripper_topic}' in {mcap_path}"
        )

    arm = _stack_joint_positions(arm_positions, topic=arm_topic)
    grip = _stack_joint_positions(grip_positions, topic=gripper_topic)
    arm_t = np.array(arm_times, dtype=np.int64)
    grip_t = np.array(grip_times, dtype=np.int64)

    # Arm and gripper are recorded on separate topics at different rates, so
    # always align the gripper trajectory onto the arm timestamps.
    grip = _align_samples_to_times(
        grip, grip_t, arm_t, label=f"gripper topic '{gripper_topic}'"
    )

    arm_names = _canonicalize_foxglove_names(
        arm_names or [],
        arm_actuator_names,
        role="arm",
        topic=arm_topic,
        fallback_prefix="",
    )
    grip_names = _canonicalize_foxglove_names(
        grip_names or [],
        eef_actuator_names,
        role="gripper",
        topic=gripper_topic,
        fallback_prefix="gripper/",
    )

    base_position: np.ndarray | None = None
    base_orientation: np.ndarray | None = None
    if base_topic is not None:
        if not base_positions:
            raise ValueError(
                f"No PoseInFrame messages found on base topic '{base_topic}' "
                f"in {mcap_path}"
            )
        base_t = np.array(base_times, dtype=np.int64)
        base_position = _align_samples_to_times(
            np.asarray(base_positions, dtype=np.float32),
            base_t,
            arm_t,
            label=f"base topic '{base_topic}' position",
        )
        base_orientation = _align_samples_to_times(
            np.asarray(base_orientations, dtype=np.float32),
            base_t,
            arm_t,
            label=f"base topic '{base_topic}' orientation",
        )

    scene_joint: np.ndarray | None = None
    if scene_joint_topic is not None:
        scene_joint = _align_optional_scene_joint(
            scene_positions,
            scene_times,
            arm_t,
            scene_joint_topic=scene_joint_topic,
            mcap_path=mcap_path,
        )

    joint = np.concatenate([arm, grip], axis=-1)
    joint_names = list(arm_names) + list(grip_names)
    # Zero-anchor the log times (see the ROS2 loader for the rationale).
    joint_times = arm_t - int(arm_t[0]) if arm_t.size > 0 else None
    return McapDemo(
        joint=joint,
        joint_names=joint_names,
        joint_times=joint_times,
        base_position=base_position,
        base_orientation=base_orientation,
        scene_joint=scene_joint,
        scene_joint_names=scene_joint_names,
    )


def _load_mcap_demo(
    mcap_path: str,
    arm_topic: str | None,
    gripper_topic: str | None,
    base_topic: str | None = None,
    scene_joint_topic: str | None = None,
    *,
    arm_actuator_names: list[str] | None = None,
    eef_actuator_names: list[str] | None = None,
) -> McapDemo:
    """Load an mcap demo, auto-dispatching on the recording format.

    Supports the legacy ROS2 CDR format (``sensor_msgs/JointState``) and the
    newer Foxglove flatbuffer format (``foxglove.JointStates``). The format is
    detected from the file's channel schemas. ``arm_actuator_names`` /
    ``eef_actuator_names`` are only used by the flatbuffer path, where recorded
    joints are indexed locally per topic (``j0..jN``) and are canonicalised to
    the operator's actuator names by position.
    """
    channel_meta = _read_mcap_channel_meta(mcap_path)
    if _mcap_is_foxglove(channel_meta):
        return _load_mcap_demo_foxglove(
            mcap_path,
            arm_topic,
            gripper_topic,
            base_topic,
            scene_joint_topic,
            channel_meta,
            arm_actuator_names=arm_actuator_names,
            eef_actuator_names=eef_actuator_names,
        )
    return _load_mcap_demo_ros2(
        mcap_path, arm_topic, gripper_topic, base_topic, scene_joint_topic
    )
