"""motion configuration models (split from the former auto_atom.framework monolith)."""

from typing import List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from auto_atom.config.operations import Operation
from auto_atom.config.orientation import (
    AxisAlignmentOrientationGoalConfig,
    OrientationGoalConfig,
)
from auto_atom.config.primitives import Orientation, Position, Rotation
from auto_atom.config.randomization import PoseRandomRange
from auto_atom.config.reference import (
    ControlledFrameConfig,
    ControlledFrameKind,
    PoseReference,
)


class ArcControlConfig(BaseModel):
    """Configuration for arc (revolute) movement around a pivot axis.

    When attached to a ``PoseControlConfig``, the end-effector traces an arc
    around ``pivot`` instead of moving in a straight line.  The ``position``,
    ``orientation``, and ``rotation`` fields of the parent config are ignored."""

    model_config = ConfigDict(extra="forbid")

    pivot: Union[Position, str]
    """Pivot point for the arc.  Either explicit ``(x, y, z)`` coordinates in the
    coordinate frame given by the parent's ``reference``, or a **string name** of a
    site, body, or joint in the scene XML whose world position is used automatically."""
    axis: Position
    """Unit-direction of the rotation axis (x, y, z)."""
    angle: Optional[float] = None
    """Rotation angle in radians. Positive follows the right-hand rule around
    ``axis``. When ``absolute`` is False (default), this is a relative rotation
    from the current EEF position. When ``absolute`` is True and ``pivot`` is a
    joint name, this is the target joint angle and the runtime computes the
    relative rotation automatically. Mutually exclusive with ``arc_length``."""
    arc_length: Optional[float] = None
    """Signed target arc length in metres. Exactly one of ``angle`` and
    ``arc_length`` must be configured. The runtime converts this length to an
    angle using the measured pivot-to-EEF radius for each environment."""
    absolute: bool = False
    """When True, ``angle`` is treated as an absolute target joint angle (radians)
    instead of a relative rotation. Requires ``pivot`` to be a joint name so the
    runtime can read the current joint angle and compute the delta."""
    max_step: float = 0.2
    """Maximum arc sub-step in radians (~11.5 deg).  Smaller values produce smoother
    arcs at the cost of more waypoints."""
    joint_tolerance: PositiveFloat = 0.01
    """Joint-angle tolerance in radians for completing an absolute arc.  Reaching
    one local end-effector target is not sufficient until the named pivot joint is
    also within this tolerance of ``angle``.  Relative arcs ignore this field."""
    timeout_steps: PositiveInt = 1000
    """Maximum aggregate control updates for one absolute arc.  This task-level
    limit remains effective when successive local end-effector targets reset a
    backend controller's per-pose timeout.  Relative arcs ignore this field."""
    reverse: bool = False
    """When True, the arc is traced in the opposite direction around the axis.

    Implemented as ``axis → -axis`` (rather than ``angle → -angle``) so the
    behaviour is correct in both relative and absolute modes:

    - Relative: negating the axis is mathematically equivalent to negating
      the angle, so the rotation direction flips as expected.
    - Absolute: ``angle`` is the target joint value, not a rotation amount.
      Negating ``angle`` would change the goal (e.g. +0.45 → -0.45) and the
      runtime would chase an unreachable target. Flipping the axis preserves
      the goal while reversing the world-frame rotation direction."""

    @model_validator(mode="after")
    def validate_reverse(self):
        """If reverse is True, negate the axis to reverse the rotation direction."""
        if self.reverse:
            self.axis = tuple(-v for v in self.axis)
        return self

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        """Require one unambiguous arc target and keep absolute arcs joint-based."""
        if (self.angle is None) == (self.arc_length is None):
            raise ValueError("arc requires exactly one of angle or arc_length")
        if self.absolute and self.arc_length is not None:
            raise ValueError("arc_length does not support absolute=true")
        if self.absolute and self.angle is None:
            raise ValueError("absolute arc requires an angle target")
        return self


class WaypointToleranceConfig(BaseModel):
    """Per-waypoint tolerance override. When set on a waypoint, these values
    take precedence over the operator-level tolerance for that waypoint only.

    Position tolerance can be a single float (L2 norm) or a list of three
    floats ``[x, y, z]`` for per-axis tolerance checking."""

    model_config = ConfigDict(extra="forbid")

    position: Optional[Union[float, List[float]]] = None
    """Position tolerance. A scalar applies as an L2-norm threshold;
    a 3-element list ``[x, y, z]`` checks each axis independently."""
    orientation: Optional[float] = None
    """Orientation tolerance in radians.

    This is quaternion angular distance for a complete orientation and axis
    angular error for an axis-alignment goal.
    """


class PlacedToleranceConfig(BaseModel):
    """Tolerance for the PLACED post-condition. Each dimension can be null
    to skip checking that dimension."""

    model_config = ConfigDict(extra="forbid")

    position: Optional[Union[float, List[Optional[float]]]] = [None, None, None]
    """Position tolerance. Scalar = L2-norm threshold. List ``[x, y, z]`` =
    per-axis thresholds where ``null`` means no constraint on that axis."""

    orientation: Optional[Union[float, List[Optional[float]]]] = [None, None, None]
    """Orientation tolerance in radians. Scalar = quaternion angular distance
    threshold for complete orientations, or axis angular error for an
    axis-alignment goal. List ``[roll, pitch, yaw]`` = per-axis Euler
    thresholds where ``null`` means no constraint on that axis; lists are not
    valid for axis-alignment goals."""


class PoseControlConfig(BaseModel):
    """Configuration for the pose control"""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    position: Optional[Position] = None
    """Target controlled-frame position as three floats ``(x, y, z)``."""
    orientation: Optional[Orientation] = None
    """Legacy full controlled-frame orientation as an ``xyzw`` quaternion."""
    rotation: Optional[Rotation] = None
    """Legacy full controlled-frame rotation as Euler angles in ``rpy`` order."""
    controlled_frame: ControlledFrameConfig = ControlledFrameConfig()
    """The frame whose pose this waypoint controls; defaults to the end effector."""
    orientation_goal: Optional[OrientationGoalConfig] = None
    """Optional full or partial orientation goal for the controlled frame."""
    reference: PoseReference = PoseReference.AUTO
    """The reference frame for the pose control."""
    static: bool = False
    """Whether the reference frame should be snapshotted at action start.

    By default, ``OBJECT`` / ``OBJECT_WORLD`` references are re-evaluated
    on every control tick, so the target tracks the object as it moves.
    That is the correct behavior when the object moves independently of
    the gripper. However, when the gripper is *rigidly gripping* the
    object, a tracking target is unreachable — the reference frame moves
    with the gripper, so the residual never closes.

    Set ``static: true`` to freeze the reference pose at the first tick
    of this waypoint, giving a fixed world-frame target. ``EEF`` /
    ``EEF_WORLD`` are always snapshotted and ignore this flag."""
    relative: bool = False
    """Whether the pose control is relative to the current pose. The current pose is determined by the reference frame. """
    use_slerp: bool = False
    """Whether to use SLERP interpolation for smooth orientation transitions."""
    max_linear_step: float = 0.0
    """Maximum Cartesian translation step (metres) applied per control tick.
    When > 0, the runtime moves toward the target position incrementally instead
    of commanding the full translation at once."""
    max_angular_step: float = 0.0
    """Maximum orientation step (radians) applied per control tick.
    When > 0, the runtime SLERPs toward the target orientation incrementally
    instead of commanding the full rotation at once."""
    arc: Optional[ArcControlConfig] = None
    """Optional arc movement configuration. When set, the end-effector traces an arc
    around the specified pivot instead of moving in a straight line to the target position."""
    tolerance: Optional[WaypointToleranceConfig] = None
    """Optional per-waypoint tolerance override. When set, these values take
    precedence over the operator-level tolerance for this waypoint only."""
    randomization: Optional[PoseRandomRange] = None
    """Optional per-waypoint pose randomization. When set, a random offset is
    sampled from these ranges and added to the waypoint position/orientation
    at the start of each episode."""

    @model_validator(mode="after")
    def validate_orientation_goal(self) -> Self:
        """Reject ambiguous or unsupported orientation-goal combinations."""
        if self.orientation_goal is not None and self.arc is not None:
            raise ValueError("orientation_goal does not support arc movement")
        if (
            self.controlled_frame.kind == ControlledFrameKind.HELD_OBJECT
            and self.arc is not None
        ):
            raise ValueError(
                "held_object controlled_frame does not support arc movement"
            )
        if self.orientation_goal is None:
            return self
        if self.orientation is not None or self.rotation is not None:
            raise ValueError(
                "orientation_goal cannot be combined with orientation or rotation"
            )
        if self.randomization is not None and any(
            getattr(self.randomization, axis) is not None
            for axis in ("roll", "pitch", "yaw")
        ):
            raise ValueError(
                "orientation_goal cannot be combined with rotational randomization"
            )
        if isinstance(
            self.orientation_goal,
            AxisAlignmentOrientationGoalConfig,
        ):
            if self.relative:
                raise ValueError(
                    "axis_alignment orientation_goal does not support relative=true"
                )
        return self


class EefControlConfig(BaseModel):
    """Configuration for the end-effector control"""

    model_config = ConfigDict(extra="forbid")

    close: bool
    """Whether to close the end-effector. True for closing the end-effector, False for opening the end-effector. This will set the end-effector joint positions to the lower limit or upper limit defined in the environment model."""
    joint_positions: List[float] = []
    """The target joint positions for the end-effector control. The order and meaning of the joint positions depend on the specific end-effector used in the environment."""
    require_grasp: bool = False
    """When closing on a Stage target, require the backend to verify that target is
    physically grasped before reporting the end-effector primitive as reached."""

    @model_validator(mode="after")
    def validate_require_grasp(self):
        """A grasp completion requirement is only meaningful for closing."""
        if self.require_grasp and not self.close:
            raise ValueError("require_grasp=true requires close=true")
        return self


class StageControlConfig(BaseModel):
    """Configuration for the control of each stage of the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    pre_move: List[PoseControlConfig] = Field(default_factory=list)
    """Optional pose controls to execute before the main stage action."""
    post_move: List[PoseControlConfig] = Field(default_factory=list)
    """Optional pose controls to execute after the main stage action."""
    eef: Optional[EefControlConfig] = None
    """The configuration for the end-effector control in this stage. If not specified, no end-effector control will be performed in this stage."""
    placed_reference: str = "object"
    """Target reference for the PLACED post-condition. ``'object'`` uses the
    stage object's current pose (the destination); ``'pre_move'`` uses the
    last pre_move waypoint resolved position. When the stage has no object,
    ``'pre_move'`` is always used regardless of this setting. A resolved
    held-object pre-move goal is authoritative regardless of this legacy
    selector."""
    placed_tolerance: Optional[PlacedToleranceConfig] = PlacedToleranceConfig()
    """Per-stage tolerance override for the PLACED post-condition. Falls back
    to the operator-level placed tolerance. If neither level configures a
    non-null position or orientation tolerance, placement degrades to
    released-only."""
    displacement_threshold: Optional[float] = None
    """Per-stage threshold (meters) for the DISPLACED post-condition. When set,
    overrides the backend default of 0.01 m used by ``is_object_displaced``.
    Only meaningful for operations whose success constraint is DISPLACED
    (e.g., ``push``)."""


class StageConfig(BaseModel):
    """Configuration for each stage of the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    """The optional human-readable name of this stage."""
    object: str
    """The name of the object to be manipulated in this stage. The object should be defined in the environment and should have a unique name. An empty name means that the corresponding operation does not involve the target object; the target pose is obtained from the corresponding param."""
    site: Optional[str] = None
    """Optional site/body/geom/joint name used as the reference frame for
    ``reference: object_world`` / ``reference: object`` waypoints in this
    stage. When set, its world pose replaces ``object``'s pose as the
    reference origin (and, for ``reference: object``, also as the
    reference orientation). When ``None``, the ``object`` body's pose is
    used as before. This field only affects pose reference resolution —
    ``object`` is still used for contact detection, GS rendering mask,
    ``set_pose``/randomization, and arc pivot fallback."""
    operation: Operation
    """The operation that the AutoAtom operator performs in this stage."""
    param: StageControlConfig
    """The parameter for the operation."""
    operator: str = ""
    """The name of the operator that performs the operation in this stage. The operator should be defined in the environment and should have a unique name. If there is only one operator in the environment, this field can be left empty, and the operator will automatically select that operator to perform the operation."""
    blocking: bool = True
    """Whether the operator should wait for the completion of the operation before proceeding to the next stage. If set to False, the operator will proceed to the next stage immediately after initiating the operation. However, if the operator in the next stage is the same as the current stage, the operator will still wait for the completion of the operation to avoid conflicts."""
