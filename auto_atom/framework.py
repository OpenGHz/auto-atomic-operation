from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, ImportString, Field, field_validator


Position = Tuple[float, float, float]
"""A 3D position represented as a tuple of three floats (x, y, z)."""
Orientation = Tuple[float, float, float, float]
"""A quaternion orientation represented as a tuple of four floats (x, y, z, w)."""
Rotation = Tuple[float, float, float]
"""A rotation represented as Euler angles in radians, as a tuple of three floats (roll, pitch, yaw)."""


class Operation(str, Enum):
    """Enumeration of possible operations that the AutoAtom operator can perform.
    `MOVE`, `GRASP`, `RELEASE` are three fundamental operations that can be used to construct more complex operations like `PICK`, `PLACE`, `PUSH`, `PULL`, and `PRESS`."""

    MOVE = "move"
    """Execute pre_move waypoints to reach the target pose without interacting with any object. No pre-condition is checked; post-condition `reached` is checked after the final pose action. Failure occurs when the operator fails to reach the target pose within the position tolerance within the time limit."""
    GRASP = "grasp"
    """Execute the eef phase (close gripper) at the current position. Pre-condition `released` is checked before the eef phase; post-condition `grasped` is checked after the eef phase. Failure occurs when the post-condition `grasped` is not satisfied (the gripper closes but no object is effectively grasped)."""
    RELEASE = "release"
    """Execute the eef phase (open gripper) at the current position. Pre-condition `grasped` is checked before the eef phase; post-condition `released` is checked after the eef phase. Failure occurs when the post-condition `released` is not satisfied (the gripper opens but the object is still effectively grasped)."""
    PICK = "pick"
    """Execute pre_move → eef (close gripper) → post_move to approach an object and grasp it. Pre-condition `released` is checked before the pre_move phase; post-condition `grasped` is checked after the post_move phase. Failure occurs when the post-condition `grasped` is not satisfied."""
    PLACE = "place"
    """Execute pre_move → eef (open gripper) → post_move to approach a target pose and release the held object. Pre-condition `grasped` is checked before the pre_move phase; post-condition `released` is checked after the post_move phase. Failure occurs when the post-condition `released` is not satisfied."""
    PUSH = "push"
    """Execute pre_move → post_move to approach and push an object to a target pose. No pre-condition is checked; post-condition `displaced` is checked after the post_move phase. Failure occurs when the post-condition `displaced` is not satisfied (the object has not moved beyond the displacement threshold)."""
    PULL = "pull"
    """Execute pre_move → eef (close gripper) → post_move to approach an object, grasp it, and pull it to a target pose. Pre-condition `grasped` is checked after the eef phase (confirming a successful grasp before pulling); post-condition `grasped` is checked after the post_move phase. Failure occurs when either condition `grasped` is not satisfied."""
    PRESS = "press"
    """Execute pre_move → eef → post_move to approach and press an object at the target pose. No pre-condition is checked; post-condition `contacted` is checked after the eef phase (at the moment of contact, before retreat). Failure occurs when the post-condition `contacted` is not satisfied (the operator end-effector is not in contact with the target object after the eef phase)."""


class OperationConstraint(str, Enum):
    """Enumeration of possible constraints for the operations."""

    GRASPED = "grasped"
    """Whether the operator is currently grasping an object."""
    RELEASED = "released"
    """Whether the operator is not currently grasping any object."""
    CONTACTED = "contacted"
    """Whether the operator is in contact with the target object."""
    DISPLACED = "displaced"
    """Whether the target object has been displaced from its original pose (e.g., the distance between the current pose of the object and its original pose is greater than a certain threshold) after the operation."""
    REACHED = "reached"
    """Whether the operator end-effector is within tolerance of the final target pose for the stage."""
    PLACED = "placed"
    """Whether the operator has released the held object AND the held object
    is within tolerance of the target position/orientation."""
    NONE = "none"
    """No constraint."""


class OperationConditionType(str, Enum):
    PERFORM = "perform"
    """The condition for performing the operation. The operator will only perform the operation when the condition is satisfied."""
    SUCCESS = "success"
    """The condition for the success of the operation. The operation is considered successful when the condition is satisfied after performing the operation."""


_Condition = OperationConditionType
OPERATION_CONDITIONS = {
    Operation.MOVE: {
        _Condition.SUCCESS: OperationConstraint.REACHED,
    },
    Operation.GRASP: {
        _Condition.PERFORM: OperationConstraint.RELEASED,
        _Condition.SUCCESS: OperationConstraint.GRASPED,
    },
    Operation.RELEASE: {
        _Condition.PERFORM: OperationConstraint.GRASPED,
        _Condition.SUCCESS: OperationConstraint.RELEASED,
    },
    Operation.PICK: {
        _Condition.PERFORM: OperationConstraint.RELEASED,
        _Condition.SUCCESS: OperationConstraint.GRASPED,
    },
    Operation.PLACE: {
        _Condition.PERFORM: OperationConstraint.GRASPED,
        _Condition.SUCCESS: OperationConstraint.PLACED,
    },
    Operation.PUSH: {
        _Condition.SUCCESS: OperationConstraint.DISPLACED,
    },
    Operation.PULL: {
        _Condition.PERFORM: OperationConstraint.GRASPED,
        _Condition.SUCCESS: OperationConstraint.GRASPED,
    },
    Operation.PRESS: {
        _Condition.SUCCESS: OperationConstraint.CONTACTED,
    },
}


class RandomizationReference(str, Enum):
    """Reference mode for a :class:`PoseRandomRange`.

    Controls how the per-axis ``[min, max]`` ranges are interpreted when
    sampling a randomized pose.
    """

    RELATIVE = "relative"
    """Ranges are additive offsets from the entity's default/initial pose
    (current default behavior)."""
    ABSOLUTE_WORLD = "absolute_world"
    """Ranges are absolute world-frame values — metres for position axes,
    radians for Euler orientation axes. The entity's default pose is ignored
    for any axis that has an explicit range."""
    ABSOLUTE_BASE = "absolute_base"
    """Ranges are absolute values expressed in the operator's base frame.
    The sampled pose is transformed back into world frame before being
    applied. Only valid for operator end-effector randomization."""


class PoseReference(str, Enum):
    """Enumeration of possible pose references for the pose control."""

    WORLD = "world"
    """The pose is defined in the world coordinate system."""
    BASE = "base"
    """The pose is defined in the robot base coordinate system."""
    EEF = "eef"
    """The pose is defined in the current operator eef coordinate system."""
    OBJECT = "object"
    """The pose is defined in the object coordinate system."""
    OBJECT_WORLD = "object_world"
    """The reference is equivalent to moving the origin of the world system to the origin of the object while keeping the coordinate system direction unchanged. The pose is defined in this new coordinate system. The target pose will track the movement of the object after action start, meaning that the target pose will change accordingly as the object moves."""
    EEF_WORLD = "eef_world"
    """The reference is equivalent to moving the origin of the world system to the operator's end-effector position at the moment the action starts, while keeping the coordinate system direction unchanged. The target pose is snapshotted once at action start and does not track subsequent EEF movement."""
    AUTO = "auto"
    """The pose reference is automatically determined based on the context of the operation. For example, if an object is specified in the stage configuration, the reference will be set to OBJECT_WORLD; if no object is specified, the reference will be set to BASE."""


class ArcControlConfig(BaseModel, extra="forbid"):
    """Configuration for arc (revolute) movement around a pivot axis.

    When attached to a ``PoseControlConfig``, the end-effector traces an arc
    around ``pivot`` instead of moving in a straight line.  The ``position``,
    ``orientation``, and ``rotation`` fields of the parent config are ignored."""

    pivot: Union[Position, str]
    """Pivot point for the arc.  Either explicit ``(x, y, z)`` coordinates in the
    coordinate frame given by the parent's ``reference``, or a **string name** of a
    site, body, or joint in the scene XML whose world position is used automatically."""
    axis: Position
    """Unit-direction of the rotation axis (x, y, z)."""
    angle: float
    """Rotation angle in radians.  Positive follows the right-hand rule around ``axis``.
    When ``absolute`` is False (default), this is a relative rotation from the current
    EEF position.  When ``absolute`` is True and ``pivot`` is a joint name, this is
    the target joint angle and the runtime computes the relative rotation automatically."""
    absolute: bool = False
    """When True, ``angle`` is treated as an absolute target joint angle (radians)
    instead of a relative rotation.  Requires ``pivot`` to be a joint name so the
    runtime can read the current joint angle and compute the delta."""
    max_step: float = 0.2
    """Maximum arc sub-step in radians (~11.5 deg).  Smaller values produce smoother
    arcs at the cost of more waypoints."""


class WaypointToleranceConfig(BaseModel, extra="forbid"):
    """Per-waypoint tolerance override. When set on a waypoint, these values
    take precedence over the operator-level tolerance for that waypoint only.

    Position tolerance can be a single float (L2 norm) or a list of three
    floats ``[x, y, z]`` for per-axis tolerance checking."""

    position: Optional[Union[float, List[float]]] = None
    """Position tolerance. A scalar applies as an L2-norm threshold;
    a 3-element list ``[x, y, z]`` checks each axis independently."""
    orientation: Optional[float] = None
    """Orientation tolerance in radians (quaternion angular distance)."""


class PlacedToleranceConfig(BaseModel, extra="forbid"):
    """Tolerance for the PLACED post-condition. Each dimension can be null
    to skip checking that dimension."""

    position: Optional[Union[float, List[Optional[float]]]] = [None, None, None]
    """Position tolerance. Scalar = L2-norm threshold. List ``[x, y, z]`` =
    per-axis thresholds where ``null`` means no constraint on that axis."""

    orientation: Optional[Union[float, List[Optional[float]]]] = [None, None, None]
    """Orientation tolerance in radians. Scalar = quaternion angular distance
    threshold. List ``[roll, pitch, yaw]`` = per-axis Euler thresholds where
    ``null`` means no constraint on that axis."""


class PoseRandomRange(BaseModel):
    """Per-entity pose randomization bounds.

    The ``reference`` field selects one of three modes, **or** names
    another entity to track:

    - ``"relative"`` (default): each per-axis ``[min, max]`` range is an
      additive offset applied to the entity's default/initial pose.
    - ``"absolute_world"``: ranges are absolute world-frame values —
      metres for position, radians for Euler orientation. The default
      pose is ignored for any axis that has an explicit range.
    - ``"absolute_base"``: ranges are absolute values expressed in the
      operator's base frame. The sampled pose is then transformed into
      world frame before being applied. Only valid for operator
      end-effector randomization.
    - **Entity name** (e.g. ``"vase1"``): the referenced entity is
      randomized first; its displacement from its default pose is
      computed (``delta = sampled * default⁻¹``) and applied to this
      entity's default pose so they move together. Then the per-axis
      ranges are applied as additive offsets on top, just like
      ``relative`` mode.

    A ``None`` value on an axis (the default) means "do not randomize
    this axis" — it keeps its value from the default pose (in the
    relevant frame) in all modes. Axes are independent, so absolute-mode
    ``x``/``y`` with ``z``/``roll``/``pitch``/``yaw`` left as ``None``
    produces the natural "place anywhere on this rectangle, keep default
    height and orientation" behavior.

    Example YAML entries::

        # Relative (default): sampled as default_pose + offset
        randomization:
          source_block:
            x: [-0.03, 0.03]
            y: [-0.03, 0.03]
            collision_radius: 0.04

        # Absolute world-frame: sampled as world-frame coordinates
        randomization:
          vase1:
            reference: absolute_world
            x: [0.10, 0.45]
            y: [-0.15, 0.15]

        # Entity reference: carry flower with vase1, then jitter ±5mm
        randomization:
          vase1:
            reference: absolute_world
            x: [0.22, 0.58]
            y: [-0.32, 0.27]
          flower:
            reference: vase1
            x: [-0.005, 0.005]
            y: [-0.005, 0.005]
    """

    model_config = ConfigDict(extra="forbid")

    x: Optional[Tuple[float, float]] = None
    """[min, max] range along the world X axis (metres), or ``None`` to
    leave this axis at the default-pose value."""
    y: Optional[Tuple[float, float]] = None
    """[min, max] range along the world Y axis (metres), or ``None`` to
    leave this axis at the default-pose value."""
    z: Optional[Tuple[float, float]] = None
    """[min, max] range along the world Z axis (metres), or ``None`` to
    leave this axis at the default-pose value."""
    roll: Optional[Tuple[float, float]] = None
    """[min, max] range for the roll Euler angle (radians), or ``None`` to
    leave this axis at the default-pose value."""
    pitch: Optional[Tuple[float, float]] = None
    """[min, max] range for the pitch Euler angle (radians), or ``None`` to
    leave this axis at the default-pose value."""
    yaw: Optional[Tuple[float, float]] = None
    """[min, max] range for the yaw Euler angle (radians), or ``None`` to
    leave this axis at the default-pose value."""
    reference: Union[RandomizationReference, str] = RandomizationReference.RELATIVE
    """One of the :class:`RandomizationReference` modes (``"relative"``,
    ``"absolute_world"``, ``"absolute_base"``) or the **name of another
    entity**. An entity name causes this entry to track the referenced
    entity's displacement (delta-carry) and then apply the per-axis
    ranges as relative offsets on top."""
    collision_radius: float = 0.05
    """Approximate bounding radius used for pairwise collision rejection (metres)."""

    @field_validator("reference", mode="before")
    @classmethod
    def _coerce_reference(cls, v: object) -> object:
        if isinstance(v, str) and not isinstance(v, RandomizationReference):
            try:
                return RandomizationReference(v)
            except ValueError:
                return v  # entity name — validated at sample time
        return v


class PoseControlConfig(BaseModel):
    """Configuration for the pose control"""

    model_config = ConfigDict(extra="forbid")

    position: Position = Field(default_factory=tuple)
    """The target position for the pose control. The position is represented as a tuple of three floats (x, y, z)."""
    orientation: Orientation = Field(default_factory=tuple)
    """The target orientation for the pose control. The orientation is represented as a quaternion in `xyzw` order."""
    rotation: Rotation = Field(default_factory=tuple)
    """The target rotation for the pose control. The rotation is represented as Euler angles in `rpy` order."""
    reference: PoseReference = PoseReference.AUTO
    """The reference frame for the pose control."""
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


class EefControlConfig(BaseModel, extra="forbid"):
    """Configuration for the end-effector control"""

    close: bool
    """Whether to close the end-effector. True for closing the end-effector, False for opening the end-effector. This will set the end-effector joint positions to the lower limit or upper limit defined in the environment model."""
    joint_positions: List[float] = []
    """The target joint positions for the end-effector control. The order and meaning of the joint positions depend on the specific end-effector used in the environment."""


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
    ``'pre_move'`` is always used regardless of this setting."""
    placed_tolerance: Optional[PlacedToleranceConfig] = PlacedToleranceConfig()
    """Per-stage tolerance override for the PLACED post-condition. Falls back
    to the operator-level placed tolerance, then to ``position=0.02,
    orientation=null``."""


class StageConfig(BaseModel):
    """Configuration for each stage of the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    """The optional human-readable name of this stage."""
    object: str
    """The name of the object to be manipulated in this stage. The object should be defined in the environment and should have a unique name. An empty name means that the corresponding operation does not involve the target object; the target pose is obtained from the corresponding param."""
    operation: Operation
    """The operation that the AutoAtom operator performs in this stage."""
    param: StageControlConfig
    """The parameter for the operation."""
    operator: str = ""
    """The name of the operator that performs the operation in this stage. The operator should be defined in the environment and should have a unique name. If there is only one operator in the environment, this field can be left empty, and the operator will automatically select that operator to perform the operation."""
    blocking: bool = True
    """Whether the operator should wait for the completion of the operation before proceeding to the next stage. If set to False, the operator will proceed to the next stage immediately after initiating the operation. However, if the operator in the next stage is the same as the current stage, the operator will still wait for the completion of the operation to avoid conflicts."""


class OperatorRandomizationConfig(BaseModel):
    """Randomization options for an operator.

    ``base`` controls the operator base pose returned by ``get_base_pose()``.
    For mocap operators this is the virtual base frame; for joint-mode
    operators this is the robot base reference frame.

    ``eef`` controls the operator home end-effector pose in world frame. After
    sampling, reset re-homes the operator to the sampled EEF pose.
    """

    model_config = ConfigDict(extra="forbid")

    base: Optional[PoseRandomRange] = None
    eef: Optional[PoseRandomRange] = None


class AutoAtomConfig(BaseModel):
    """Configuration for the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    stages: List[StageConfig]
    """A list of StageConfig objects, each representing a stage of the AutoAtom operator. The stages are executed in the order they are defined in the list."""
    env_name: str
    """The registered environment name used to resolve the basis environment instance for the selected scene."""
    seed: int = 0
    """The random seed for the AutoAtom operator. This is used to ensure reproducibility of the operator's behavior."""
    randomization: Dict[str, Union[PoseRandomRange, OperatorRandomizationConfig]] = (
        Field(default_factory=dict)
    )
    """Per-entity pose randomization applied at each reset.

    Objects accept a direct ``PoseRandomRange``.

    Operators accept either:
    - a direct ``PoseRandomRange`` (backward-compatible shorthand for
      base/virtual-base randomization), or
    - ``OperatorRandomizationConfig`` with independent ``base`` and ``eef``
      randomization ranges.
    """
    randomization_debug: bool = False
    """When True the first N resets cycle through extreme poses (each axis at its min/max, then all-min and all-max) before switching to random sampling.  Use this to verify that configured ranges are not too large."""


class ArmPoseConfig(BaseModel):
    """Structured arm pose configuration with separate position and orientation."""

    position: Optional[List[float]] = None
    """3D position [x, y, z]. When omitted, keyframe position is kept."""

    orientation: Optional[List[float]] = None
    """Orientation as Euler angles [yaw, pitch, roll] (3 floats) or quaternion [x, y, z, w] (4 floats).
    When omitted, keyframe orientation is kept."""

    reference: PoseReference = PoseReference.WORLD
    """Reference frame for interpreting position/orientation.
    WORLD (default): pose is in world frame — backward compatible with mocap mode.
    BASE: pose is relative to the operator's base frame — useful for arm setups."""


class OperatorInitialState(BaseModel):
    """Optional override for an operator's home control state applied at reset."""

    arm: Optional[Union[List[float], ArmPoseConfig]] = None
    """Override values for the arm actuator controls.

    Supports two formats:
    1. Flat list: [x, y, z, yaw, pitch, roll] (backward compatible)
    2. Structured dict: {position: [x,y,z], orientation: [yaw,pitch,roll] or [x,y,z,w]}
       - Both position and orientation are optional in structured format
       - orientation can be Euler angles (3 floats) or quaternion (4 floats)

    When omitted the keyframe value is kept."""

    eef: Optional[float] = None
    """Override value for the end-effector/gripper control (0.0 = open, 0.82 = closed).
    When omitted the keyframe value is kept."""

    base_pose: Optional[ArmPoseConfig] = None
    """Override for the operator's base world pose.
    For a real arm: the arm base body's world pose (fixed mounting position).
    For mocap: a virtual reference origin in world (defaults to keyframe mocap position).
    When omitted, the base pose is read from the simulation state at init."""


class OperatorConfig(BaseModel):
    """Configuration for constructing an operator instance from YAML."""

    model_config = ConfigDict(extra="allow")

    name: str
    """The unique operator name referenced by task stages."""

    initial_state: Optional[OperatorInitialState] = None
    """Optional initial control state applied to this operator on every reset.
    Overrides the keyframe-defined values for the specified fields."""


class TaskFileConfig(BaseModel):
    """Top-level YAML schema for a runnable task file."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    backend: ImportString
    """The backend to execute this task file. The backend should be registered in the ComponentRegistry and should be compatible with the selected scene."""
    task: AutoAtomConfig
    """The task-level configuration describing stages, scene, and environment selection."""
    task_operators: List[OperatorConfig] = []
    """The operator definitions available to the selected backend for this task file.
    Accepts both ``task_operators`` and the legacy alias ``operators`` from YAML."""
