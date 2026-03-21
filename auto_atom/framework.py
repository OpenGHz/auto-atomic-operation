from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, ImportString, Field


Position = Tuple[float, float, float]
Orientation = Tuple[float, float, float, float]
Rotation = Tuple[float, float, float]


class Operation(str, Enum):
    """Enumeration of possible operations that the AutoAtom agent can perform."""

    GRASP = "grasp"
    """Close the gripper to grasp the object at the current position. Can only be performed when the agent is not currently grasping any object. Failure occurs when the grippers close but fail to grasp an object (no object makes effective contact with the grippers)."""
    RELEASE = "release"
    """Open the gripper to release the currently grasped object. Can only be performed when the agent is currently grasping an object."""
    PICK = "pick"
    """Move to the object and grasp it. Can only be performed when the agent is not currently grasping any object. Failure occurs in the same way as the GRASP operation."""
    PLACE = "place"
    """Move to the target pose/object and release the grasped object. Can only be performed when the agent is currently grasping an object."""
    PUSH = "push"
    """Move to the object and push it to the target pose/object. Can be performed regardless of whether the agent is currently grasping an object or not."""
    PULL = "pull"
    """Move to the object, grasp and then pull it to the target pose/object. Can only be performed when the agent is not currently grasping any object."""
    MOVE = "move"
    """Move to the target pose/object without interacting with any object. Can be performed regardless of whether the agent is currently grasping an object or not."""


class OperationConstraint(str, Enum):
    """Enumeration of possible constraints for the operations that the AutoAtom agent can perform."""

    GRASPED = "grasped"
    """The operation can only be performed when the agent is currently grasping an object."""
    RELEASED = "released"
    """The operation can only be performed when the agent is not currently grasping any object."""
    ANY = "any"
    """The operation can be performed regardless of whether the agent is currently grasping an object or not."""


class OperationConditionType(str, Enum):
    PERFORM = "perform"
    """The condition for performing the operation. The agent will only perform the operation when the condition is satisfied."""
    SUCCESS = "success"
    """The condition for the success of the operation. The operation is considered successful when the condition is satisfied after performing the operation."""


_Condition = OperationConditionType
OPERATION_CONDITIONS = {
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
        _Condition.SUCCESS: OperationConstraint.RELEASED,
    },
}


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


class EefControlConfig(BaseModel, extra="forbid"):
    """Configuration for the end-effector control"""

    close: bool
    """Whether to close the end-effector. True for closing the end-effector, False for opening the end-effector. This will set the end-effector joint positions to the lower limit or upper limit defined in the environment model."""
    joint_positions: List[float] = []
    """The target joint positions for the end-effector control. The order and meaning of the joint positions depend on the specific end-effector used in the environment."""


class StageControlConfig(BaseModel):
    """Configuration for the control of each stage of the AutoAtom agent."""

    model_config = ConfigDict(extra="forbid")

    pre_move: List[PoseControlConfig] = Field(default_factory=list)
    """Optional pose controls to execute before the main stage action."""
    post_move: List[PoseControlConfig] = Field(default_factory=list)
    """Optional pose controls to execute after the main stage action."""
    eef: Optional[EefControlConfig] = None
    """The configuration for the end-effector control in this stage. If not specified, no end-effector control will be performed in this stage."""


class StageConfig(BaseModel):
    """Configuration for each stage of the AutoAtom agent."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    """The optional human-readable name of this stage."""
    object: str
    """The name of the object to be manipulated in this stage. The object should be defined in the environment and should have a unique name. An empty name means that the corresponding operation does not involve the target object; the target pose is obtained from the corresponding param."""
    operation: Operation
    """The operation that the AutoAtom agent performs in this stage."""
    param: StageControlConfig
    """The parameter for the operation."""
    operator: str = ""
    """The name of the operator that performs the operation in this stage. The operator should be defined in the environment and should have a unique name. If there is only one operator in the environment, this field can be left empty, and the agent will automatically select that operator to perform the operation."""
    blocking: bool = True
    """Whether the agent should wait for the completion of the operation before proceeding to the next stage. If set to False, the agent will proceed to the next stage immediately after initiating the operation. However, if the operator in the next stage is the same as the current stage, the agent will still wait for the completion of the operation to avoid conflicts."""


class PoseRandomRange(BaseModel):
    """Per-entity pose randomization bounds relative to its default pose.

    Each translation axis specifies a ``[min_offset, max_offset]`` range in
    world-frame metres.  Each rotation axis specifies a ``[min_offset,
    max_offset]`` range in radians applied as an additive RPY increment.

    Example YAML entry::

        randomization:
          source_block:
            x: [-0.03, 0.03]
            y: [-0.03, 0.03]
            yaw: [-0.524, 0.524]
            collision_radius: 0.04
    """

    model_config = ConfigDict(extra="forbid")

    x: Tuple[float, float] = (0.0, 0.0)
    """[min, max] displacement along the world X axis (metres)."""
    y: Tuple[float, float] = (0.0, 0.0)
    """[min, max] displacement along the world Y axis (metres)."""
    z: Tuple[float, float] = (0.0, 0.0)
    """[min, max] displacement along the world Z axis (metres)."""
    roll: Tuple[float, float] = (0.0, 0.0)
    """[min, max] roll offset added to the default roll (radians)."""
    pitch: Tuple[float, float] = (0.0, 0.0)
    """[min, max] pitch offset added to the default pitch (radians)."""
    yaw: Tuple[float, float] = (0.0, 0.0)
    """[min, max] yaw offset added to the default yaw (radians)."""
    collision_radius: float = 0.05
    """Approximate bounding radius used for pairwise collision rejection (metres)."""


class AutoAtomConfig(BaseModel):
    """Configuration for the AutoAtom agent."""

    model_config = ConfigDict(extra="forbid")

    stages: List[StageConfig]
    """A list of StageConfig objects, each representing a stage of the AutoAtom agent. The stages are executed in the order they are defined in the list."""
    env_name: str
    """The registered environment name used to resolve the basis environment instance for the selected simulator."""
    seed: int = 0
    """The random seed for the AutoAtom agent. This is used to ensure reproducibility of the agent's behavior."""
    randomization: Dict[str, PoseRandomRange] = Field(default_factory=dict)
    """Per-entity pose randomization applied at each reset.  Keys are object or operator names; values define the randomization range."""
    randomization_debug: bool = False
    """When True the first N resets cycle through extreme poses (each axis at its min/max, then all-min and all-max) before switching to random sampling.  Use this to verify that configured ranges are not too large."""


class OperatorConfig(BaseModel):
    """Configuration for constructing an operator instance from YAML."""

    model_config = ConfigDict(extra="allow")

    name: str
    """The unique operator name referenced by task stages."""


class TaskFileConfig(BaseModel):
    """Top-level YAML schema for a runnable task file."""

    model_config = ConfigDict(extra="allow")

    backend: ImportString
    """The backend to execute this task file. The backend should be registered in the ComponentRegistry and should be compatible with the selected simulator."""
    task: AutoAtomConfig
    """The task-level configuration describing stages, simulator, and environment selection."""
    operators: List[OperatorConfig] = Field(default_factory=list)
    """The operator definitions available to the selected backend for this task file."""
