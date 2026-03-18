from pydantic import BaseModel
from typing import List, Any, Union, Tuple
from enum import Enum, auto
from pathlib import Path


Position = Tuple[float, float, float]
Orientation = Tuple[float, float, float, float]
Rotation = Tuple[float, float, float]


class Operation(str, Enum):
    """Enumeration of possible operations that the AutoAtom agent can perform."""

    GRASP = auto()
    """Close the gripper to grasp the object at the current position. Can only be performed when the agent is not currently grasping any object. Failure occurs when the grippers close but fail to grasp an object (no object makes effective contact with the grippers)."""
    RELEASE = auto()
    """Open the gripper to release the currently grasped object. Can only be performed when the agent is currently grasping an object."""
    PICK = auto()
    """Move to the object and grasp it. Can only be performed when the agent is not currently grasping any object. Failure occurs in the same way as the GRASP operation."""
    PLACE = auto()
    """Move to the target pose/object and release the grasped object. Can only be performed when the agent is currently grasping an object."""
    PUSH = auto()
    """Move to the object and push it to the target pose/object. Can be performed regardless of whether the agent is currently grasping an object or not."""
    PULL = auto()
    """Move to the object, grasp and then pull it to the target pose/object. Can only be performed when the agent is not currently grasping any object."""
    MOVE = auto()
    """Move to the target pose/object without interacting with any object. Can be performed regardless of whether the agent is currently grasping an object or not."""


class OperationConstraint(str, Enum):
    """Enumeration of possible constraints for the operations that the AutoAtom agent can perform."""

    GRASPED = auto()
    """The operation can only be performed when the agent is currently grasping an object."""
    RELEASED = auto()
    """The operation can only be performed when the agent is not currently grasping any object."""
    ANY = auto()
    """The operation can be performed regardless of whether the agent is currently grasping an object or not."""


class OperationConditionType(str, Enum):
    PERFORM = auto()
    """The condition for performing the operation. The agent will only perform the operation when the condition is satisfied."""
    SUCCESS = auto()
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


class StageConfig(BaseModel):
    """Configuration for each stage of the AutoAtom agent."""

    object: str
    """The name of the object to be manipulated in this stage. The object should be defined in the environment and should have a unique name. An empty name means that the corresponding operation does not involve the target object; the target pose is obtained from the corresponding param."""
    operation: Operation
    """The operation that the AutoAtom agent performs in this stage."""
    param: Any
    """The parameter for the operation."""
    operator: str = ""
    """The name of the operator that performs the operation in this stage. The operator should be defined in the environment and should have a unique name. If there is only one operator in the environment, this field can be left empty, and the agent will automatically select that operator to perform the operation."""
    blocking: bool = True
    """Whether the agent should wait for the completion of the operation before proceeding to the next stage. If set to False, the agent will proceed to the next stage immediately after initiating the operation. However, if the operator in the next stage is the same as the current stage, the agent will still wait for the completion of the operation to avoid conflicts."""


class PoseReference(str, Enum):
    """Enumeration of possible pose references for the pose control."""

    WORLD = auto()
    """The pose is defined in the world coordinate system."""
    BASE = auto()
    """The pose is defined in the robot base coordinate system."""
    END_EFFECTOR = auto()
    """The pose is defined in the end-effector coordinate system."""
    OBJECT = auto()
    """The pose is defined in the object coordinate system."""
    OBJECT_WORLD = auto()
    """The reference is equivalent to moving the origin of the world system to the origin of the object while keeping the coordinate system direction unchanged. The pose is defined in this new coordinate system."""
    AUTO = auto()
    """The pose reference is automatically determined based on the context of the operation. For example, if an object is specified in the stage configuration, the reference will be set to OBJECT_WORLD; if no object is specified, the reference will be set to BASE."""


class PoseControlConfig(BaseModel):
    """Configuration for the pose control"""

    position: Position = ()
    """The target position for the pose control. The position is represented as a tuple of three floats (x, y, z)."""
    orientation: Orientation = ()
    """The target orientation for the pose control. The orientation is represented as a tuple of four floats (x, y, z, w) representing a quaternion."""
    rotation: Rotation = ()
    """The target rotation for the pose control. The rotation is represented as a tuple of three floats (roll, pitch, yaw) representing the Euler angles."""
    reference: PoseReference = PoseReference.AUTO
    """The reference frame for the pose control."""
    relative: bool = False
    """Whether the pose control is relative to the current pose. The current pose is determined by the reference frame. """


class AutoAtomConfig(BaseModel):
    """Configuration for the AutoAtom agent."""

    stages: List[StageConfig]
    """A list of StageConfig objects, each representing a stage of the AutoAtom agent. The stages are executed in the order they are defined in the list."""
    env_path: Union[str, Path]
    """The path to the environment model file. The environment model should be defined in a format that can be loaded by the simulation environment used by the AutoAtom agent."""
    simulator: str = "mujoco"
    """The name of the simulator to be used by the AutoAtom agent. The simulator should be compatible with the environment model."""
    seed: int = 0
    """The random seed for the AutoAtom agent. This is used to ensure reproducibility of the agent's behavior."""
    episodes: int = 1
    """The number of episodes to run the AutoAtom agent. An episode is a complete executionn of the stages defined in the configuration. The agent will reset the environment at the beginning of each episode."""
