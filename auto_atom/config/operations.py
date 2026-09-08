"""operations configuration models (split from the former auto_atom.framework monolith)."""

from enum import Enum


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
    """Execute pre_move → eef (close gripper) → post_move to approach the Stage target and grasp it. Pre-condition `released` is checked before the pre_move phase; the target-specific post-condition `grasped` is checked after the post_move phase. Failure occurs when the Stage target is not grasped."""
    PLACE = "place"
    """Execute pre_move → eef (open gripper) → post_move to approach a target pose and release the held object. Pre-condition `grasped` is checked before the pre_move phase; post-condition `placed` is checked after the post_move phase. Failure occurs when the held object is still grasped or, when a placement target is available, outside placement tolerance."""
    PUSH = "push"
    """Execute pre_move → post_move to approach and push an object to a target pose. No pre-condition is checked; post-condition `displaced` is checked after the post_move phase. Failure occurs when the post-condition `displaced` is not satisfied (the object has not moved beyond the displacement threshold)."""
    PULL = "pull"
    """Execute pre_move → eef (close gripper) → post_move to approach the Stage target, grasp it, and apply an effect trajectory. Target-specific `grasped` conditions are checked after the eef phase and after the post_move phase. Failure occurs when the Stage target is not grasped at either boundary."""
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
    """Whether the final waypoint's controlled frame is within its pose tolerance."""
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
