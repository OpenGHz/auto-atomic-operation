# Auto Atomic Operation

This context defines the execution boundaries used to describe progress through an
automatic atomic-operation task.

## Language

**Control tick**:
One low-level controller update and its corresponding backend simulation step.
_Avoid_: Step, frame

**Primitive**:
One runtime action issued to an operator, such as a pose command, an arc sub-action,
or an end-effector command.
_Avoid_: Keypoint, waypoint

**Keypoint**:
One configured task point identified by stage, phase, and YAML waypoint index; one
keypoint may expand into multiple primitives.
_Avoid_: Primitive, control point

**Stage**:
One configured task operation containing an ordered sequence of keypoints and its
operation conditions.
_Avoid_: Phase
