# Auto Atomic Operation

This context defines the language used to describe motion goals, execution
boundaries, and reusable scene assets in an automatic atomic-operation task.

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

**Controlled frame**:
The end-effector or held-object frame whose pose a waypoint semantically constrains.
_Avoid_: Reference frame, target frame

**Reference frame**:
The coordinate frame in which a waypoint's target position, orientation, or axis is
expressed; it does not identify which frame is being moved.
_Avoid_: Controlled frame, controlled object

**Pose override**:
A setup-time declaration that replaces selected position and/or orientation
components of an entity's initial pose in a reference frame.  The same concept
applies to scene objects, cameras, and an operator's base or home EEF pose;
omitted components retain the resolved fallback pose.
_Avoid_: Arm-only pose config, ad-hoc placement offset

**Operator home state**:
The operator state restored at setup and reset: its base pose, home EEF pose,
and gripper control value.  A home state is distinct from a Stage's transient
motion target and is resolved before pose randomization baselines are recorded.
_Avoid_: EEF initial pose, waypoint pose

**Orientation goal**:
A full or partial rotational requirement on a controlled frame; an axis-only goal
leaves rotation about the constrained axis free.
_Avoid_: EEF quaternion, rotation mask

**Grasp binding**:
The measured rigid relationship between an end effector and its held object after a
verified grasp.
_Avoid_: Configured grasp offset, target pose

**Stage**:
One configured task operation containing an ordered sequence of keypoints and its
operation conditions.
_Avoid_: Phase

**Pick operation**:
An atomic-operation contract that succeeds only when the stage target is grasped
after its effect sequence.
_Avoid_: Generic grasp

**Pull operation**:
An atomic-operation contract that requires the stage target to be grasped at the
effect boundary and retain that same target through completion; the term does
not specify motion direction.
_Avoid_: Directional pull

**Push operation**:
An atomic-operation contract that judges success by target displacement without
a grasp-retention requirement; the term does not specify motion direction.
_Avoid_: Directional push

**Stage execution**:
The progression of one stage from its perform condition, through its primitive
sequence, to a succeeded or failed terminal status.
_Avoid_: Stage lifecycle, rollout

**Execution timeline**:
The ordered view of a task's stages, keypoints, and primitives, including the
before/after boundaries around each keypoint.
_Avoid_: Rollout schedule, execution plan

**Scene assembly**:
The declarative composition of one host scene and its ordered scene layers,
including pre-authored MJCF modules and asset-package assemblies.
_Avoid_: Vendor-specific scene loader, robot-only composition

**Scene assembler**:
The implementation adapter that resolves one scene-layer recipe into a
self-contained contribution and semantic exports.  It does not mutate the host
scene or own temporary-file lifecycle.
_Avoid_: Scene service, asset loader

**Scene asset package**:
A relocatable, versioned package descriptor plus component manifests, artifact
references, frames, anchors, mechanism data, integrity records, and provenance.
Legacy vendor manifests may be retained as provenance but are not the runtime
contract.
_Avoid_: Combination XML catalog, vendor product-space file

**Collision supplement**:
A versioned scene-asset record that supplies a component's physical collision
representation independently of its visual artifact. Its frame and identity
are bound to the component manifest.
_Avoid_: Collision metadata, visual mesh
