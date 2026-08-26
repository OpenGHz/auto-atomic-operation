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
