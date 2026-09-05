# MuJoCo Initialization & Randomization

This page explains how the built-in MuJoCo backend realizes the shared
[Scene Initialization & Randomization](../task-configuration/randomization.md)
configuration contract. The field meanings, frame modes, dependency rules, and
sampling behavior are defined by the shared task-configuration page; this page
only documents MuJoCo bindings and implementation behavior.

## Names and Scene Frames

- `task.initial_pose` keys resolve through MuJoCo object handlers. Both
  free-joint bodies and fixed bodies whose pose can be changed at model level
  are supported.
- `task_operators` keys resolve through MuJoCo operator handlers and their
  configured actuator, root-body, base-frame, and EEF bindings.
- `task.camera_initial_pose` and `task.camera_randomization` keys are MuJoCo
  camera names from the composed model.
- Named pose references may resolve to a site, body, geom, or joint in the
  composed model. Names include any prefix introduced while attaching an MJCF
  layer.

Named initialization references are resolved independently for every batched
environment when setup/reset applies the override; during reset this happens
after the native model reset. They are anchors: later motion of an articulated
reference does not move an initialized object, camera, or operator base with
it. A `task.initial_pose` reference that names another configured object creates
a dependency; the backend applies those entries in topological order and
rejects cycles before mutating scene poses.

## Joint Initialization

The MuJoCo environment additionally exposes
`env.initial_joint_positions` for scene-level or basis-level joint defaults.
These are environment configuration, not part of the shared task schema.

- Values are raw MuJoCo `qpos` coordinates. Joint names must match the composed
  model, including attachment prefixes.
- Values are written after the keyframe reset and before `mj_forward`.
- Set `env.initial_joint_positions: null` in a composed task to clear an
  inherited environment mapping.
- If the model has equality constraints, reset performs a short physics settle
  so passive linkage joints can converge to a constraint-consistent state.

`task_operators.<name>.initial_state.joint_positions` uses the same raw-`qpos`
units, but is applied through the operator home seam after the low-level
environment reset. Names must belong to that operator's declared arm
actuators. If an arm joint appears in both the environment mapping and the
operator mapping, backend construction fails instead of relying on reset order.

When an operator has an [`eef_mapper`](eef_mapper.md), joint initialization
still uses raw `qpos`; it does not use the mapper's user-space finger distance.
The MCAP replay path therefore excludes mapped EEF joints from
`initial_joint_positions` and applies their recorded values through the mapper.

## Reset and Baseline Mapping

During `setup()`, `MujocoTaskBackend` homes the registered operators, applies
`task.initial_pose`, applies operator initial states (all bases first, then
joint/EEF home values), applies `task.camera_initial_pose`, and records the
effective object, operator-base, home-EEF, and camera poses as the baselines for
relative randomization.

During every `reset()` it:

1. Calls the MuJoCo environment reset, which restores keyframe/model state and
   applies `env.initial_joint_positions`.
2. Homes the registered operators.
3. Reapplies `task.initial_pose`, operator initial states, and
   `task.camera_initial_pose` in the same ownership order used during setup.
4. Samples and applies object/operator randomization, then camera
   randomization, and refreshes the viewer.

Because the native state and configured overrides are restored before every
sample, randomization offsets do not accumulate between episodes.

The backend resolves randomization targets through object handlers and then
operator handlers. Operator entries must use the nested `base` / `eef` form.
An unknown target currently emits a warning and is skipped. Invalid reference
forms, such as `.base` or `.eef` on a non-operator, raise an error.

## Operator Control Modes

For a joint-controlled operator, `initial_state.base_pose` relocates the
configured physical root body so the robot geometry and base-frame kinematics
remain aligned. For a pure mocap operator, it changes the virtual base frame
used for coordinate conversion while the registered mocap home remains the
physical pose. Use the operator's EEF/mocap-home configuration when the
physical mocap body itself must move.

Parallel-linkage grippers can require the equality-constraint settle described
above after raw joint initialization. Gripper model units and directions are
documented in [Gripper Joint Semantics](gripper_joint_semantics.md).

## Camera and GS Rendering

For ordinary MuJoCo rendering, camera initial poses and randomization directly
update the model camera pose before observations are captured.

Gaussian Splatting cameras with `env.cameras.<name>.is_static: true` cache their
background at the first render. If such a camera is randomized on later
episodes, its viewpoint changes while the cached background remains tied to the
first rendered pose. Set `is_static: false` when the background must be
re-rendered after camera randomization.

## Implementation Pointers

- `auto_atom/backend/mjc/mujoco_backend.py` owns object, operator, and camera
  initialization/randomization and the reset lifecycle.
- `auto_atom/basis/mjc/model_initialization.py` owns raw MuJoCo joint-state
  injection and equality-constraint settling.
- `auto_atom/runtime.py` exposes the realized reset poses through
  `TaskUpdate.details["initial_poses"]`.
