# Tune Randomization Extremes

Interactive tkinter inspector for verifying that configured randomization ranges keep objects and operators within a reasonable workspace. Opens the MuJoCo viewer alongside a control panel that lets you cycle through extreme randomization cases.

The inspector also reloads the YAML values that define the **default** state:
`task.initial_pose` and `task_operators.<name>.initial_state`. This makes it
useful for iterating on both the nominal operator/object placement and the
randomization ranges around that placement.

All structured pose overrides use the shared `PoseOverrideConfig` contract.
In particular, operator `base_pose` and structured `eef_pose` entries may use
a named scene frame (or `base` for an EEF pose); the six-value EEF shorthand
remains available as a complete world-frame pose. The inspector applies these
through the backend before rebuilding its nominal targets. Named references
are resolved at reload/reset time and are held as fixed world anchors while a
task runs.

**Script:** [examples/tune_randomization_extremes.py](../examples/tune_randomization_extremes.py)

## Usage

```bash
python examples/tune_randomization_extremes.py
python examples/tune_randomization_extremes.py --config-name cup_on_coaster
python examples/tune_randomization_extremes.py --config-name arrange_flowers
```

The default config is `pick_and_place`. The script extracts the `task.randomization` section from the YAML config and builds a set of extreme cases.

### Multiple disjoint regions

An object (or an operator `base` / `eef` entry) may define mutually exclusive
candidate regions with `regions`:

```yaml
task:
  randomization:
    source_block:
      regions:
        - reference: absolute_world
          x: [0.20, 0.30]
          y: [-0.10, 0.00]
        - reference: absolute_world
          x: [0.60, 0.70]
          y: [0.10, 0.20]
    arm:
      eef:
        regions:
          - x: [-0.02, 0.02]
          - x: [0.20, 0.24]
```

The inspector keeps one physical target per entry. Every generated case selects
at most one region for that target; regions are never applied simultaneously.
Each region gets its own all-min/all-max and per-axis min/max cases, and the
`Random Sample` button selects one region uniformly before sampling that
region's axes. The current-pose panel reports the selected zero-based region
index. A region with only `reference` / `collision_radius` and no configured
axis still gets region-qualified all-min/all-max entries so it can be selected
and inspected explicitly.

## Control panel

The tkinter panel provides:

- **Randomization summary** -- shows every region's reference, axis ranges, and non-default collision radius
- **Extreme case selector** -- dropdown to pick a case, with Prev/Next buttons
- **Apply / Reset Default** -- apply the selected case or return to the nominal pose
- **Random Sample** -- draw a fresh random sample uniformly from each configured range
- **Reload Randomization** -- re-read the YAML config from disk, apply updated defaults from `task.initial_pose` / `task_operators.<name>.initial_state`, then rebuild the randomization cases
- **Full Reload** -- rebuild the entire scene and backend from the current config
- **Current Poses** -- live display of each target's position, quaternion, and RPY

## Generated extreme cases

The inspector automatically generates these cases from the config:

| Case | Description |
|---|---|
| `default` | No randomization offset; all targets at nominal pose |
| `all-min` | Every randomized axis at its minimum value simultaneously (the first region is used for multi-region targets) |
| `all-max` | Every randomized axis at its maximum value simultaneously (the first region is used for multi-region targets) |
| `<target> <axis>=min` | Single target, single axis at minimum; everything else at default |
| `<target> <axis>=max` | Single target, single axis at maximum; everything else at default |

For multi-region targets, region-qualified cases are named
`<target> [region N] ...` and include region-level all-min/all-max cases in
addition to the per-axis cases. This gives every configured region an
independent inspection path without turning disjoint regions into simultaneous
targets.

These cases cover each axis endpoint plus the synchronized all-min/all-max
states. They do not exhaust mixed corners such as `x=min, y=max`, every
multi-target combination, or every cross-region combination. Use Random Sample
and task-specific checks for those combinations. If a generated case pushes an
object outside the workspace, off the table, or into collision, the range
should be tightened.

`default` is the current YAML-defined baseline after initial pose/state
overrides have been applied. Operators with `initial_state` are shown even if
they do not have a randomization range, so you can edit the YAML directly and
press **Reload Randomization** to inspect the updated nominal base, EEF pose,
and gripper control.

## Workflow

1. Configure `task.randomization` in your YAML config (see [Pose Randomization](../task-configuration/randomization.md))
2. Run this inspector to visually verify the extremes
3. Use `all-min` and `all-max` to check the two same-direction simultaneous endpoint states
4. Step through per-axis cases to identify which specific axis causes problems
5. Use `Random Sample` to spot-check typical randomized states
6. Edit your YAML `initial_pose`, operator `initial_state`, or randomization ranges and press `Reload Randomization` to iterate without restarting

Use **Full Reload** when you add new object/operator names that were not
registered when the scene was first opened.

## Joint-limit proximity warnings

The inspector force-enables `UnifiedMujocoEnv.set_joint_limit_warning_enabled(True)`
on startup, even though the env's default is off. Surfacing borderline IK
solutions is the entire point of this tool, so the env will log a `WARNING`
whenever an IK-solved joint angle lands within ~0.05 rad (≈ 2.9°) of a hard
joint limit. The log line includes the operator, joint name, current angle
(rad + deg), distance to limit, and the limit value.

Use these warnings as a signal that:

- The randomization range is letting the arm reach poses that just barely
  satisfy IK and would behave poorly in production — tighten the per-axis
  range until the warnings stop, or
- The default `initial_state.base_pose` / `initial_state.eef_pose` itself is
  already close to a limit and any randomization on top will push past it,
  so the home pose should be re-tuned.

Each (joint, side) only warns once per entry into the danger band; the warning
re-arms after the joint moves at least 0.10 rad back from the limit, so the
log will not flap when a solution sits right at the boundary.

See [IK Control § 关节限位接近告警](../ik-motion-control/ik_control.md) for the
full warning-system contract.
