# Physical Replay Presentation

Physical replay executes the original task controller and every physics step,
while independently controlling which states are synchronized to the MuJoCo
viewer. Use it when a task prefix contains contact-driven state such as a
grasped object, drawer, door, or lever and therefore cannot be reconstructed
safely with kinematic teleportation.

## Visible-window model

Combine `task.physical_replay` and `task.stop_at` to select a visible segment of
a complete task:

| Task range | Physics/controller | Viewer |
| --- | --- | --- |
| Before `physical_replay` target | Fully executed | Hidden |
| `physical_replay` target | Already physically reached | Always shown as the start |
| Between start and end | Fully executed | Controlled by `presentation` |
| `stop_at` target | Fully executed | Always shown, then returns success |
| After `stop_at` target | Not executed | Not shown |

The start and end targets have priority over the presentation policy. In the
default `waypoint` mode, “skip” means skipping viewer refreshes between
waypoints. It does **not** skip controller updates, MuJoCo substeps, contacts,
callbacks, or object dynamics.

## Minimal configuration

Create a child config instead of replacing the complete task definition:

```yaml
defaults:
  - place_blocks_on_disk_airbot_play_g2
  - _self_

task:
  # Visible start. Everything before this target is physically replayed but
  # not displayed. This final pick approach makes gripper close the first
  # visible action.
  physical_replay:
    stage: pick_cube_yellow_2
    phase: pre_move
    waypoint: 1
    presentation:
      mode: waypoint
      preserve_arcs: false
      keyframe_hold_seconds: 0.05

  # Visible end. The endpoint is displayed and the task immediately succeeds;
  # the remaining task suffix is not executed.
  stop_at:
    stage: place_cube_orange_3_in_disk
    phase: post_move
    waypoint: 0
```

This runnable configuration is provided as
[`aao_configs/place_blocks_on_disk_airbot_play_g2_segment.yaml`](../../aao_configs/place_blocks_on_disk_airbot_play_g2_segment.yaml).
Run it with:

```bash
aao-demo --config-name place_blocks_on_disk_airbot_play_g2_segment
```

Its behavior is:

1. Reset and physically execute the task through
   `pick_cube_yellow_2.pre_move[1]` without displaying that prefix.
2. Display the selected start state.
3. Continue toward `place_cube_orange_3_in_disk.post_move[0]`, displaying only
   reached waypoint and gripper endpoints.
4. Display the selected end state and return `done=true`, `success=true`.

## Selecting start and end coordinates

A waypoint selector contains:

- `stage`: a unique `task.stages[].name`;
- `phase`: `pre_move` or `post_move`;
- `waypoint`: a zero-based index in that phase's YAML list;
- `frame_offset` (optional): additional controller frames after the selected
  waypoint reports `REACHED`.

For example, this starts 20 controller frames after a pick approach waypoint:

```yaml
task:
  physical_replay:
    stage: pick_source
    phase: pre_move
    waypoint: 1
    frame_offset: 20
```

Only pose waypoints in `pre_move` and `post_move` can be addressed directly.
Gripper close/open actions are still presentation keyframes. To make a gripper
close the first visible action, choose the last `pre_move` waypoint of that
pick stage as the start.

Absolute controller frames are also supported:

```yaml
task:
  physical_replay:
    frame: 300
  stop_at:
    frame: 450
```

Frame `0` is the freshly reset randomized state. Frame `N` is the state after
exactly `N` controller updates; it is not a raw MuJoCo substep or video frame.

## Presentation options

### Default: waypoint-to-waypoint jumps

```yaml
presentation:
  mode: waypoint
  preserve_arcs: false
  keyframe_hold_seconds: 0.05
```

- The selected start and end are always shown.
- Ordinary linear motion is hidden between waypoint endpoints.
- With `preserve_arcs: false` (the default), curved/arc motion is hidden too.
- Gripper open/close completion is shown as an endpoint.
- `keyframe_hold_seconds` controls only wall-clock viewer hold time and never
  advances simulation time.

Set `keyframe_hold_seconds: 0` for the fastest possible visual jump sequence.

### Preserve door or drawer arc animation

```yaml
presentation:
  mode: waypoint
  preserve_arcs: true
  keyframe_hold_seconds: 0.05
```

Linear actions still jump between endpoints, but actions configured with
`arc:` are synchronized every controller tick. This is useful for an opening
door animation. Set it back to `false` when an arc should also be skipped.

See [Arc Motion Tuning](../ik-motion-control/arc_motion_tuning.md) for arc
trajectory configuration itself.

### Display every visible controller tick

```yaml
presentation:
  mode: full
```

`full` displays all controller ticks between the selected start and end.
The physical prefix before the start remains hidden, and execution still ends
at `stop_at`.

### Hide the segment interior

```yaml
presentation:
  mode: hidden
```

With `stop_at` configured, only the selected start and end are shown. All
interior motion executes physically without viewer synchronization.

## Fast-forward all the way to the endpoint

Set `physical_replay` and `stop_at` to the same coordinate when no visible
interior is needed:

```yaml
task:
  physical_replay:
    stage: place_cube_orange_3_in_disk
    phase: post_move
    waypoint: 0
    presentation:
      mode: waypoint

  stop_at:
    stage: place_cube_orange_3_in_disk
    phase: post_move
    waypoint: 0
```

The complete prefix is physically executed during `reset()`, only the endpoint
is presented, and `reset()` returns with `done=true` and `success=true`. Use two
different coordinates when intermediate waypoint-to-waypoint states should be
visible.

## Command-line overrides

For the provided segment config, arc presentation and keyframe hold can be
changed without editing YAML:

```bash
aao-demo --config-name place_blocks_on_disk_airbot_play_g2_segment \
  task.physical_replay.presentation.preserve_arcs=true

aao-demo --config-name place_blocks_on_disk_airbot_play_g2_segment \
  task.physical_replay.presentation.keyframe_hold_seconds=0
```

## Constraints and troubleshooting

- `task.start_after` and `task.physical_replay` are mutually exclusive.
  `start_after` teleports; `physical_replay` executes the real controller and
  physics.
- A waypoint-based `stop_at` must be the same as or later than the physical
  replay start. A target already passed by the start is rejected.
- If no intermediate states appear, check whether `physical_replay` and
  `stop_at` select the same coordinate.
- If an arc does not animate, set `preserve_arcs: true`. If it should jump,
  leave the default `false`.
- Viewer skipping reduces display/sleep overhead, but it does not remove the
  physics work. Complex contact prefixes still take computation time.
- Presentation settings have no visible effect when the viewer is disabled.
- `gaussian_render.share_physics: true` is not supported by reset-time physical
  replay because virtual batch entries share one physical world.

For selector validation, frame semantics, and the difference from teleport
fast-forward, see [Stages & Waypoints](stages_and_waypoints.md#reset-time-task-prefixes).
