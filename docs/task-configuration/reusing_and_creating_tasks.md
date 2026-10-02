# Reusing & Creating Tasks — Agent Guide

This guide tells an agent (or a human) how to satisfy a request like *"I need a
task that does X"* **efficiently**: reuse what exists, tweak it the cheapest way
that works, and only add a config file when something is fundamentally new —
and then put it in the config group that owns that kind of difference.

> **Golden rule.** A new file under `aao_configs/` is justified **only** by a
> *fundamental* difference, and the kind of difference decides where it goes:
> a new **operation flow** (or object set) is a new `task/<name>.yaml`; a new
> **robot** is a new `embodiment/<name>.yaml`; an existing task that needs
> **tuning for a robot** gets `adapt/<task>/<embodiment>.yaml`; **GS assets**
> go under `render_assets/`. If none of those change, do **not** add a file:
> adjust the existing task in place, or pass a command-line override.

## How a run is composed

Every run composes one primary config, `aao_configs/config.yaml`, from config
groups. Nothing is selected by file name any more — a run is a choice of group
options:

```bash
aao-demo task=<task> [embodiment=<robot>] [render=gs] [backend=warp] [platform=egl] ...
```

The groups compose in this order; later entries win, and command-line
overrides win over everything:

| Order | Group | Directory | Owns |
|---|---|---|---|
| 1 | `simulator` | `simulator/` | Environment shell (`mujoco`, or `mock` for simulator-free tasks) |
| 2 | `execution` | `execution/` | `physical` or `object_only` execution |
| 3 | `observation` | `observation/` | Sensor streams and camera defaults (`observation.camera.width`, `enable_depth`, ...) |
| 4 | `camera_layout` | `camera_layout/` | Which camera roles are kept (`all`, `operator_only`, `no_operator`) |
| 5 | `scene` | `scene/` | `scene_name` (the MJCF under `assets/xmls/scenes/<scene_name>/`), static cameras, scene asset layers; `model_name: demo_gs` when native rendering shares the GS-aligned layout (`arrange_flowers`, `cup_on_coaster`) |
| 6 | `embodiment` | `embodiment/` | Robot MJCF layer, operator binding, IK, home pose, wrist camera, `eef_top_down_orientation` |
| 7 | `task` | `task/` | Stages, objects, operations, randomization, viewer framing |
| 8 | `render` | `render/` | `mujoco` (native) or `gs` (Gaussian splatting) |
| 9 | `render_assets/*` | `render_assets/{embodiment,scene,task}/<render>/` | Per-renderer asset bindings, auto-selected (optional) |
| 10 | `adapt` | `adapt/<task>/<embodiment>.yaml` | Task × embodiment tuning, auto-selected (optional) |
| 11 | `backend` | `backend/` | `cpu` or `warp` |
| 12 | `platform` | `platform/` | `egl` headless-rendering environment variables (unset by default) |

Rules that follow from this layout:

- A task selects its scene and default robot inside its own `defaults` list
  with `override /scene: <scene>` and `override /embodiment: <robot>`;
  `embodiment=...` on the command line still takes priority.
- `render`, `backend`, and `platform` compose after the task, so they are
  user-level choices a task cannot override.
- `adapt` and `render_assets/*` are **specialization slots**: Hydra picks
  `adapt/${task}/${embodiment}.yaml` and
  `render_assets/<axis>/${render}/<name>.yaml` from the final group choices and
  silently skips them when no file matches. Never list them in a task.
- `env.cameras` and `env.scene.layers` are dicts keyed by slot name
  (`wrist`, `env0`, `robot`, `door`, ...). A later config replaces one slot by
  restating its key, or removes it with `null`;
  `prepare_task_config_for_instantiation` flattens them into the ordered
  lists the environment expects. Every camera inherits the
  `observation.camera` defaults.
- `env.initial_joint_positions.<joint>: null` drops one inherited joint; other
  lists (for example `task.stages`) are replaced wholesale, so parameterize a
  task with named values when only numbers differ per robot.

## Decision flow

```
User asks for a task
        │
        ▼
1. Does a matching variant already exist?  ──► `aao-info` (search / filter / --vocab)
        │ yes                                       │ a close one exists
        ▼                                           ▼
   Run it as-is: copy its `run:` line       2. What differs?
   aao-demo task=<t> [embodiment=<e>]          │
                                               ├─ a value (waypoint, range, camera, seed)
                                               │      → 3. CLI override, or edit in place
                                               ├─ renderer / backend / platform / GS background
                                               │      → render=gs, backend=warp, platform=egl,
                                               │        render_assets/background=<name> (no file)
                                               ├─ an existing task on an existing robot
                                               │      → embodiment=<e>; if it needs tuning,
                                               │        4c. adapt/<task>/<e>.yaml
                                               ├─ a new robot / gripper
                                               │      → 4b. embodiment/<robot>.yaml
                                               ├─ new objects / scene, or a new operation flow
                                               │      → 4a. task/<task>.yaml (+ scene/<scene>.yaml)
                                               └─ GS assets for a robot, scene, or task
                                                      → 4d. render_assets/...
                                                                │
                                                                ▼
                                                5. Verify with `aao-info` + a run
```

## Step 1 — Discover what already exists

`aao-info` is the reuse-discovery tool. Never hand-scan `aao_configs/`; query
it. It lists every task once for its default embodiment and once more for
every embodiment with an `adapt/<task>/<embodiment>.yaml`, and prints the
`aao-demo` command for each variant:

```bash
aao-info                     # every runnable task x embodiment variant
aao-info -o press            # variants that use a given operation
aao-info --object cup        # variants that manipulate a given object
aao-info -r p7               # variants for a given embodiment / robot model
aao-info 'cup_on_coaster*'   # glob over task names
aao-info --vocab             # keyword glossary (tasks, embodiments, scenes, objects, ...)
```

See the [CLI Reference](../getting-started/cli_reference.md#aao-info) for every
flag. Start from the closest existing task — most requests are a small delta on
one that already ships.

## Step 2 — Classify the difference

| The request changes… | Fundamental? | What to do |
|---|---|---|
| A waypoint height / grasp offset / approach pose | No | Edit in place (task, or its `adapt` file for one robot), or CLI override |
| Randomization range, tolerance, seed, rounds, batch size | No | Edit in place, or CLI override |
| Camera / viewer framing, camera resolution or modalities | No | Edit in place, or CLI override (`observation.camera.*`, `camera_layout=...`) |
| Renderer, GS background, backend, platform | No | `render=gs`, `render_assets/background=<name>`, `backend=warp`, `platform=egl` |
| An existing task on another **existing** robot | No new task | `embodiment=<name>`; add `adapt/<task>/<embodiment>.yaml` only if it needs tuning |
| A **new robot / gripper** | **Yes** | New `embodiment/<name>.yaml` |
| The **object set or the scene** | **Yes** | New `task/<name>.yaml` selecting a (new) `scene/<scene>.yaml` + scene XML / assets |
| The **operation flow** (which operations, in what order) | **Yes** | New `task/<name>.yaml` with new `task.stages` |
| GS rendering for a robot / scene / task that has none | **Yes** (assets) | New `render_assets/{embodiment,scene,task}/gs/<name>.yaml` |

When in doubt, ask: *"Would this change break the task for its current users?"*
If yes it is fundamental; if it is just a better/alternate value for the same
task, it is not (edit or override).

## Step 3 — Non-fundamental changes (the common case)

### 3a. CLI override — for one-off, experimental, or swept values

Hydra lets you override any key on the command line, so you can explore without
touching a file. Dotted paths index into lists too. Keys the composed config
does not define need a `+` prefix (`+rounds`, `+max_updates`):

```bash
# tweak a named task parameter and a stage waypoint, run 3 rounds, 4 envs, fixed seed
aao-demo task=pick_and_place \
    pick_place.pick_z=0.008 \
    task.stages.1.param.pre_move.0.position="[0.0, 0.0, 0.14]" \
    +rounds=3 env.batch_size=4 task.seed=1

# user-level dimensions are group selections, not files
aao-demo task=cup_on_coaster render=gs render_assets/background=table
aao-demo task=pick_and_place backend=warp observation.camera.width=320
```

Use this for quick experiments and parameter sweeps — anything you would not
want to commit as the task's new default.

### 3b. Edit in place — for a permanent change to an existing task

If the new value *should become* the task's behaviour, edit
`task/<task>.yaml` directly (its `task.stages`, `task.randomization`,
`env.viewer`, …). If the value is only right for one robot, edit that robot's
`adapt/<task>/<embodiment>.yaml` instead so the other embodiments keep theirs.
**Do not add a file to change one number** — a near-duplicate silently drifts
from the original. See [Stages & Waypoints](stages_and_waypoints.md) and
[Randomization](randomization.md) for the fields.

## Step 4 — Fundamental changes: add a file to the right group

Task, scene, embodiment, and adapt files start with `# @package _global_` (their
keys land at the config root) and end their `defaults` list with `_self_` so
their local values override what they include. `_self_` is a precedence marker, not a ceremonial
suffix: a shared fragment may deliberately put it earlier when a later include
is intended to win.

### 4a. Add a task — `task/<name>.yaml`

A task is an operation flow on a scene. It selects the scene and a default
robot, and stays robot-agnostic where it can: use
`${eef_top_down_orientation}` (defined by the mocap-gripper, P7, and Franka
embodiments — define it in any new embodiment) and named parameters for values
a robot may need to retune.

```yaml
# @package _global_
# aao_configs/task/cup_on_coaster.yaml
defaults:
  - override /scene: cup_on_coaster       # scene/cup_on_coaster.yaml
  - override /embodiment: robotiq_mocap   # default robot; `embodiment=...` still wins
  - _self_

env:
  mask_objects: ["cup", "coaster"]
  operations: ["pick", "place"]

task:
  randomization:
    entities:
      cup: {x: [0.2, 0.5], y: [-0.32, 0.25], collision_radius: 0.04, reference: absolute_world}
  stages:
    - name: pick_cup
      object: cup
      operation: pick
      operator: arm
      param:
        pre_move:
          - position: [0.0, 0.0, 0.12]
            orientation: ${eef_top_down_orientation}
            reference: object_world
        ...
```

If the task needs new objects, add a scene: `scene/<scene>.yaml` sets
`scene_name` (the robot-less host MJCF is
`assets/xmls/scenes/<scene_name>/demo.xml`, or `demo_gs.xml` with
`model_name: demo_gs`) and its static cameras as slots:

```yaml
# @package _global_
scene_name: cup_on_coaster
model_name: demo_gs   # native and GS rendering share the GS-aligned layout

env:
  cameras:
    env1: {name: env1_cam, is_static: true}
    env0: {name: env0_cam, is_static: true}
```

Several tasks can share one scene (`open_drawer` / `close_drawer` both use
`open_close_drawer`). Tasks that differ only by a parameter share a fragment:
`task/press_blue_button.yaml` is just `defaults: [_press_button, _self_]`
plus `object_name: button_blue`.

### 4b. Add a robot — `embodiment/<name>.yaml`

An embodiment is task-agnostic: the robot MJCF layer, operator binding, home
pose, wrist camera name, IK solver/backend, and the top-down grasp orientation.
Build it on the closest shared fragment (`_mocap_gripper`, `_fixed_arm`,
`_p7`, `_airbot_play`) or on a sibling option:

```yaml
# @package _global_
# aao_configs/embodiment/xf9600_mocap.yaml
defaults:
  - _mocap_gripper
  - _self_

eef_top_down_orientation: [0.70710678, 0.0, -0.70710678, 0.0]

env:
  sim_freq: 1200
  scene:
    layers:
      robot:
        kind: mjcf
        path: ${assets_dir}/xmls/robots/xf9600_mocap.xml
        role: operator
  cameras:
    wrist:
      name: eef_wrist_cam
  initial_joint_positions:
    xf9600_freejoint: [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
    eef_clawj: 0.0
  operators:
    arm:
      eef_actuators: [eef_claw_joint]
      root_body: xf9600_interface
      mocap_body: xf9600_mocap
      freejoint: xf9600_freejoint
```

A variant of an existing robot can include it and restate only what differs
(`embodiment/p7_v4_umi_v3.yaml` includes `p7_v3_umi_v3` and replaces
`env.scene.layers.robot.path` and the IK kinematics). The new robot then runs
any task with `embodiment=<name>`.

### 4c. Tune a task for a robot — `adapt/<task>/<embodiment>.yaml`

When `task=<t> embodiment=<e>` composes but the robot needs different heights,
base pose, cameras, or control settings, add `adapt/<t>/<e>.yaml`. It is
picked up automatically, composes after the task (so it wins over it), and
makes the combination appear in `aao-info` as a validated variant:

```yaml
# @package _global_
# aao_configs/adapt/pick_and_place/xf9600_mocap.yaml
pick_place:
  pick_z: 0.045
  place_approach_z: 0.15
  place_z: 0.10
  retreat_z: 0.18
```

Prefer retuning the task's named parameters over restating `task.stages`;
restate the stages only when the robot forces a different program (as
`adapt/pick_and_place/franka_robotiq.yaml` does for per-waypoint IK step
limits). Remove inherited entries that do not apply with `null`, e.g.
`env.initial_joint_positions.robotiq_freejoint: null` or
`env.cameras.env0: null`. When several task × embodiment pairs share one
adaptation, put it in a fragment and include it by absolute path:

```yaml
# @package _global_
# aao_configs/adapt/open_door/p7_xf9600.yaml
defaults:
  - /adapt/_open_door/p7
  - _self_

p7_open_door:
  door_angle: 0.35
```

### 4d. GS assets — `render_assets/...`

`render=gs` (`render/gs.yaml`) switches the environment to the GS renderer and
loads the default background. Everything asset-specific is auto-selected from
`render_assets/` for the chosen task, scene, and embodiment — a task never
lists GS files:

| File | Package | Contents |
|---|---|---|
| `render_assets/embodiment/gs/<embodiment>.yaml` | `env.gaussian_render` | The robot's per-body PLYs (`body_gaussians`, `body_transforms`) |
| `render_assets/scene/gs/<scene>.yaml` | `_global_` | Object PLYs under `env.gaussian_render.body_gaussians`, GS viewer framing, scene-specific `model_name` / backgrounds |
| `render_assets/task/gs/<task>.yaml` | `env.gaussian_render` | Task-specific GS corrections (e.g. `body_mirrors` for `open_door_back`) |
| `render_assets/background/<name>.yaml` | `env.gaussian_render` | `background_ply` + `background_transforms`; select with `render_assets/background=<name>` |

```yaml
# @package _global_
# aao_configs/render_assets/scene/gs/cup_on_coaster.yaml
env:
  gaussian_render:
    body_gaussians:
      cup_gs: ${gs_dir}/cup.ply
      coaster_gs: ${gs_dir}/coaster.ply
```

`aao-info` lists `render: mujoco | gs` for a variant once its scene has a
`render_assets/scene/gs/<scene>.yaml`.

## Step 5 — Naming and layout conventions

- Option names are `snake_case`. A **task** names the operation flow on a
  scene (`cup_on_coaster`, `open_door_back`) and carries **no robot or
  renderer suffix** — those are `embodiment=` and `render=`.
- An **embodiment** names arm and gripper, or the gripper and `_mocap` for a
  free-floating gripper: `p7_g2p`, `p7_v3_umi_v3`, `airbot_play_g2p`,
  `franka_robotiq`, `robotiq_mocap`, `xf9600_mocap`. **Match existing
  spellings** rather than inventing new ones.
- A **scene** option is named like its XML directory under
  `assets/xmls/scenes/`.
- `adapt/<task>/<embodiment>.yaml` and
  `render_assets/<axis>/<render>/<name>.yaml` must match option names
  exactly: the slots are optional, so a misspelt file is silently ignored.
- **`_`-prefixed files and directories are shared fragments**
  (`embodiment/_p7.yaml`, `task/_press_button.yaml`, `adapt/_open_door/`,
  `render_assets/embodiment/gs/_airbot_play.yaml`). They are included through
  `defaults`, never selected directly, and `aao-info` skips them.
- Scratch / experimental overrides go in `aao_configs/test/` and are appended
  with `+test=<name>` (for example `+test=open_the_door`).

## Step 6 — Verify

```bash
aao-info <task>                               # the task / new variant is listed with the
                                              # expected embodiment, objects, and workflow
aao-demo task=<task> [embodiment=<e>] --info defaults   # which scene / adapt / render_assets files composed
aao-demo task=<task> [embodiment=<e>] --cfg job         # print the composed config
aao-demo task=<task> [embodiment=<e>]                   # actually run it
```

If `aao-info` does not list your variant, it composed without a non-empty
`task.stages` (so it is not a task), the composition errored, or — for a
non-default embodiment — there is no `adapt/<task>/<embodiment>.yaml`. Run
`aao-info --verbose` to see skip reasons. A task × embodiment pair without an
adapt file still runs with `embodiment=<e>`; it is just not advertised as
validated.

## Anti-patterns

- **Adding a file to change one waypoint / range / seed.** → Edit in place, or
  use a CLI override.
- **Naming a task after a robot or renderer** (`<task>_<robot>`,
  `<task>_gs`). → Use `embodiment=` / `render=`; tune with `adapt/`, add GS
  assets under `render_assets/`.
- **Putting task-specific values in an embodiment** (or robot properties in a
  task). → Task × robot values belong in `adapt/<task>/<embodiment>.yaml`.
- **Restating `task.stages` in an adapt file** when only heights differ. →
  Parameterize the task (`pick_place.*`) and retune the parameters.
- **Restating a whole camera or layer list.** → Replace or `null` the one slot
  that differs.
- **Listing `render_assets/*` or `adapt/*` in a task's `defaults`.** → They are
  auto-selected; a task cannot override `render` / `backend` / `platform`.
- **Selecting a `_`-prefixed fragment directly** (`task=_press_button`). →
  Include it from an option's `defaults`.
- **Forgetting `_self_`**, or placing it at a precedence point that does not
  match the intended override order.

## Related

- [CLI Reference — aao-info](../getting-started/cli_reference.md#aao-info) — discovery, filtering, and the `--vocab` glossary
- [Stages & Waypoints](stages_and_waypoints.md) — the fields you edit in place
- [Scene Composition](scene_composition.md) — scene layers and asset assembly
- [Randomization](randomization.md) — per-object/per-camera randomization ranges
- [Action Space](action_space.md) — operations and operators per embodiment
- [Config groups migration note](../migration-notes/aao_configs_config_groups.md) — old `--config-name` names → `task=` / `embodiment=` / `render=`
