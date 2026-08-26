# Reusing & Creating Tasks — Agent Guide

This guide tells an agent (or a human) how to satisfy a request like *"I need a
task that does X"* **efficiently**: reuse what exists, tweak it the cheapest way
that works, and only create a new config file when the task is fundamentally
different.

> **Golden rule.** A new `aao_configs/*.yaml` is justified **only** by a
> *fundamental* difference — a different **robot/embodiment**, a different
> **object set / scene**, or a different **operation flow** (the sequence of
> stages/operations). If none of those change, do **not** add a file: adjust the
> existing task in place, or pass a command-line override.

## Decision flow

```
User asks for a task
        │
        ▼
1. Does a matching task already exist?  ──► run `aao-info` (search / filter / --vocab)
        │ yes                                        │ no exact match, but a close one exists
        ▼                                            ▼
   Run it as-is.                          2. Is the difference FUNDAMENTAL?
   `aao-demo --config-name <name>`            (robot? objects/scene? operation flow?)
                                                 │ no                        │ yes
                                                 ▼                           ▼
                                    3a. One-off / experimental?     4. Create a NEW config by
                                        → CLI override                  COMPOSING the closest
                                        (no file change)                existing base; override
                                    3b. Permanent change?               only what differs.
                                        → edit the task YAML            Follow the naming rules.
                                        in place (no new file)
                                                 │                           │
                                                 └──────────► 5. Verify with `aao-info` + a run.
```

## Step 1 — Discover what already exists

`aao-info` is the reuse-discovery tool. Never hand-scan `aao_configs/`; query it:

```bash
aao-info                     # every runnable task: operators, robots, objects, workflow
aao-info -o press            # tasks that use a given operation
aao-info --object cup        # tasks that manipulate a given object
aao-info -r airbot           # tasks for a given robot model
aao-info 'cup_on_coaster*'   # glob over config names
aao-info --vocab             # keyword glossary (all operations/objects/robots/scenes) for retrieval
```

See the [CLI Reference](../getting-started/cli_reference.md#aao-info) for every
flag. Start from the closest existing task — most requests are a small delta on
one that already ships.

## Step 2 — Classify the difference

| The request changes… | Fundamental? | What to do |
|---|---|---|
| A waypoint height / grasp offset / approach pose | No | Edit in place, or CLI override |
| Randomization range, tolerance, seed, rounds, batch size | No | Edit in place, or CLI override |
| Camera / viewer framing | No | Edit in place, or CLI override |
| Toggling an already-parameterised option (e.g. an existing GS/render flag) | No | CLI override |
| The **robot / gripper / embodiment** | **Yes** | New config composing a different `basis_*` |
| The **object set or the scene** | **Yes** | New config (+ scene XML / assets if truly new) |
| The **operation flow** (which operations, in what order) | **Yes** | New config with new `task.stages` |

When in doubt, ask: *"Would this change break the task for its current users?"*
If yes it is fundamental (make a new config); if it is just a better/alternate
value for the same task, it is not (edit or override).

## Step 3 — Non-fundamental changes (the common case)

### 3a. CLI override — for one-off, experimental, or swept values

Hydra lets you override any key on the command line, so you can explore without
touching a file. Dotted paths index into lists too:

```bash
# tweak a single waypoint of stage 0, run 3 rounds, 4 envs, fixed seed
aao-demo --config-name pick_and_place \
    task.stages.0.param.pre_move.1.position="[0.0, 0.0, 0.008]" \
    rounds=3 env.batch_size=4 task.seed=1
```

Use this for quick experiments and parameter sweeps — anything you would not
want to commit as the task's new default.

### 3b. Edit in place — for a permanent change to an existing task

If the new value *should become* the task's behaviour, edit that task's YAML
directly (its `task.stages`, `task.randomization`, `env.viewer`, …). **Do not
clone the file to change one number** — that creates a near-duplicate that
silently drifts from the original. See
[Stages & Waypoints](stages_and_waypoints.md) and
[Randomization](randomization.md) for the fields.

## Step 4 — Fundamental changes: create a new config by composing

A new variant should **compose** the closest existing config through `defaults`
and override *only* what differs — never copy a whole task to change one layer.
Always end the `defaults` list with `_self_` (see
[GS demo pattern / composition order](scene_composition.md)).

**Render (GS) variant — pure composition, a handful of lines:**

```yaml
# cup_on_coaster_gs.yaml
defaults:
  - cup_on_coaster     # reuse the full base task (stages, objects, randomization)
  - robotiq_gs         # robot's GS building block
  - gs_mixin           # GS rendering mixin
  - _self_
env:
  gaussian_render:
    body_gaussians:
      cup_gs: ${gs_dir}/cup.ply
      coaster_gs: ${gs_dir}/coaster.ply
```

**Robot variant — swap the `basis_*`, keep the intent:** a different robot
generally needs its own grasp orientation / IK, so its stages legitimately
differ. Compose the robot base and redeclare only what the new embodiment
requires:

```yaml
# pick_and_place_franka.yaml
defaults:
  - basis_franka       # robot + eef definition (instead of basis_mocap_eef)
  - _self_
scene_name: pick_and_place
env:
  scene:
    layers:
      - kind: mjcf
        path: ${assets_dir}/xmls/robots/panda_robotiq.xml
  mask_objects: ["source_block", "target_pedestal"]
  operations: ["pick", "place"]
task:
  stages: ...          # only where the robot forces different poses/IK
```

If two variants share *identical* stages and differ only by robot base, prefer a
shared mixin over duplicated stages so they cannot drift apart.

## Step 5 — Naming conventions (only when a file is actually created)

Runnable task configs live at the **top level** of `aao_configs/` and are named
`snake_case`, scene-descriptive, with optional suffixes composed left→right:

```
<task>[_<robot>][_gs].yaml
```

| Kind | Pattern | Examples |
|---|---|---|
| Base task | `<task>.yaml` | `pick_and_place`, `cup_on_coaster`, `open_door` |
| Render variant | `<task>_gs.yaml` | `cup_on_coaster_gs`, `stack_color_blocks_gs` |
| Robot variant | `<task>_<robot>.yaml` | `pick_and_place_franka`, `pick_and_place_xf9600` |
| Robot + render | `<task>_<robot>_gs.yaml` | `open_door_airbot_play_gs` |

- The `<robot>` segment names the embodiment (arm and, where it matters,
  gripper): `franka`, `xf9600`, `umi_v3`, `airbot_p7`, `airbot_play_g2p`, …
  **Match sibling variants** of the same task rather than inventing a new spelling.
- **Reserved building-block names — never name a runnable task like these:**
  `basis_*` (robot/eef/scene bases), `*_mixin` (reusable fragments),
  `*_vars` (variable files), `robotiq_gs` / `gs_mixin` / `airbot_*_gs`
  (GS building blocks). These are meant to be *included*, not run; `aao-info`
  hides them (they declare no `task.stages`), and the name signals intent to
  readers.
- Scratch / experimental configs go in `aao_configs/test/`, not the top level.

## Step 6 — Verify

```bash
aao-info <new_or_edited_name>            # confirm it is detected as a task with the
                                         # expected operators / robots / objects / workflow
aao-demo --config-name <name>            # actually run it (use `mock` backend for a dry check)
```

If `aao-info` does not list your config, it composed without a non-empty
`task.stages` (so it is not a task) or the composition errored — run
`aao-info --verbose` to see the skip reason.

## Anti-patterns

- **Cloning a whole task file to change one waypoint / range / seed.** → Edit in
  place, or use a CLI override.
- **Duplicating `task.stages` across variants that only differ by robot base**
  when the stages are identical. → Share via the base / a mixin.
- **Naming a runnable task `basis_*` / `*_mixin` / `*_vars`.** → It will be
  hidden by `aao-info` and misleads readers.
- **Adding a config without composing** (copy-pasting a full base you could have
  listed in `defaults`). → Compose and override only the delta.
- **Forgetting `_self_`** at the end of a new `defaults` list.

## Related

- [CLI Reference — aao-info](../getting-started/cli_reference.md#aao-info) — discovery, filtering, and the `--vocab` glossary
- [Stages & Waypoints](stages_and_waypoints.md) — the fields you edit in place
- [Scene Composition](scene_composition.md) — `defaults`, composition order, `_self_`
- [Randomization](randomization.md) — per-object/per-camera randomization ranges
- [Action Space](action_space.md) — operations and operators
