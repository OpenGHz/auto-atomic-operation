# Integrating AAO with External Data Collection Programs

This guide explains how to embed AAO in a data collection program without
depending on a particular collector, dataset format, message bus, or training
framework. The common integration boundary is the AAO runner lifecycle and the
environment observation API.

Use this integration when AAO should execute a configured, rule-based task and
the host program should record the resulting trajectory. If an external policy
should produce the actions instead, use
[Policy Evaluation](policy_evaluation.md).

## Responsibilities

Keep task execution and data management separate:

| Component | Responsibility |
|---|---|
| AAO task config | Scene, robot, objects, operation stages, randomization, sensors, and cameras |
| `TaskRunner` | Reset the task, generate and execute primitive actions, and report per-environment completion |
| AAO environment | Advance the simulator and expose timestamped observations and commanded actions |
| Host collector | Schedule episodes, map schemas, stream samples, commit or discard output, and retry work |

The data path is:

```text
AAO task config -> TaskRunner -> environment -> capture_observation()
                         |                         |
                         +---- TaskUpdate --------+-> adapter -> dataset writer
```

The adapter should contain all target-dataset naming, dtype, shape, unit, and
serialization rules. Do not change an AAO task merely to make its observation
keys look like another dataset's schema.

## Choose and Load a Task

Search the existing tasks before creating another config:

```bash
aao-info
aao-info -o pick --object cup
aao-info 'open_door*'
aao-info --json
```

Follow [Reusing & Creating Tasks](../task-configuration/reusing_and_creating_tasks.md):
reuse a matching task and apply Hydra overrides for non-fundamental differences
such as batch size, seed, camera options, or randomization ranges.

Shipped tasks commonly use Hydra `defaults`, so load them with
`load_task_file_hydra()` rather than reading one YAML file directly:

```python
from auto_atom import ComponentRegistry, load_task_file_hydra

ComponentRegistry.clear()
task_file = load_task_file_hydra(
    "pick_and_place",
    config_dir="aao_configs",
    overrides=[
        "env.batch_size=4",
        "+env.viewer.disable=true",
        "task.seed=42",
    ],
)
```

`load_task_file_hydra()` instantiates and registers the configured environment.
Clear the registry before loading a new task in a reused worker process so a
closed environment from an earlier run cannot be selected accidentally.

To collect only a keypoint interval with `TaskRunner`, add
`execution.interval_selection` through Hydra overrides. For example, expose
one complete YAML waypoint per public update, starting in the state after the
completed pick retract and ending in the state after the completed place
retract:

```python
overrides=[
    "+execution.update_boundary=keypoint",
    "+execution.interval_selection.start.stage=pick_source",
    "+execution.interval_selection.start.phase=post_move",
    "+execution.interval_selection.start.waypoint=0",
    "+execution.interval_selection.start.side=after",
    "+execution.interval_selection.stop.stage=place_source",
    "+execution.interval_selection.stop.phase=post_move",
    "+execution.interval_selection.stop.waypoint=0",
    "+execution.interval_selection.stop.side=after",
]
```

`runner.reset()` executes the prefix internally and returns with the start
keypoint fully completed because this example explicitly uses
`start.side=after`. Consequently, `writer.write_initial()` receives the
state after that keypoint, while no prefix frames are exposed to the host
collection loop. The update that completes the stop keypoint and its
completion-bound condition reports terminal success; the host's post-update
observation captures the resulting state. See
[Stages & Waypoints](../task-configuration/stages_and_waypoints.md#task-interval-boundary-selection).

`side=before` instead exposes the state immediately before the referenced
keypoint, without running that keypoint's action or completion-bound
condition. The start default is `before`; the stop default is `after`. This
example spells out both sides because its start deliberately overrides
the default.

`execution.update_boundary` supports four collection granularities:

| Value | One host-visible `runner.update()` |
|---|---|
| `control_tick` | Advances one controller update; default and backward-compatible |
| `primitive` | Completes one runtime primitive; each arc sub-action is a separate boundary |
| `keypoint` | Completes one YAML waypoint; an arc's sub-actions remain grouped |
| `stage` | Completes one whole task stage |

For macro boundaries, AAO still executes the same physics and controller
updates internally. The host receives only the boundary state because there is
currently no public per-internal-update observation callback. Use the default
`control_tick` for dense trajectory collection; use `primitive`, `keypoint`, or
`stage` only when boundary-only samples are intentional.

`execution.render_internal_updates: false` only coalesces passive-viewer
refreshes and skips viewer `step_delay`; it does not change which observations
the host captures or how many physics ticks run.

Two independent safeguards both default to `10000` controller updates per
environment:

- `execution.max_internal_updates_per_update` limits one public macro
  `runner.update()`.
- `execution.interval_selection.max_fast_forward_updates` limits the prefix
  executed by `runner.reset()` to reach the interval start boundary.

An interval stop takes priority over a coarser update boundary. With
`stop.side=before`, a stop in the middle of a stage is captured without
executing the referenced keypoint or the rest of the stage; with `after`, the
referenced keypoint and its completion-bound condition finish first.

These execution options are specific to `TaskRunner` / `aao-demo`.
`PolicyEvaluator` / `aao-eval` rejects interval selection, every
non-`control_tick` boundary, and `render_internal_updates: false` because an
external policy must supply a fresh action for each control tick.

## Core Runner Contract

`TaskRunner` exposes the lifecycle required by a host collector:

| API | Meaning |
|---|---|
| `TaskRunner().from_config(task_file)` | Construct the backend, task plan, and environment |
| `runner.reset(env_mask=None)` | Reset all or selected environments, including AAO randomization |
| `runner.update(env_mask=None)` | Advance to the configured `execution.update_boundary` |
| `runner.get_env()` | Return the environment used by the runner |
| `env.capture_observation()` | Capture timestamped measurements and current command targets |
| `runner.records` | Return accumulated stage-level execution records |
| `runner.close()` | Tear down the backend and environment resources |

Both `reset()` and `update()` return a `TaskUpdate`. Its fields are batched and
include:

- `stage_index`, `stage_name`, `status`, `phase`, and `phase_step`
- `done`: the task reached a terminal state for each environment
- `success`: the terminal state is successful for each environment
- `details`: backend and condition details for the latest update

AAO does not emit a separate `truncated` flag. If the host imposes a maximum
episode length, mark unfinished environments as truncated in the dataset
metadata itself.

## Minimal Collection Loop

The following example shows the complete AAO side of a batched collector.
`writer.start_episode()`, `writer.write_initial()`, `writer.append()`, and
`writer.finish_episode()` are placeholders for the host program's storage API;
they are not AAO methods.

```python
from pathlib import Path
from typing import Any

import numpy as np

from auto_atom import ComponentRegistry, TaskRunner, load_task_file_hydra


def select_env(
    observation: dict[str, dict[str, Any]], env_index: int
) -> dict[str, dict[str, Any]]:
    """Remove the leading batch dimension from one AAO observation."""
    return {
        key: {
            "data": payload["data"][env_index],
            "t": payload["t"][env_index],
        }
        for key, payload in observation.items()
    }


def collect(
    writer,
    *,
    task_name: str,
    config_dir: Path,
    rounds: int,
    batch_size: int,
    max_updates: int,
) -> None:
    ComponentRegistry.clear()
    task_file = load_task_file_hydra(
        task_name,
        config_dir=config_dir,
        overrides=[
            f"env.batch_size={batch_size}",
            "+env.viewer.disable=true",
        ],
    )
    runner = TaskRunner().from_config(task_file)
    env = runner.get_env()

    try:
        for round_index in range(rounds):
            # Complete any host-side mode transition that might reset the
            # simulator before calling runner.reset().
            for env_index in range(batch_size):
                writer.start_episode(round_index, env_index)

            record_start = len(runner.records)
            update = runner.reset()

            # The reset observation is useful as the initial state. Whether it
            # becomes a dataset sample depends on the target schema.
            initial_observation = env.capture_observation()
            for env_index in range(batch_size):
                writer.write_initial(
                    env_index,
                    select_env(initial_observation, env_index),
                )

            updates_used = 0
            while not bool(np.all(update.done)) and updates_used < max_updates:
                active_mask = ~np.asarray(update.done, dtype=bool)
                update = runner.update(active_mask)
                observation = env.capture_observation()

                # active_mask was computed before update(), so this also writes
                # the final sample for environments that completed on this tick.
                for env_index in np.flatnonzero(active_mask):
                    writer.append(
                        int(env_index),
                        select_env(observation, int(env_index)),
                        task_update=update,
                    )
                updates_used += 1

            truncated = ~np.asarray(update.done, dtype=bool)
            round_records = runner.records[record_start:]
            for env_index in range(batch_size):
                writer.finish_episode(
                    env_index,
                    success=bool(update.success[env_index]),
                    truncated=bool(truncated[env_index]),
                    records=[
                        record
                        for record in round_records
                        if record.env_index == env_index
                    ],
                )
    finally:
        runner.close()
        ComponentRegistry.clear()
```

`runner.records` accumulates for the lifetime of the runner. Snapshot its
length at the start of a round, as above, when the writer needs only that
round's records.

`max_updates` in this host loop counts public `runner.update()` calls. With a
macro boundary, one such call can contain many controller updates; use
`TaskUpdate.details[env_index]["execution"]["internal_updates"]` when the
distinction matters for diagnostics or metadata.

## Embed AAO in an Existing Collector State Machine

Collectors that keep batch slots in different states should use the optional
boolean masks accepted by `reset()` and `update()`. A mask must have exactly one
entry per AAO environment and the slot order must remain stable.

The generic pattern is:

```python
handled = np.zeros(batch_size, dtype=bool)

while host.is_running():
    reset_mask = host.slots_ready_to_start()
    if reset_mask.any():
        # Entering sampling may trigger a reset in some collectors.
        host.enter_sampling(reset_mask)
        runner.reset(reset_mask)  # AAO reset and randomization happen last.
        handled[reset_mask] = False

    update_mask = host.slots_in_sampling()
    if not update_mask.any():
        continue

    update = runner.update(update_mask)
    observation = runner.get_env().capture_observation()
    host.write_selected(observation, update_mask)

    new_done = update_mask & np.asarray(update.done) & ~handled
    for env_index in np.flatnonzero(new_done):
        host.finish(
            int(env_index),
            success=bool(update.success[env_index]),
            details=update.details[env_index],
        )
    handled[new_done] = True
```

The `handled` mask is important because `done=True` remains visible until that
environment is reset. Without event de-duplication, a host loop can commit or
discard the same episode more than once.

> [!WARNING]
> `runner.reset()` resets the environment, homes the operator, and applies AAO
> randomization. If the host collector resets the simulator again afterward,
> it can erase that randomization and desynchronize the runner from the data
> being recorded. Make one component the reset owner, or ensure the AAO reset
> is the final reset before sampling begins.

Batch environments may finish at different times. Either keep updating only
unfinished slots, as shown above, or deliberately define a synchronized-batch
policy. Do not end the entire batch merely because one slot is done.

## Observation and Sample Mapping

For the built-in MuJoCo environment, `capture_observation()` returns:

```text
{
    "<key>": {
        "data": <batched value>,
        "t": <batched simulator timestamp>,
    },
    ...
}
```

Important mapping rules:

- Array-like values have a leading batch dimension. With
  `env.structured=true`, nested structured values are collected as one item per
  environment instead.
- `t` is simulator time for that value, not wall-clock time. It is nanoseconds
  when `env.stamp_ns=true` and seconds otherwise.
- Measurement keys such as `arm/pose/position` describe current state. Keys
  under `action/...` describe the current command target produced by AAO.
- Pose positions are expressed in the documented frame, and pose quaternions
  use `xyzw` ordering. Preserve or explicitly convert both conventions in the
  adapter.
- Enabled sensor categories and per-camera `enable_color`, `enable_depth`,
  `enable_mask`, and `enable_heat_map` fields determine which keys exist. A
  collector must validate required keys at startup instead of assuming every
  task exposes the same modalities.

Capture after each `runner.update()` when recording AAO's command targets. This
keeps the post-update simulator state and the `action/...` values on the same
simulator timestamp. If the target dataset uses transition tuples such as
`(observation_t, action_t, observation_t+1)`, perform that temporal shift in the
adapter and test it explicitly.

See [Pose Observation](../mujoco-backend/pose_observation.md),
[Joint State Observation](../mujoco-backend/joint_state_observation.md), and
[sim_freq & update_freq](../task-configuration/sim_freq_update_freq.md) for the
field and timing conventions.

## Episode Completion and Output Transactions

Treat `done` and `success` independently:

| State | Host action |
|---|---|
| `done=True`, `success=True` | Commit a successful episode |
| `done=True`, `success=False` | Save as a failed episode or discard it according to dataset policy |
| `done=False` at the host limit | Mark the episode as host-truncated |

Persist enough metadata to diagnose and reproduce a sample:

- task config name and effective overrides
- random seed, round index, process/worker ID, and environment index
- observation timestamps and sample indices
- final `TaskUpdate` status, success, and details
- the round's `ExecutionRecord` entries

If episodes arrive through a reliable work queue, acknowledge an input only
after the dataset writer has committed and validated the corresponding output.
Keep input consumption, execution, output commit, and acknowledgement as
separate states so an acknowledgement retry cannot accidentally rerun or
duplicate an episode.

## Process and Resource Boundaries

- Always call `runner.close()` in `finally`; it tears down the backend and the
  environment owned by that runner.
- Give each process and environment slot a unique output namespace. Never let
  workers append to the same episode file without an explicit transactional
  writer.
- For multi-GPU collection, prefer independent worker processes and select the
  device before constructing the AAO environment. Process scheduling and
  result merging belong to the host collector.
- Gaussian Splatting is optional. Tasks that enable it require the configured
  assets and compatible GPU dependencies; native MuJoCo rendering can be used
  for tasks that do not require it.
- For headless MuJoCo rendering, configure an off-screen backend before the
  worker imports or constructs the environment. See
  [MuJoCo EGL Troubleshooting](../troubleshooting/mujoco-egl-troubleshooting.md).

## Integration Checklist

- The task was selected with `aao-info` and reused or overridden where possible.
- AAO and the host agree on batch size and stable environment-slot ordering.
- Host mode switches happen before `runner.reset()`, with no later hidden reset.
- Observations are split per environment and mapped in one adapter layer.
- Required keys, shapes, dtypes, units, frames, quaternion order, and timestamps
  are validated before a long run.
- Each completion event is handled once; success, failure, and host truncation
  remain distinct.
- Output is committed before a queued input is acknowledged.
- The runner is closed on normal exit and on exceptions.

## Related

- [Data Collection](data_collection.md) — AAO's built-in recording examples
- [Policy Evaluation](policy_evaluation.md) — let an external policy produce actions
- [Data Replay](mcap_data_replay.md) — replay recorded actions with another runner
- [Execution Completion Flow](../task-configuration/execution_completion_flow.md) — how AAO determines stage success and failure
- [CLI Reference — `aao-info`](../getting-started/cli_reference.md#aao-info) — discover runnable tasks
