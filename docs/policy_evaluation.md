# Policy Evaluation

This document explains how to evaluate an external policy model with `PolicyEvaluator`.

The goal is:

- let your model output low-level actions
- apply those actions directly to the environment
- reuse the framework's stage definitions and success/failure conditions
- collect the same result data classes used by `TaskRunner`

The shared outputs are:

- `TaskUpdate`
- `ExecutionRecord`
- `ExecutionSummary`

## Overview

`TaskRunner` and `PolicyEvaluator` serve different roles:

- `TaskRunner`
  - expands a stage into internal primitive actions such as `pre_move`, `eef`, and `post_move`
  - executes those primitive actions itself
- `PolicyEvaluator`
  - does not generate or execute primitive actions
  - accepts external policy actions
  - applies those actions to the environment
  - checks whether each stage has become successful according to the task definition

This is useful when you already have a trained policy and want to ask:

- how many stages were completed before termination
- which stage the rollout stopped at
- whether the whole task succeeded
- why a rollout failed

## Core API

`PolicyEvaluator` lives in [policy_eval.py](../auto_atom/policy_eval.py).

Typical usage:

```python
from auto_atom import PolicyEvaluator, TaskFileConfig

task_file = TaskFileConfig.model_validate(raw_cfg)

evaluator = PolicyEvaluator(
    action_applier=my_action_applier,
    observation_getter=my_observation_getter,
).from_config(task_file)

update = evaluator.reset()
max_updates = None
step = -1

for step in range(max_updates or 10**18):
    observation = evaluator.get_observation()
    action = policy.act(observation, update=update, evaluator=evaluator)
    update = evaluator.update(action)
    if update.done.all():
        break

summary = evaluator.summarize(update, max_updates=max_updates, updates_used=step + 1)
records = evaluator.records
```

## Shared Result Types

### `TaskUpdate`

Returned by `reset()` and every `update()`.

Use it for online rollout state:

- current stage index and name
- stage status
- whether each env is done
- whether each env succeeded
- details for the latest step

### `ExecutionRecord`

Available through `evaluator.records`.

Each record corresponds to one stage result for one env:

- `SUCCEEDED`
- `FAILED`

### `ExecutionSummary`

Returned by `summarize(...)`.

Use it for final aggregate metrics:

- `updates_used`
- `completed_stage_count`
- `final_stage_index`
- `final_stage_name`
- `final_done`
- `final_success`

## Policy Interface

The package entry point is [auto_atom/runner/policy_eval.py](../auto_atom/runner/policy_eval.py).
By default, `aao_eval` resolves Hydra configs from `./aao_configs/` relative to the current working directory.

If the config does not provide a `policy` section, `aao_eval` will default to
`auto_atom.ConfigDrivenDemoPolicy`, so you can directly evaluate a normal demo
task config without creating a separate evaluation YAML.

When you do provide a custom policy, the object is expected to support:

- optional `reset()`
- `act(observation)`
- or `act(observation, update=..., evaluator=...)`
- optional `action_applier(context, action, env_mask=None)`
- optional `observation_getter(context)`

You may also provide a plain callable instead of an object with `act()`.

The example script detects the callable signature and passes:

- `observation`
- optionally `update`
- optionally `evaluator`

So these are all valid:

```python
class MyPolicy:
    def reset(self):
        ...

    def act(self, observation):
        return action
```

```python
class MyPolicy:
    def act(self, observation, update, evaluator):
        return action
```

```python
def my_policy(observation, update=None, evaluator=None):
    return action
```

If the instantiated policy object exposes `action_applier` or `observation_getter`,
`aao_eval` will prefer those over the built-in defaults.

## Config-Driven Demo Policy

The package includes `auto_atom.ConfigDrivenDemoPolicy`, and `aao_eval` uses it
by default when `policy` is omitted from the config.

It does not use a learned model. Instead, it rebuilds the same primitive
`pre_move` / `eef` / `post_move` action sequence that `TaskRunner` uses in
`aao_demo`, and applies those primitives through the same operator handlers.

This is useful as a consistency check:

- if `aao_demo` succeeds on a task
- then `aao_eval` with `ConfigDrivenDemoPolicy` should also succeed
- so you can verify that the stage-completion logic used by `PolicyEvaluator`
  is aligned with the framework runtime

Optional explicit config:

```yaml
policy:
  _target_: auto_atom.ConfigDrivenDemoPolicy
```

## Default Action Applier

The package runner ships with a default action applier that expects:

- `backend.env.step(action, env_mask=...)`

This matches the MuJoCo basis environment's batched low-level action API.

Supported policy return types are intentionally narrow:

- `None`
  - do not step the environment on this tick
- `np.ndarray`
  - shape `(B, action_dim)`
  - or shape `(action_dim,)` when `batch_size == 1`
- `torch.Tensor`
  - same shape rules as above
- `list` or `tuple`
  - must be convertible to the same array shapes
- `dict` with an `"action"` field
  - the `"action"` value must be one of the supported payload types above

Invalid shapes are rejected early.

### Shape Rules

When `batch_size == 1`:

- `(action_dim,)` is accepted and reshaped to `(1, action_dim)`
- `(1, action_dim)` is also accepted

When `batch_size > 1`:

- actions must already be batched
- required shape is `(batch_size, action_dim)`

## Custom Action Applier

If your policy output does not match `env.step(...)` directly, provide your own `action_applier`.

Typical reasons:

- your model outputs normalized actions and you need to de-normalize
- your model outputs a dict with multiple heads
- your environment expects separate arm/gripper channels
- you want to post-process actions before applying them

Example:

```python
def my_action_applier(context, action, env_mask=None):
    backend = context.backend
    env = backend.env

    arm = action["arm"]
    gripper = action["gripper"]
    low_level = convert_to_env_action(arm, gripper)
    env.step(low_level, env_mask=env_mask)
```

## Observation Getter

If you do not provide `observation_getter`, `PolicyEvaluator.get_observation()` tries:

- `backend.env.capture_observation()`

That is enough for the built-in MuJoCo setup.

If your policy needs a different observation format, provide a custom getter:

```python
def my_observation_getter(context):
    raw = context.backend.env.capture_observation()
    return convert_obs_for_model(raw)
```

## Stage Success Logic

`PolicyEvaluator` reuses the same task-stage success logic as the runtime.

Examples:

- `move`
  - success condition: `reached`
- `grasp`
  - success condition: `grasped`
- `release`
  - success condition: `released`
- `pick`
  - success condition: `grasped`
- `place`
  - success condition: `released`
- `push`
  - success condition: `displaced`
- `pull`
  - success condition: `grasped`
- `press`
  - success condition: `contacted`

This means policy evaluation is not based on ad-hoc metrics. It uses the same stage semantics defined by the framework.

## Example Script

The repository includes:

- [auto_atom/runner/policy_eval.py](../auto_atom/runner/policy_eval.py)
- [policy_eval_mock.yaml](../aao_configs/policy_eval_mock.yaml)

Run evaluation directly on a demo config:

```bash
aao_eval --config-name pick_and_place
```

Or run the mock example:

```bash
aao_eval --config-name policy_eval_mock
```

`policy_eval_mock.yaml` explicitly pins `auto_atom.ConfigDrivenDemoPolicy`, but
that is now equivalent to omitting `policy` entirely.

The mock example is mainly for verifying the control loop and outputs:

- `TaskUpdate`
- `ExecutionRecord`
- `ExecutionSummary`

To evaluate a real model:

1. replace the mock config with your task config
2. instantiate your policy from `cfg.policy`
3. ensure the returned action matches the default action applier format, or provide a custom one

## Remote Evaluation (rpyc)

When the client machine does not have simulation dependencies (MuJoCo, etc.), you can split evaluation into a **server** (holds the simulator) and a **client** (only needs `rpyc` + `auto_atom`).

```
Server (MuJoCo installed)              Client (no sim deps)
┌──────────────────────────┐           ┌──────────────────────────┐
│ serve_policy_evaluator() │   rpyc    │ RemotePolicyEvaluator    │
│ wraps PolicyEvaluator    │◄─────────►│ same public API          │
└──────────────────────────┘           └──────────────────────────┘
```

### Install Dependencies

```bash
pip install rpyc
```

The server machine also needs the simulation backend (e.g. `mujoco`). The client machine only needs `rpyc` and `auto_atom`.

### Start the Server

```bash
python examples/policy_eval_server.py
python examples/policy_eval_server.py --host 0.0.0.0 --port 9999
```

Or from Python:

```python
from auto_atom.ipc import serve_policy_evaluator

serve_policy_evaluator(host="0.0.0.0", port=18861)
```

The server starts with a default `action_applier` that calls `env.apply_pose_action(...)` and a default `observation_getter` that calls `env.capture_observation()`. To use custom callbacks:

```python
serve_policy_evaluator(
    host="0.0.0.0",
    port=18861,
    action_applier=my_action_applier,
    observation_getter=my_observation_getter,
)
```

### Use the Client

```python
from auto_atom.ipc import RemotePolicyEvaluator

evaluator = RemotePolicyEvaluator(host="localhost", port=18861)

# Initialize — the server loads config and creates the simulator
evaluator.from_config("press_three_buttons", overrides=["env.batch_size=1"])
# or: evaluator.from_yaml("path/to/task.yaml")

# Same API as PolicyEvaluator
update = evaluator.reset()

for step in range(max_steps):
    obs = evaluator.get_observation()
    action = policy.act(obs, update)
    update = evaluator.update(action)
    if update.done.all():
        break

summary = evaluator.summarize(max_updates=max_steps, updates_used=step + 1)
records = evaluator.records
evaluator.close()
```

All return types (`TaskUpdate`, `ExecutionRecord`, `ExecutionSummary`) are real `auto_atom` dataclass instances, not plain dicts.

### Client API Reference

| Method / Property | Returns | Description |
|---|---|---|
| `from_config(name, overrides=)` | `self` | Load task config on the server by Hydra config name |
| `from_yaml(path)` | `self` | Load task config on the server from a YAML path |
| `reset(env_mask=)` | `TaskUpdate` | Reset environments |
| `get_observation()` | `Any` | Get current observation |
| `update(action, env_mask=)` | `TaskUpdate` | Apply action and advance state |
| `summarize(...)` | `ExecutionSummary` | Get execution summary |
| `records` | `List[ExecutionRecord]` | Stage results |
| `batch_size` | `int` | Number of environments |
| `stage_plans` | `List[dict]` | Stage info (index, name, operator, operation, object) |
| `ping()` | `dict` | Health check |
| `close()` | `None` | Release resources and disconnect |

### Import Isolation

The `auto_atom.ipc` subpackage is fully isolated. `import auto_atom` does **not** trigger `import rpyc`:

```python
# This always works, even without rpyc installed
import auto_atom

# This requires rpyc
from auto_atom.ipc import RemotePolicyEvaluator
from auto_atom.ipc import serve_policy_evaluator
```

### Complete Example

See [examples/policy_eval_server.py](../examples/policy_eval_server.py) and [examples/policy_eval_client.py](../examples/policy_eval_client.py) for a working example that replays a recorded demo through the remote evaluator.

```bash
# Terminal 1
python examples/policy_eval_server.py

# Terminal 2
python examples/policy_eval_client.py
python examples/policy_eval_client.py --host 10.0.0.5 --port 9999
```

## Recommended Integration Pattern

For a trained low-level policy, the cleanest setup is:

1. keep task/stage definitions in YAML
2. keep the backend responsible for object and contact/grasp state
3. let the model control the environment directly
4. let `PolicyEvaluator` judge stage completion and task success

That gives you a rollout evaluator without duplicating the framework's task semantics.
