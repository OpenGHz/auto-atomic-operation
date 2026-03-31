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

- how many stages were completed within `max_updates`
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

for step in range(max_updates):
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

The example entry script is [run_policy_eval.py](../examples/run_policy_eval.py).

The policy object is expected to support:

- optional `reset()`
- `act(observation)`
- or `act(observation, update=..., evaluator=...)`

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

## Default Action Applier

The example script ships with a default action applier that expects:

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

- [run_policy_eval.py](../examples/run_policy_eval.py)
- [policy_eval_mock.yaml](../examples/mujoco/policy_eval_mock.yaml)

Run the mock example:

```bash
python examples/run_policy_eval.py --config-name policy_eval_mock
```

The mock example is mainly for verifying the control loop and outputs:

- `TaskUpdate`
- `ExecutionRecord`
- `ExecutionSummary`

To evaluate a real model:

1. replace the mock config with your task config
2. instantiate your policy from `cfg.policy`
3. ensure the returned action matches the default action applier format, or provide a custom one

## Recommended Integration Pattern

For a trained low-level policy, the cleanest setup is:

1. keep task/stage definitions in YAML
2. keep the backend responsible for object and contact/grasp state
3. let the model control the environment directly
4. let `PolicyEvaluator` judge stage completion and task success

That gives you a rollout evaluator without duplicating the framework's task semantics.
