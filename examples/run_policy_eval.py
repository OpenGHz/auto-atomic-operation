"""Run a policy-driven evaluation rollout.

Config files live in the ``mujoco/`` subdirectory. The default is
``mujoco/policy_eval_mock.yaml``.

This script is intentionally lightweight:

- the environment/backend still come from the task YAML
- a policy object can be optionally instantiated from ``cfg.policy``
- the evaluator reuses ``TaskUpdate`` / ``ExecutionRecord`` / ``ExecutionSummary``

The policy interface is intentionally strict:

- optional ``reset()``
- either ``act(observation)`` or ``act(observation, update=..., evaluator=...)``
- or any callable with the same signature

Supported policy return types for the default action applier:

- ``None``: do not step the environment this tick
- ``np.ndarray`` with shape ``(B, action_dim)``
- ``np.ndarray`` with shape ``(action_dim,)`` when ``batch_size == 1``
- ``torch.Tensor`` with the same shape rules
- Python ``list`` / ``tuple`` convertible to the same array shapes
- ``dict`` containing an ``"action"`` field with one of the above payloads
"""

from __future__ import annotations

import inspect
import sys
from itertools import count
from pathlib import Path
from pprint import pprint
from typing import Any

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from auto_atom import ComponentRegistry, PolicyEvaluator, TaskFileConfig


class NoOpPolicy:
    """Fallback example policy that keeps the current low-level controls unchanged."""

    def reset(self) -> None:
        return None

    def act(self, observation: Any, **_: Any) -> Any:
        _ = observation
        return None


def _list_demos() -> None:
    config_dir = Path(__file__).parent / "mujoco"
    names = sorted(p.stem for p in config_dir.glob("*.yaml"))
    print(f"Available demos ({len(names)}):")
    for name in names:
        print(f"  {name}")


def _default_observation_getter(context) -> Any:
    backend = context.backend
    env = getattr(backend, "env", None)
    if env is not None and hasattr(env, "capture_observation"):
        return env.capture_observation()
    return {}


def _default_action_applier(
    context, action: Any, env_mask: np.ndarray | None = None
) -> None:
    if action is None:
        return
    backend = context.backend
    env = getattr(backend, "env", None)
    if env is None or not hasattr(env, "step"):
        raise RuntimeError(
            "Default action applier requires backend.env.step(action, env_mask=...)."
        )
    normalized_action = _normalize_action_for_env_step(action, backend.batch_size)
    env.step(normalized_action, env_mask=env_mask)


def _extract_action_payload(action: Any) -> Any:
    if isinstance(action, dict):
        if "action" not in action:
            raise TypeError(
                "When policy returns a dict, it must contain an 'action' field."
            )
        return action["action"]
    return action


def _to_numpy_action(action: Any) -> np.ndarray:
    payload = _extract_action_payload(action)
    if payload is None:
        raise TypeError("Action payload cannot be None after extraction.")
    if isinstance(payload, np.ndarray):
        return payload.astype(np.float64, copy=False)
    if (
        hasattr(payload, "detach")
        and hasattr(payload, "cpu")
        and hasattr(payload, "numpy")
    ):
        return payload.detach().cpu().numpy().astype(np.float64, copy=False)
    if isinstance(payload, (list, tuple)):
        return np.asarray(payload, dtype=np.float64)
    raise TypeError(
        "Unsupported action payload type for default action applier. "
        "Expected numpy array, torch tensor, list/tuple, or {'action': ...}."
    )


def _normalize_action_for_env_step(action: Any, batch_size: int) -> np.ndarray:
    array = _to_numpy_action(action)
    if array.ndim == 1:
        if batch_size != 1:
            raise ValueError(
                "1D action is only supported when batch_size == 1. "
                f"Got batch_size={batch_size} and action shape {array.shape}."
            )
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            "Action for default action applier must be rank-1 or rank-2. "
            f"Got shape {array.shape}."
        )
    if array.shape[0] != batch_size:
        raise ValueError(
            "Batched action must have leading dimension equal to backend.batch_size. "
            f"Expected {batch_size}, got shape {array.shape}."
        )
    return array


def _call_policy(
    policy: Any, observation: Any, update: Any, evaluator: PolicyEvaluator
) -> Any:
    fn = getattr(policy, "act", None)
    if fn is None:
        if callable(policy):
            fn = policy
        else:
            raise TypeError("Policy must be callable or expose an act() method.")
    signature = inspect.signature(fn)
    kwargs = {}
    if "update" in signature.parameters:
        kwargs["update"] = update
    if "evaluator" in signature.parameters:
        kwargs["evaluator"] = evaluator
    return fn(observation, **kwargs)


if "--list" in sys.argv:
    _list_demos()
    sys.exit(0)


@hydra.main(config_path="mujoco", config_name="policy_eval_mock", version_base=None)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    ComponentRegistry.clear()
    if "env" in cfg and cfg.env is not None:
        instantiate(cfg.env)

    task_file = TaskFileConfig.model_validate(raw)
    policy_cfg = cfg.get("policy")
    policy = instantiate(policy_cfg) if policy_cfg is not None else NoOpPolicy()
    max_updates = int(cfg.get("max_updates", 100))
    rounds = int(cfg.get("rounds", 1))
    use_input = bool(cfg.get("use_input", False))
    round_summaries = []

    evaluator = PolicyEvaluator(
        action_applier=_default_action_applier,
        observation_getter=_default_observation_getter,
    ).from_config(task_file)

    try:
        for r in range(rounds):
            if hasattr(policy, "reset"):
                policy.reset()

            if rounds > 1:
                print(f"Round {r + 1}/{rounds}")
                print("=" * 50)

            print("Reset evaluator")
            update = evaluator.reset()
            pprint(update, sort_dicts=False)
            print("Starting policy rollout...")
            print()

            steps_used = 0
            for step in count():
                if use_input:
                    input("Press Enter to continue...")
                if step >= max_updates:
                    print(f"Reached max_updates={max_updates}, stopping rollout.")
                    break

                observation = evaluator.get_observation()
                action = _call_policy(policy, observation, update, evaluator)
                update = evaluator.update(action)
                steps_used += 1
                print(f"Step {step}:" + "=" * 40)
                pprint(update, sort_dicts=False)
                if bool(np.all(update.done)):
                    break

            summary = evaluator.summarize(
                update,
                max_updates=max_updates,
                updates_used=steps_used,
            )

            print()
            print("Execution records:")
            for record in evaluator.records:
                pprint(record)

            print()
            print("Summary:")
            pprint(summary)

            round_summaries.append(summary)

            if rounds > 1:
                print()
    finally:
        evaluator.close()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_success = sum(1 for s in round_summaries if bool(np.all(s.final_success)))
    print(f"Success rate: {n_success}/{len(round_summaries)}")
    print()
    for i, summary in enumerate(round_summaries, start=1):
        tag = "OK" if bool(np.all(summary.final_success)) else "FAIL"
        print(f"  Round {i}: [{tag}]")
        print(f"    total updates: {summary.updates_used}")
        print(f"    completed stages: {summary.completed_stage_count.tolist()}")
        print(f"    final stage: {summary.final_stage_name}")
        print(f"    final success: {summary.final_success.tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
