"""Console entry point for policy evaluation."""

from __future__ import annotations

import inspect
import sys
from typing import Any

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator

from .common import (
    ExampleLoopHooks,
    get_config_dir,
    list_demos,
    prepare_task_file,
    print_final_summary,
    run_example_rounds,
)


def _default_observation_getter(context) -> Any:
    backend = context.backend
    env = getattr(backend, "env", None)
    if env is not None and hasattr(env, "capture_observation"):
        return env.capture_observation()
    return {}


def _default_action_applier(context, action: Any, env_mask: Any = None) -> None:
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
    list_demos(get_config_dir())
    sys.exit(0)


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    task_file = prepare_task_file(cfg)
    policy_cfg = cfg.get("policy")
    policy = (
        instantiate(policy_cfg) if policy_cfg is not None else ConfigDrivenDemoPolicy()
    )
    action_applier = getattr(policy, "action_applier", _default_action_applier)
    observation_getter = getattr(
        policy,
        "observation_getter",
        _default_observation_getter,
    )
    max_updates_cfg = cfg.get("max_updates")
    max_updates = None if max_updates_cfg is None else int(max_updates_cfg)
    rounds = int(cfg.get("rounds", 1))
    use_input = bool(cfg.get("use_input", False))
    get_obs = bool(cfg.get("get_obs", False))

    evaluator = PolicyEvaluator(
        action_applier=action_applier,
        observation_getter=observation_getter,
    ).from_config(task_file)

    try:
        round_summaries = run_example_rounds(
            rounds=rounds,
            use_input=use_input,
            hooks=ExampleLoopHooks(
                reset_fn=evaluator.reset,
                step_fn=lambda _step, update: evaluator.update(
                    _call_policy(
                        policy,
                        evaluator.get_observation() if get_obs else {},
                        update,
                        evaluator,
                    )
                ),
                summarize_fn=lambda update, steps_used, max_updates, elapsed_time_sec: (
                    evaluator.summarize(
                        update,
                        max_updates=max_updates,
                        updates_used=steps_used,
                        elapsed_time_sec=elapsed_time_sec,
                    )
                ),
                records_fn=lambda: evaluator.records,
                before_round_fn=lambda _r: getattr(policy, "reset", lambda: None)(),
                reset_label="Reset evaluator",
                start_label="Starting policy rollout...",
                max_updates=max_updates,
            ),
        )
    finally:
        evaluator.close()

    print_final_summary(round_summaries)


if __name__ == "__main__":
    main()
