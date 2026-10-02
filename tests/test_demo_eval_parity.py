"""Parity tests between TaskRunner (aao-demo) and PolicyEvaluator + ConfigDrivenDemoPolicy (aao-eval).

The PolicyEvaluator path duplicates parts of TaskRunner so an external
policy can be plugged in. The ConfigDrivenDemoPolicy is supposed to
replay exactly the same primitive actions as TaskRunner. This test
fixes a deterministic seed and asserts both paths produce equivalent
outcomes for the same config; any future divergence (e.g. a new
``_run_action`` parameter mirrored on only one side) will be caught
here.
"""

from pathlib import Path
import sys
from typing import Any, List, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.config_loader import compose_task_config
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator
from auto_atom.runner.common import prepare_task_file
from auto_atom.runner.policy_eval import (
    _default_action_applier,
    _default_observation_getter,
)
from auto_atom.runtime import ComponentRegistry, TaskRunner


# Configs chosen to cover:
#   - pick_and_place: no `site`, no waypoint randomization (baseline)
#   - open_door on airbot_play_g2p: uses `site: handle_grasp_front_site` AND
#     has per-waypoint `randomization` on the grasp pose — covers exactly the
#     fields whose missing forwarding caused the original eval regression.
PARITY_CASES = [
    ("pick_and_place", []),
    ("open_door", ["embodiment=airbot_play_g2p"]),
]


def _compose_cfg(task: str, extra_overrides: List[str]) -> Any:
    overrides = [
        "env.batch_size=2",
        "env.viewer=null",
        "task.seed=20260508",
        *extra_overrides,
    ]
    return compose_task_config(task, overrides, config_dir=ROOT / "aao_configs")


def _run_demo(task: str, extra_overrides: List[str]) -> Tuple[List[bool], List[str]]:
    ComponentRegistry.clear()
    cfg = _compose_cfg(task, extra_overrides)
    task_file = prepare_task_file(cfg)
    runner = TaskRunner().from_config(task_file)
    try:
        update = runner.reset()
        for _ in range(2000):
            if bool(np.all(update.done)):
                break
            update = runner.update()
        assert bool(np.all(update.done)), "demo did not finish in 2000 updates"
        return (
            update.success.tolist(),
            [r.status.value for r in runner.records],
        )
    finally:
        runner.close()
        ComponentRegistry.clear()


def _run_eval(task: str, extra_overrides: List[str]) -> Tuple[List[bool], List[str]]:
    ComponentRegistry.clear()
    cfg = _compose_cfg(task, extra_overrides)
    task_file = prepare_task_file(cfg)
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(
        action_applier=policy.action_applier,
        observation_getter=_default_observation_getter,
    ).from_config(task_file)
    try:
        update = evaluator.reset()
        for _ in range(2000):
            if bool(np.all(update.done)):
                break
            action = policy.act({}, update, evaluator)
            update = evaluator.update(action)
        assert bool(np.all(update.done)), "eval did not finish in 2000 updates"
        return (
            update.success.tolist(),
            [r.status.value for r in evaluator.records],
        )
    finally:
        evaluator.close()
        ComponentRegistry.clear()


@pytest.mark.parametrize("task,extra_overrides", PARITY_CASES)
def test_demo_eval_parity(task: str, extra_overrides: List[str]) -> None:
    demo_success, demo_records = _run_demo(task, extra_overrides)
    eval_success, eval_records = _run_eval(task, extra_overrides)
    case = " ".join([f"task={task}", *extra_overrides])
    assert demo_success == eval_success, (
        f"per-env success differs for {case}: demo={demo_success} eval={eval_success}"
    )
    assert demo_records == eval_records, (
        f"stage record statuses differ for {case}: "
        f"demo={demo_records} eval={eval_records}"
    )


if __name__ == "__main__":
    for case in PARITY_CASES:
        test_demo_eval_parity(*case)
        print(f"OK: {' '.join([f'task={case[0]}', *case[1]])}")
