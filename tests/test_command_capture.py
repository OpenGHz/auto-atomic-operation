"""capture_commands(): the command channels of capture_observation(), unrendered."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from auto_atom import CommandObservationEnvProtocol, load_task_file_hydra
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator
from auto_atom.runtime import ComponentRegistry

CONFIG_DIR = Path(__file__).resolve().parents[1] / "aao_configs"


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(map(_same, left, right))
    return np.array_equal(np.asarray(left), np.asarray(right))


@pytest.mark.parametrize("structured", [False, True])
def test_commands_match_the_action_channels_of_a_full_capture(
    structured: bool, monkeypatch
) -> None:
    ComponentRegistry.clear()
    task_file = load_task_file_hydra(
        "pick_and_place",
        config_dir=CONFIG_DIR,
        overrides=[
            "env.viewer=null",
            "++task.seed=1",
            "++env.batch_size=2",
            f"++env.structured={structured}",
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        task_file
    )
    try:
        update = evaluator.reset()
        for _ in range(5):
            update = evaluator.update(policy.act({}, update, evaluator))
        env = evaluator.get_env()
        assert isinstance(env, CommandObservationEnvProtocol)
        full = env.capture_observation()

        renders = []
        for replica in env.envs:
            for renderer in replica._renderers.values():
                monkeypatch.setattr(
                    renderer, "render", lambda: renders.append(1) or None
                )
        commands = env.capture_commands()
    finally:
        evaluator.close()
        ComponentRegistry.clear()

    expected = {key for key in full if "/action/" in f"/{key}"}
    assert expected and set(commands) == expected
    for key in expected:
        assert _same(full[key]["data"], commands[key]["data"]), key
        np.testing.assert_array_equal(full[key]["t"], commands[key]["t"])
    assert renders == []
