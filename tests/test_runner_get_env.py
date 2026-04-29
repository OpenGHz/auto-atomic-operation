from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.mock import MockEnv
from auto_atom.runner.data_replay import DataReplayRunner, DataReplayTaskFileConfig
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


def _task_payload(env_name: str, batch_size: int = 2) -> dict:
    ComponentRegistry.register_env(
        env_name,
        {"kind": "mock_env", "batch_size": batch_size},
    )
    return {
        "backend": "auto_atom.mock.build_mock_backend",
        "task": {
            "env_name": env_name,
            "stages": [
                {
                    "name": "move_block",
                    "object": "block",
                    "operation": "move",
                    "operator": "arm",
                    "param": {
                        "pre_move": [
                            {
                                "reference": "world",
                                "position": [0.5, 0.0, 0.3],
                                "orientation": [0.0, 0.0, 0.0, 1.0],
                            }
                        ]
                    },
                }
            ],
        },
        "task_operators": {"arm": {}},
    }


def test_task_runner_get_env_returns_backend_env() -> None:
    ComponentRegistry.clear()
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(_task_payload("mock_runner_get_env"))
    )

    try:
        env = runner.get_env()
        assert isinstance(env, MockEnv)
        assert env.batch_size == 2
    finally:
        runner.close()
        ComponentRegistry.clear()

    with pytest.raises(RuntimeError, match="not initialized"):
        runner.get_env()


def test_data_replay_runner_get_env_returns_backend_env() -> None:
    ComponentRegistry.clear()
    payload = _task_payload("mock_replay_runner_get_env", batch_size=3)
    payload["replay"] = {"load_on_initialize": False}
    runner = DataReplayRunner().from_config(
        DataReplayTaskFileConfig.model_validate(payload)
    )

    try:
        env = runner.get_env()
        assert isinstance(env, MockEnv)
        assert env.batch_size == 3
    finally:
        runner.close()
        ComponentRegistry.clear()

    with pytest.raises(RuntimeError, match="not initialized"):
        runner.get_env()
