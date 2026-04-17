from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


def test_task_runner_update_accepts_env_mask() -> None:
    ComponentRegistry.clear()
    ComponentRegistry.register_env("mock_batch", {"kind": "mock_env", "batch_size": 2})
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            {
                "backend": "auto_atom.mock.build_mock_backend",
                "task": {
                    "env_name": "mock_batch",
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
        )
    )

    try:
        reset_update = runner.reset()
        assert reset_update.phase == [None, None]

        first_update = runner.update(np.asarray([True, False], dtype=bool))
        assert first_update.phase == ["pre_move", None]
        assert first_update.phase_step.tolist() == [0, -1]
        assert first_update.status.tolist() == ["running", "pending"]

        second_update = runner.update(np.asarray([True, False], dtype=bool))
        assert second_update.done.tolist() == [True, False]
        assert second_update.success.tolist() == [True, None]
        assert second_update.status.tolist() == ["succeeded", "pending"]

        third_update = runner.update(np.asarray([False, True], dtype=bool))
        assert third_update.done.tolist() == [True, False]
        assert third_update.phase == [None, "pre_move"]
        assert third_update.phase_step.tolist() == [-1, 0]
        assert third_update.status.tolist() == ["succeeded", "running"]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_task_runner_update_requires_reset_for_selected_envs() -> None:
    ComponentRegistry.clear()
    ComponentRegistry.register_env("mock_batch", {"kind": "mock_env", "batch_size": 2})
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            {
                "backend": "auto_atom.mock.build_mock_backend",
                "task": {
                    "env_name": "mock_batch",
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
        )
    )

    try:
        try:
            runner.update(np.asarray([True, False], dtype=bool))
        except RuntimeError as exc:
            message = str(exc)
            assert "have not been reset" in message
            assert "[0]" in message
        else:
            raise AssertionError("Expected update() to fail before reset().")

        runner.reset(np.asarray([True, False], dtype=bool))

        try:
            runner.update(np.asarray([False, True], dtype=bool))
        except RuntimeError as exc:
            message = str(exc)
            assert "have not been reset" in message
            assert "[1]" in message
        else:
            raise AssertionError("Expected update() to fail for an unreset env.")
    finally:
        runner.close()
        ComponentRegistry.clear()
