from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_replay_demo_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "replay_demo.py"
    spec = importlib.util.spec_from_file_location("replay_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_policy_uses_first_frame_for_reset_without_replaying_it():
    replay_demo = _load_replay_demo_module()

    demo = {
        "position": np.asarray(
            [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            dtype=np.float32,
        ),
        "orientation": np.asarray(
            [[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0, 0.0]]],
            dtype=np.float32,
        ),
        "gripper": np.asarray([[[0.1]], [[0.2]]], dtype=np.float32),
    }
    demo = replay_demo.normalize_demo_for_batch(demo, batch_size=1, mode="pose")

    policy = replay_demo.ReplayPolicy(demo, mode="pose")
    policy.reset()

    reset_action = policy.apply_first_frame_as_reset()
    next_action = policy.act()

    np.testing.assert_allclose(reset_action["position"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(reset_action["gripper"], [0.1])
    np.testing.assert_allclose(next_action["position"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(next_action["gripper"], [0.2])
    assert policy.remaining_steps == 0


def test_apply_first_frame_reset_uses_kinematic_joint_application():
    replay_demo = _load_replay_demo_module()

    class DummyEnv:
        batch_size = 1

        def __init__(self) -> None:
            self.calls: list[tuple[str, np.ndarray, bool]] = []

        def apply_joint_action(
            self,
            operator,
            action,
            env_mask=None,
            kinematic: bool = False,
        ) -> None:
            self.calls.append(
                (operator, np.asarray(action, dtype=np.float64), kinematic)
            )

    class DummyBackend:
        def __init__(self, env) -> None:
            self.env = env

    class DummyContext:
        def __init__(self, env) -> None:
            self.backend = DummyBackend(env)

    class DummyEvaluator:
        def __init__(self, env) -> None:
            import threading

            self._context = DummyContext(env)
            self.sim_lock = threading.Lock()

    demo = {"joint": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    policy = replay_demo.ReplayPolicy(demo, mode="joint")
    evaluator = DummyEvaluator(DummyEnv())

    reset_action = replay_demo.apply_first_frame_reset(evaluator, policy)

    assert reset_action is not None
    assert len(evaluator._context.backend.env.calls) == 1
    operator, action, kinematic = evaluator._context.backend.env.calls[0]
    assert operator == "arm"
    np.testing.assert_allclose(action, [1.0, 2.0])
    assert kinematic is True


def test_observation_getter_keeps_frames_for_each_batch_env():
    replay_demo = _load_replay_demo_module()

    class DummyEnv:
        def __init__(self) -> None:
            self.calls = 0

        def capture_observation(self) -> dict:
            base = self.calls * 10
            self.calls += 1
            frames = np.asarray(
                [
                    np.full((2, 3, 3), base + 1, dtype=np.uint8),
                    np.full((2, 3, 3), base + 2, dtype=np.uint8),
                ]
            )
            return {"env2_cam/color/image_raw": {"data": frames, "t": 0.0}}

    class DummyBackend:
        def __init__(self, env) -> None:
            self.env = env

    class DummyContext:
        def __init__(self, env) -> None:
            self.backend = DummyBackend(env)

    frames_by_camera_env = {"env2_cam": defaultdict(list)}
    observation_getter = replay_demo.make_observation_getter(
        frames_by_camera_env, ["env2_cam"]
    )

    env = DummyEnv()
    observation_getter(DummyContext(env))
    observation_getter(DummyContext(env))

    assert sorted(frames_by_camera_env["env2_cam"]) == [0, 1]
    assert len(frames_by_camera_env["env2_cam"][0]) == 2
    assert len(frames_by_camera_env["env2_cam"][1]) == 2
    np.testing.assert_array_equal(
        frames_by_camera_env["env2_cam"][0][0], np.full((2, 3, 3), 1, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        frames_by_camera_env["env2_cam"][1][0], np.full((2, 3, 3), 2, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        frames_by_camera_env["env2_cam"][0][1], np.full((2, 3, 3), 11, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        frames_by_camera_env["env2_cam"][1][1], np.full((2, 3, 3), 12, dtype=np.uint8)
    )
