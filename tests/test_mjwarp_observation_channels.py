"""MJWarp operator observation channels match the native env's.

Both backends run the same task with the same demo policy, without reset
randomization, and report the same keys with matching values: joint state,
the EEF pose in base frame, and the ``action/...`` command channels the
episode stream records as actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom import ComponentRegistry, load_task_file_hydra  # noqa: E402
from auto_atom.data import stream  # noqa: E402
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator  # noqa: E402
from auto_atom.utils.transformations import euler_matrix  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "aao_configs"
TICKS = 40

# (task, overrides): a mocap operator; a joint-mode arm with per-step IK and a
# finger-distance EEF mapper; the same arm with solve-once interpolation.
CASES = [
    ("pick_and_place", []),
    ("open_door", ["embodiment=airbot_play_g2p"]),
    (
        "open_door",
        [
            "embodiment=airbot_play_g2p",
            "++task_operators.arm.ik.joint_control_mode=solve_once_interpolate",
            "++task_operators.arm.ik.joint_interp_speed=0.01",
        ],
    ),
]
CASE_IDS = ["mocap", "per_step_ik", "solve_once_interpolate"]

Frames = List[Dict[str, np.ndarray]]


def _rollout(task: str, overrides: List[str], *, warp: bool) -> Tuple[Frames, object]:
    """Operator channels after the reset and after each of TICKS demo ticks."""
    ComponentRegistry.clear()
    task_file = load_task_file_hydra(
        task,
        config_dir=CONFIG_DIR,
        overrides=[
            "env.viewer=null",
            "++task.seed=1",
            "++env.batch_size=1",
            "~task.randomization",
            *overrides,
            *(["backend=warp"] if warp else []),
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        task_file
    )
    try:
        env = evaluator.get_env()

        def channels() -> Dict[str, np.ndarray]:
            observation = env.capture_observation()
            return {
                key: np.asarray(payload["data"], dtype=np.float64)
                for key, payload in observation.items()
                if "/joint_state/" in key or "/pose/" in key
            }

        update = evaluator.reset()
        frames = [channels()]
        for _ in range(TICKS):
            update = evaluator.update(policy.act({}, update, evaluator))
            frames.append(channels())
        commands = env.capture_commands()
        return frames, commands
    finally:
        evaluator.close()
        ComponentRegistry.clear()


def _difference(key: str, left: np.ndarray, right: np.ndarray) -> float:
    if key.endswith("/pose/rotation"):
        # Euler angles are not unique (pi and -pi, and roll/yaw at a gimbal
        # lock), so compare the rotations they describe.
        left, right = (
            np.stack([euler_matrix(*angles)[:3, :3] for angles in values])
            for values in (left, right)
        )
    return float(np.max(np.abs(left - right)))


@pytest.mark.parametrize(("task", "overrides"), CASES, ids=CASE_IDS)
def test_warp_reports_the_native_operator_channels(
    task: str, overrides: List[str]
) -> None:
    native, _ = _rollout(task, overrides, warp=False)
    warp, warp_commands = _rollout(task, overrides, warp=True)

    assert set(warp[0]) == set(native[0])
    assert any(key.startswith("action/") for key in warp[0])
    # The same model and initial state: the reset frames agree to float error.
    for key in native[0]:
        assert _difference(key, native[0][key], warp[0][key]) < 1e-5, key
    # Commands, arm joint positions and EEF poses track each other while the
    # two solvers' physics drifts apart slowly. Contact forces (effort) and
    # velocities are compared only at the reset. The gripper's measured
    # opening differs more once the fingers touch an object: MJWarp has no
    # noslip solver, so the contact settles differently.
    for key in native[0]:
        if key.endswith("/effort") or key.endswith("/velocity"):
            continue
        bound = 5e-3 if key.startswith("gripper/") else 1e-3
        worst = max(_difference(key, n[key], w[key]) for n, w in zip(native, warp))
        assert worst < bound, (key, worst)
    # capture_commands() is exactly the action/ subset, unrendered.
    assert set(warp_commands) == {key for key in warp[-1] if key.startswith("action/")}
    for key, payload in warp_commands.items():
        np.testing.assert_array_equal(payload["data"], warp[-1][key])


def test_a_reset_arm_commands_the_pose_it_holds() -> None:
    frames, _ = _rollout("open_door", ["embodiment=airbot_play_g2p"], warp=True)

    reset = frames[0]
    np.testing.assert_allclose(
        reset["action/arm/pose/position"], reset["arm/pose/position"], atol=1e-9
    )
    np.testing.assert_allclose(
        reset["action/arm/joint_state/position"],
        reset["arm/joint_state/position"],
        atol=1e-9,
    )


def test_a_warp_stream_records_actions() -> None:
    with stream(
        "open_door",
        base_seed=0,
        config_dir=str(CONFIG_DIR),
        overrides=["embodiment=airbot_play_g2p", "env.viewer=null", "backend=warp"],
        num_episodes=1,
        max_updates=10,
        on_invalid="keep",
    ) as episodes:
        (episode,) = list(episodes)

    actions = episode.transitions.action
    assert len(episode) == 10
    assert {
        "action/arm/joint_state/position",
        "action/arm/pose/position",
        "action/arm/pose/orientation",
        "action/gripper/joint_state/position",
    } <= set(actions)
    assert actions["action/arm/pose/position"].shape == (10, 3)
    assert episode.transitions.sim_time.dtype == np.float64
    assert np.all(np.diff(episode.transitions.sim_time) > 0)
