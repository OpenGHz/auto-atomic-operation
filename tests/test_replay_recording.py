from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auto_atom.runner.replay_recording import (
    NpzRecordingAdapter,
    ReplayTimeline,
    ReplayTrajectory,
    load_mcap_recording,
    prepare_joint_trajectory,
)
from auto_atom.runner.data_replay import (
    DataReplayConfig,
    DataReplayRunner,
    DataReplayTaskFileConfig,
)
from auto_atom.runtime import ComponentRegistry


def _joint_trajectory(*, frames: int = 5, batch: int | None = None) -> ReplayTrajectory:
    shape = (frames, 2) if batch is None else (frames, batch, 2)
    return ReplayTrajectory(
        mode="joint",
        arrays={"joint": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)},
        joint_names=("j0", "j1"),
        timestamps_ns=np.arange(frames, dtype=np.int64) * 10,
    )


def test_trajectory_validates_channels_and_time() -> None:
    with pytest.raises(ValueError, match="share frame count"):
        ReplayTrajectory(
            mode="pose",
            arrays={
                "position": np.zeros((3, 3)),
                "orientation": np.zeros((2, 4)),
            },
        )

    with pytest.raises(ValueError, match="zero-anchored"):
        ReplayTrajectory(
            mode="joint",
            arrays={"joint": np.zeros((2, 2))},
            timestamps_ns=np.asarray([1, 2]),
        )


def test_batch_normalization_prefers_views_and_squeezes_one() -> None:
    trajectory = _joint_trajectory(batch=3)
    normalized = trajectory.normalize_for_batch(2)

    assert normalized.arrays["joint"].shape == (5, 2, 2)
    assert np.shares_memory(normalized.arrays["joint"], trajectory.arrays["joint"])

    squeezed = trajectory.normalize_for_batch(1)
    assert squeezed.arrays["joint"].shape == (5, 2)
    assert np.shares_memory(squeezed.arrays["joint"], trajectory.arrays["joint"])


def test_subsample_keeps_final_frame_and_timestamps() -> None:
    trajectory = _joint_trajectory(frames=6)
    sampled = trajectory.subsample(4)

    np.testing.assert_array_equal(sampled.arrays["joint"][:, 0], [0, 8, 10])
    np.testing.assert_array_equal(sampled.timestamps_ns, [0, 40, 50])
    assert sampled.joint_names == trajectory.joint_names


def test_timeline_is_lazy_at_action_boundary() -> None:
    trajectory = _joint_trajectory(frames=2)
    timeline = ReplayTimeline(trajectory)

    assert timeline.remaining_steps == 2
    assert timeline.current_log_time_ns() is None
    first = timeline.act()
    assert np.shares_memory(first["joint"], trajectory.arrays["joint"])
    assert timeline.current_log_time_ns() == 0
    assert timeline.remaining_steps == 1
    reset = timeline.apply_first_frame_as_reset()
    assert np.shares_memory(reset["joint"], trajectory.arrays["joint"])


def test_npz_adapter_reads_selected_mode_and_optional_base(tmp_path) -> None:
    keys = np.asarray(
        [
            "action/arm/pose/position",
            "action/arm/pose/orientation",
            "action/arm/base_pose/position",
            "action/arm/base_pose/orientation",
            "unused/image/features",
        ]
    )
    arrays = {
        "low_dim_keys": keys,
        "low_dim_data__0": np.zeros((3, 1, 3), dtype=np.float32),
        "low_dim_data__1": np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (3, 1, 1)
        ),
        "low_dim_data__2": np.ones((3, 1, 3), dtype=np.float32),
        "low_dim_data__3": np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (3, 1, 1)
        ),
        "low_dim_data__4": np.ones((3, 1, 1024), dtype=np.float32),
    }
    path = tmp_path / "demo.npz"
    np.savez(path, **arrays)

    trajectory = NpzRecordingAdapter().load(path, "pose")

    assert set(trajectory.arrays) == {
        "position",
        "orientation",
        "base_position",
        "base_orientation",
    }
    assert trajectory.frame_count == 3
    assert trajectory.timestamps_ns is None


def test_joint_preparation_uses_owned_array_in_place() -> None:
    trajectory = ReplayTrajectory(
        mode="joint",
        arrays={"joint": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        joint_names=("raw_a", "raw_b"),
    )
    source = trajectory.arrays["joint"]

    prepared = prepare_joint_trajectory(
        trajectory,
        ("a", "b"),
        joint_name_mapping={"raw_a": "a", "raw_b": "b"},
        joint_axis_scale=[-1.0, 1.0],
        joint_clip={"a": {"min": -2.0, "max": -1.5}},
    )

    assert prepared is trajectory
    assert prepared.joint_names == ("a", "b")
    np.testing.assert_allclose(prepared.arrays["joint"], [[-1.5, 2.0], [-2.0, 4.0]])
    assert not np.shares_memory(source, prepared.arrays["joint"])


@pytest.mark.parametrize(
    ("path", "arm_topic", "gripper_topic", "expected_frames"),
    [
        ("data/replay/george.mcap", None, None, 1813),
        (
            "data/replay_old/open-door.mcap",
            "/robot/right_arm/joint_state",
            "/robot/right_gripper/joint_state",
            162,
        ),
    ],
)
def test_mcap_source_adapters_share_canonical_shape(
    path: str,
    arm_topic: str | None,
    gripper_topic: str | None,
    expected_frames: int,
) -> None:
    if not Path(path).exists():
        pytest.skip(f"recording fixture is not present: {path}")
    trajectory = load_mcap_recording(path, arm_topic, gripper_topic)

    assert trajectory.mode == "joint"
    assert trajectory.frame_count == expected_frames
    assert trajectory.timestamps_ns is not None
    assert trajectory.timestamps_ns[0] == 0


def test_data_replay_runner_consumes_canonical_npz_timeline(tmp_path) -> None:
    path = tmp_path / "demo.npz"
    np.savez(
        path,
        low_dim_keys=np.asarray(
            [
                "action/arm/pose/position",
                "action/arm/pose/orientation",
            ]
        ),
        low_dim_data__0=np.zeros((2, 1, 3), dtype=np.float32),
        low_dim_data__1=np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1, 1)
        ),
    )
    ComponentRegistry.register_env("replay_recording_runner", {"batch_size": 1})
    config = DataReplayTaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {
                "env_name": "replay_recording_runner",
                "stages": [],
            },
            "task_operators": {"arm": {}},
            "replay": {
                "demo_dir": str(tmp_path),
                "demo_name": "demo",
                "mode": "pose",
            },
        }
    )
    runner = DataReplayRunner().from_config(config)
    try:
        assert isinstance(runner._policy, ReplayTimeline)
        assert runner.remaining_steps == 2
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_untimestamped_replay_resets_supported_simulation_clock() -> None:
    class ClockEnv:
        batch_size = 2

        def __init__(self) -> None:
            self.calls: list[tuple[float, object]] = []

        def set_simulation_time(
            self,
            time_sec: float,
            env_mask: np.ndarray | None = None,
        ) -> None:
            self.calls.append((time_sec, env_mask))

    class Evaluator:
        def __init__(self, env: ClockEnv) -> None:
            self.env = env

        def reset(self, env_mask: np.ndarray | None = None) -> object:
            return object()

        def get_env(self) -> ClockEnv:
            return self.env

    trajectory = ReplayTrajectory(
        mode="joint",
        arrays={"joint": np.zeros((2, 2), dtype=np.float32)},
    )
    env = ClockEnv()
    runner = DataReplayRunner()
    runner._evaluator = Evaluator(env)
    runner._policy = ReplayTimeline(trajectory)
    runner._replay_cfg = DataReplayConfig(reset_from_first_frame=False)
    env_mask = np.asarray([False, True], dtype=bool)

    runner.reset(env_mask)

    assert len(env.calls) == 1
    assert env.calls[0][0] == 0.0
    np.testing.assert_array_equal(env.calls[0][1], env_mask)
