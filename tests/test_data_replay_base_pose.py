from __future__ import annotations

import logging

import numpy as np

from auto_atom.runner.data_replay import (
    DataReplayConfig,
    ReplayPolicy,
    TransformResetConfig,
    _align_optional_scene_joint,
    _align_samples_to_times,
    _extract_joint_state_positions,
    _extract_pose_stamped_xyzw,
    _make_replay_action_applier,
    _select_transform_reset_message_index,
    _validate_replay_environment_capabilities,
    normalize_demo_for_batch,
)
from auto_atom.runner.replay_recording import ReplayTrajectory


def test_replay_policy_includes_optional_base_pose_channels() -> None:
    demo = {
        "position": np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32),
        "orientation": np.asarray([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
        "base_position": np.asarray([[[0.5, -0.2, 0.1]]], dtype=np.float32),
        "base_orientation": np.asarray([[[0.0, 0.0, 0.2, 0.98]]], dtype=np.float32),
    }
    demo = normalize_demo_for_batch(demo, batch_size=1, mode="pose")

    policy = ReplayPolicy(demo, mode="pose")
    action = policy.act()

    np.testing.assert_allclose(action["base_position"], [0.5, -0.2, 0.1])
    np.testing.assert_allclose(action["base_orientation"], [0.0, 0.0, 0.2, 0.98])


def test_replay_action_applier_moves_base_before_pose_action() -> None:
    class DummyEnv:
        batch_size = 1

        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple]] = []

        def set_operator_base_pose(
            self,
            op_name: str,
            pos_w,
            quat_w,
            env_mask=None,
        ) -> None:
            self.calls.append(
                (
                    "base",
                    (
                        op_name,
                        tuple(np.asarray(pos_w, dtype=np.float64)),
                        tuple(np.asarray(quat_w, dtype=np.float64)),
                        env_mask,
                    ),
                )
            )

        def apply_pose_action(
            self,
            operator: str,
            position,
            orientation,
            gripper=None,
            env_mask=None,
            kinematic: bool = False,
        ) -> None:
            self.calls.append(
                (
                    "pose",
                    (
                        operator,
                        tuple(np.asarray(position, dtype=np.float64)),
                        tuple(np.asarray(orientation, dtype=np.float64)),
                        gripper,
                        env_mask,
                        kinematic,
                    ),
                )
            )

    class DummyBackend:
        def __init__(self, env: DummyEnv) -> None:
            self.env = env

        def get_env(self) -> DummyEnv:
            return self.env

    class DummyContext:
        def __init__(self, env: DummyEnv) -> None:
            self.backend = DummyBackend(env)

    env = DummyEnv()
    applier = _make_replay_action_applier(kinematic=True)
    applier(
        DummyContext(env),
        {
            "base_position": np.asarray([0.5, -0.2, 0.1], dtype=np.float32),
            "base_orientation": np.asarray([0.0, 0.0, 0.2, 0.98], dtype=np.float32),
            "position": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            "orientation": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        },
    )

    assert [name for name, _ in env.calls] == ["base", "pose"]
    assert env.calls[0][1][0] == "arm"
    np.testing.assert_allclose(env.calls[0][1][1], [0.5, -0.2, 0.1])
    assert env.calls[1][1][0] == "arm"
    assert env.calls[1][1][-1] is True


def test_pose_stamped_base_topic_helpers_extract_and_align() -> None:
    msg = {
        "header": {"frame_id": "world"},
        "pose": {
            "position": {"x": 0.5, "y": -0.2, "z": 0.1},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.2, "w": 0.98},
        },
    }

    position, orientation = _extract_pose_stamped_xyzw(msg, topic="/robot/base_pose")

    np.testing.assert_allclose(position, [0.5, -0.2, 0.1])
    np.testing.assert_allclose(orientation, [0.0, 0.0, 0.2, 0.98])

    samples = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    aligned = _align_samples_to_times(
        samples,
        sample_times=np.asarray([10, 20, 40], dtype=np.int64),
        target_times=np.asarray([9, 18, 35, 41], dtype=np.int64),
        label="base",
    )

    np.testing.assert_allclose(
        aligned,
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
    )


def test_replay_policy_includes_optional_scene_joint_channels() -> None:
    demo = {
        "joint": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "scene_joint": np.asarray([[0.45, -0.2], [0.55, -0.1]], dtype=np.float32),
        "scene_joint_names": ["handle_hinge", "door_hinge"],
    }
    demo = normalize_demo_for_batch(demo, batch_size=1, mode="joint")

    policy = ReplayPolicy(demo, mode="joint")
    action = policy.act()

    np.testing.assert_allclose(action["joint"], [0.1, 0.2])
    np.testing.assert_allclose(action["scene_joint_positions"], [0.45, -0.2])
    assert action["scene_joint_names"] == ["handle_hinge", "door_hinge"]


def test_joint_state_scene_topic_helpers_extract_reordered_positions() -> None:
    msg = {
        "name": ["door_hinge", "handle_hinge"],
        "position": [-0.25, 0.45],
    }

    names, positions = _extract_joint_state_positions(
        msg,
        topic="/scene/door/joint_states",
        expected_names=["handle_hinge", "door_hinge"],
    )

    assert names == ["handle_hinge", "door_hinge"]
    np.testing.assert_allclose(positions, [0.45, -0.25])


def test_optional_scene_joint_alignment_skips_missing_topic(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="auto_atom.runner.data_replay")

    aligned = _align_optional_scene_joint(
        [],
        [],
        np.asarray([10, 20, 30], dtype=np.int64),
        scene_joint_topic="/scene/door/joint_states",
        mcap_path="/tmp/demo.mcap",
    )

    assert aligned is None
    assert "skipping scene joint replay" in caplog.text
    assert "/scene/door/joint_states" in caplog.text


def test_replay_action_applier_writes_scene_joints() -> None:
    class DummyEnv:
        batch_size = 1

        def __init__(self) -> None:
            self.scene_calls: list[tuple[list[str], np.ndarray, object]] = []

        def set_scene_joint_positions(
            self,
            joint_names,
            positions,
            env_mask=None,
        ) -> None:
            self.scene_calls.append(
                (
                    list(joint_names),
                    np.asarray(positions, dtype=np.float64).copy(),
                    env_mask,
                )
            )

        def apply_joint_action(
            self,
            operator,
            action,
            env_mask=None,
            kinematic: bool = False,
        ) -> None:
            return None

    class DummyBackend:
        def __init__(self, env: DummyEnv) -> None:
            self.env = env

        def get_env(self) -> DummyEnv:
            return self.env

    class DummyContext:
        def __init__(self, env: DummyEnv) -> None:
            self.backend = DummyBackend(env)

    env = DummyEnv()
    applier = _make_replay_action_applier(kinematic=True)
    applier(
        DummyContext(env),
        {
            "scene_joint_names": ["handle_hinge", "door_hinge"],
            "scene_joint_positions": np.asarray([0.45, -0.25], dtype=np.float32),
            "joint": np.asarray([0.1, 0.2], dtype=np.float32),
        },
    )

    assert len(env.scene_calls) == 1
    assert env.scene_calls[0][0] == ["handle_hinge", "door_hinge"]
    np.testing.assert_allclose(env.scene_calls[0][1], [[0.45, -0.25]])


def test_replay_capability_validation_rejects_scene_joint_env_early() -> None:
    trajectory = ReplayTrajectory(
        mode="joint",
        arrays={
            "joint": np.zeros((1, 2), dtype=np.float32),
            "scene_joint": np.zeros((1, 1), dtype=np.float32),
        },
        scene_joint_names=("door_hinge",),
    )

    with np.testing.assert_raises_regex(
        RuntimeError,
        "scene joint actions.*set_scene_joint_positions",
    ):
        _validate_replay_environment_capabilities(
            object(),
            DataReplayConfig(),
            trajectory,
        )


def test_replay_capability_validation_rejects_timestamp_env_early() -> None:
    trajectory = ReplayTrajectory(
        mode="joint",
        arrays={"joint": np.zeros((1, 2), dtype=np.float32)},
        timestamps_ns=np.asarray([0], dtype=np.int64),
    )

    with np.testing.assert_raises_regex(
        RuntimeError,
        "recording timestamps.*set_simulation_time",
    ):
        _validate_replay_environment_capabilities(
            object(),
            DataReplayConfig(),
            trajectory,
        )


def test_replay_capability_validation_rejects_core_action_env_early() -> None:
    trajectory = ReplayTrajectory(
        mode="joint",
        arrays={"joint": np.zeros((1, 2), dtype=np.float32)},
    )
    env = type("ObservationOnlyEnv", (), {"batch_size": 1})()

    with np.testing.assert_raises_regex(
        RuntimeError,
        "joint actions.*apply_joint_action",
    ):
        _validate_replay_environment_capabilities(
            env,
            DataReplayConfig(),
            trajectory,
        )


def test_transform_reset_selector_supports_first_last_and_index() -> None:
    translations = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rotations = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert (
        _select_transform_reset_message_index(
            translations,
            rotations,
            TransformResetConfig.model_validate(
                {
                    "topic": "/tf_static",
                    "parent": {"kind": "site", "name": "p"},
                    "child": {"kind": "site", "name": "c"},
                    "message_selector": "first",
                }
            ),
        )
        == 0
    )
    assert (
        _select_transform_reset_message_index(
            translations,
            rotations,
            TransformResetConfig.model_validate(
                {
                    "topic": "/tf_static",
                    "parent": {"kind": "site", "name": "p"},
                    "child": {"kind": "site", "name": "c"},
                    "message_selector": "last",
                }
            ),
        )
        == 2
    )
    assert (
        _select_transform_reset_message_index(
            translations,
            rotations,
            TransformResetConfig.model_validate(
                {
                    "topic": "/tf_static",
                    "parent": {"kind": "site", "name": "p"},
                    "child": {"kind": "site", "name": "c"},
                    "message_selector": "index",
                    "message_index": 1,
                }
            ),
        )
        == 1
    )


def test_transform_reset_selector_supports_first_jump_and_fallback(caplog) -> None:
    caplog.set_level(logging.INFO, logger="auto_atom.runner.data_replay")

    translations = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.060, 0.0, 0.0],
            [0.061, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rotations = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cfg = TransformResetConfig.model_validate(
        {
            "topic": "/robot/right_arm/base_handle/transform",
            "parent": {"kind": "operator_base", "name": "arm"},
            "child": {"kind": "site", "name": "handle_grasp_front_site"},
            "message_selector": "first_jump",
            "jump_position_threshold": 0.02,
        }
    )

    assert _select_transform_reset_message_index(translations, rotations, cfg) == 2
    assert "selected_index=2 differs from first frame" in caplog.text
    assert "jump_from=1 jump_to=2" in caplog.text
    assert "pos_delta=" in caplog.text
    assert "ori_delta=" in caplog.text

    no_jump_cfg = TransformResetConfig.model_validate(
        {
            "topic": "/robot/right_arm/base_handle/transform",
            "parent": {"kind": "operator_base", "name": "arm"},
            "child": {"kind": "site", "name": "handle_grasp_front_site"},
            "message_selector": "first_jump",
            "jump_position_threshold": 1.0,
            "jump_orientation_threshold": 1.0,
        }
    )
    assert (
        _select_transform_reset_message_index(translations, rotations, no_jump_cfg) == 0
    )
    assert "falling back to first message" in caplog.text
