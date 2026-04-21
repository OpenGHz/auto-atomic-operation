from __future__ import annotations

import numpy as np

from auto_atom.runner.data_replay import (
    ReplayPolicy,
    _align_samples_to_times,
    _align_optional_scene_joint,
    _extract_joint_state_positions,
    _extract_pose_stamped_xyzw,
    _make_replay_action_applier,
    normalize_demo_for_batch,
)


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
            operator: str,
            position,
            orientation,
            env_mask=None,
        ) -> None:
            self.calls.append(
                (
                    "base",
                    (
                        operator,
                        tuple(np.asarray(position, dtype=np.float64)),
                        tuple(np.asarray(orientation, dtype=np.float64)),
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


def test_optional_scene_joint_alignment_skips_missing_topic(capsys) -> None:
    aligned = _align_optional_scene_joint(
        [],
        [],
        np.asarray([10, 20, 30], dtype=np.int64),
        scene_joint_topic="/scene/door/joint_states",
        mcap_path="/tmp/demo.mcap",
    )

    assert aligned is None
    captured = capsys.readouterr()
    assert "skipping scene joint replay" in captured.out
    assert "/scene/door/joint_states" in captured.out


def test_replay_action_applier_writes_scene_joints() -> None:
    class DummyModel:
        jnt_qposadr = np.asarray([1, 4], dtype=np.int32)
        jnt_dofadr = np.asarray([2, 5], dtype=np.int32)

    class DummyData:
        def __init__(self) -> None:
            self.qpos = np.zeros(8, dtype=np.float64)
            self.qvel = np.ones(8, dtype=np.float64)

    class DummySingleEnv:
        def __init__(self) -> None:
            self.model = DummyModel()
            self.data = DummyData()

    class DummyEnv:
        batch_size = 1

        def __init__(self) -> None:
            self.envs = [DummySingleEnv()]

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

    class DummyContext:
        def __init__(self, env: DummyEnv) -> None:
            self.backend = DummyBackend(env)

    env = DummyEnv()
    applier = _make_replay_action_applier(kinematic=True)

    import mujoco

    original_name2id = mujoco.mj_name2id
    original_forward = mujoco.mj_forward

    mapping = {"handle_hinge": 0, "door_hinge": 1}

    def fake_name2id(model, obj_type, name):  # noqa: ANN001
        return mapping.get(name, -1)

    def fake_forward(model, data):  # noqa: ANN001
        return None

    mujoco.mj_name2id = fake_name2id
    mujoco.mj_forward = fake_forward
    try:
        applier(
            DummyContext(env),
            {
                "scene_joint_names": ["handle_hinge", "door_hinge"],
                "scene_joint_positions": np.asarray([0.45, -0.25], dtype=np.float32),
                "joint": np.asarray([0.1, 0.2], dtype=np.float32),
            },
        )
    finally:
        mujoco.mj_name2id = original_name2id
        mujoco.mj_forward = original_forward

    np.testing.assert_allclose(env.envs[0].data.qpos[[1, 4]], [0.45, -0.25])
    np.testing.assert_allclose(env.envs[0].data.qvel[[2, 5]], [0.0, 0.0])
