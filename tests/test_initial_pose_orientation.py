from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.framework import InitialPoseConfig
from auto_atom.utils.pose import PoseState, euler_to_quaternion


@dataclass
class DummyEnv:
    batch_size: int = 1


@dataclass
class DummyObjectHandler:
    name: str
    pose: PoseState

    def get_pose(self) -> PoseState:
        return self.pose

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        self.pose = pose.broadcast_to(self.pose.batch_size)


def test_initial_pose_euler_orientation_uses_roll_pitch_yaw_order() -> None:
    handler = DummyObjectHandler(
        name="door",
        pose=PoseState(
            position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={},
        object_handlers={"door": handler},
        initial_poses={
            "door": InitialPoseConfig(
                orientation=[0.4, -0.2, 1.0],
            )
        },
    )

    backend._apply_initial_poses()

    expected = np.asarray(euler_to_quaternion((0.4, -0.2, 1.0)))
    actual = handler.get_pose().orientation[0]
    assert np.allclose(actual, expected)


def test_camera_initial_pose_euler_orientation_uses_roll_pitch_yaw_order() -> None:
    stored_pose = PoseState(
        position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={},
        object_handlers={},
        camera_initial_poses={
            "cam0": InitialPoseConfig(
                orientation=[0.4, -0.2, 1.0],
            )
        },
    )

    def _get_camera_pose(cam_name: str) -> PoseState:
        assert cam_name == "cam0"
        return stored_pose

    def _set_camera_pose(cam_name: str, pose: PoseState, env_mask: np.ndarray) -> None:
        nonlocal stored_pose
        assert cam_name == "cam0"
        assert bool(env_mask[0])
        stored_pose = pose.broadcast_to(1)

    backend._get_camera_pose = _get_camera_pose  # type: ignore[method-assign]
    backend._set_camera_pose = _set_camera_pose  # type: ignore[method-assign]

    backend._apply_camera_initial_poses()

    expected = np.asarray(euler_to_quaternion((0.4, -0.2, 1.0)))
    actual = stored_pose.orientation[0]
    assert np.allclose(actual, expected)
