from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.framework import OperatorInitialState, PoseOverrideConfig, PoseReference
from auto_atom.utils.pose import (
    PoseState,
    compose_pose,
    euler_to_quaternion,
    resolve_pose_override,
)


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
            "door": PoseOverrideConfig(
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
            "cam0": PoseOverrideConfig(
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


def test_pose_override_accepts_named_reference_and_composes_partial_pose() -> None:
    config = PoseOverrideConfig(
        reference="anchor",
        position=[0.2, 0.0, 0.0],
    )
    assert config.reference == "anchor"

    anchor = PoseState(
        position=[1.0, 2.0, 3.0],
        orientation=euler_to_quaternion((0.0, 0.0, 0.5)),
    )
    fallback = PoseState(position=[9.0, 9.0, 9.0], orientation=[0.0, 0.0, 0.0, 1.0])
    resolved = resolve_pose_override(config, fallback, anchor)
    expected = compose_pose(
        anchor,
        PoseState(
            position=[0.2, 0.0, 0.0],
            # The omitted orientation preserves the fallback orientation in the
            # anchor frame, which is the identity in this example.
            orientation=euler_to_quaternion((0.0, 0.0, -0.5)),
        ),
    )
    np.testing.assert_allclose(resolved.position, expected.position)
    np.testing.assert_allclose(resolved.orientation, expected.orientation, atol=1e-12)


def test_pose_override_builtin_reference_is_typed() -> None:
    config = PoseOverrideConfig(reference="world")
    assert config.reference == PoseReference.WORLD


def test_legacy_eef_tuple_keeps_historical_yaw_pitch_roll_order() -> None:
    fallback = PoseState(position=[9.0, 9.0, 9.0], orientation=[0.0, 0.0, 0.0, 1.0])
    config = OperatorInitialState.model_validate(
        {"eef_pose": (1.0, 2.0, 3.0, 0.7, -0.2, 0.4)}
    )

    resolved = resolve_pose_override(config.eef_pose, fallback)  # type: ignore[arg-type]

    np.testing.assert_allclose(resolved.position[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        resolved.orientation[0],
        euler_to_quaternion((0.4, -0.2, 0.7)),
    )


def test_camera_pose_round_trip_uses_world_coordinates_for_mounted_camera() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="mount" pos="1 2 0" quat="0.7071067812 0 0 0.7071067812">
              <camera name="wrist_cam" pos="1 0 0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data)
    batch_env = SimpleNamespace(batch_size=1, envs=[env])
    backend = MujocoTaskBackend(
        env=batch_env,
        operator_handlers={},
        object_handlers={},
    )

    initial = backend._get_camera_pose("wrist_cam")
    np.testing.assert_allclose(initial.position[0], [1.0, 3.0, 0.0], atol=1e-6)

    target = PoseState(position=[4.0, 5.0, 6.0], orientation=[0.0, 0.0, 0.0, 1.0])
    backend._set_camera_pose("wrist_cam", target, np.asarray([True]))
    actual = backend._get_camera_pose("wrist_cam")
    np.testing.assert_allclose(actual.position, target.position, atol=1e-6)
    np.testing.assert_allclose(actual.orientation, target.orientation, atol=1e-6)


def test_empty_operator_initial_state_does_not_trigger_an_extra_home() -> None:
    class Handler:
        def __init__(self) -> None:
            self.home_calls = 0

        def home(self, _mask=None) -> None:
            self.home_calls += 1

    handler = Handler()
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={"arm": handler},  # type: ignore[arg-type]
        object_handlers={},
        operator_initial_states={"arm": OperatorInitialState()},
    )

    backend.apply_operator_initial_states(home=True)

    assert handler.home_calls == 0


def test_pose_override_and_legacy_eef_shapes_are_validated_at_config_boundary() -> None:
    with pytest.raises(ValueError, match="position"):
        PoseOverrideConfig(position=[0.0, 1.0])
    with pytest.raises(ValueError, match="orientation"):
        PoseOverrideConfig(orientation=[0.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="eef_pose"):
        OperatorInitialState(eef_pose=[0.0, 1.0, 2.0, 3.0, 4.0])
