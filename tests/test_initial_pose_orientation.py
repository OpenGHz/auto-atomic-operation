from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import mujoco
import numpy as np
import pytest

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.framework import (
    OperatorInitialState,
    PoseAxisConfig,
    PoseOrientationConfig,
    PoseOverrideConfig,
    PosePositionConfig,
    PoseReference,
)
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


def test_pose_override_supports_axis_level_references_with_global_fallback() -> None:
    config = PoseOverrideConfig.model_validate(
        {
            "reference": "anchor",
            "position": {
                "x": 0.2,
                "y": {"value": 0.3, "reference": "world"},
                "z": {"value": 0.4},
            },
            "orientation": {
                "roll": 0.1,
                "yaw": {"value": 0.2, "reference": "world"},
            },
        }
    )

    assert isinstance(config.position, PosePositionConfig)
    assert isinstance(config.orientation, PoseOrientationConfig)
    assert config.axis_references() == ("anchor", PoseReference.WORLD)
    assert config.position.x == 0.2
    assert isinstance(config.position.y, PoseAxisConfig)
    assert config.position.y.reference == PoseReference.WORLD
    assert config.position.z.reference is None


def test_pose_override_supports_component_level_references_before_axis_overrides() -> (
    None
):
    config = PoseOverrideConfig.model_validate(
        {
            "reference": "anchor",
            "position": {
                "reference": "world",
                "x": 0.2,
                "y": {"value": 0.3, "reference": "other"},
            },
            "orientation": {
                "reference": "world",
                "roll": 0.1,
                "pitch": {"value": 0.2, "reference": "anchor"},
            },
        }
    )

    assert config.position.reference == "world"
    assert config.orientation.reference == "world"
    assert config.axis_references() == (
        "anchor",
        PoseReference.WORLD,
        "other",
    )


def test_pose_override_component_reference_resolves_all_scalar_axes() -> None:
    config = PoseOverrideConfig.model_validate(
        {
            "reference": "anchor",
            "position": {
                "reference": "world",
                "x": 0.2,
                "y": 0.3,
                "z": 0.4,
            },
            "orientation": {
                "reference": "world",
                "roll": 0.1,
                "pitch": -0.2,
                "yaw": 0.3,
            },
        }
    )
    anchor = PoseState(
        position=[10.0, 20.0, 30.0],
        orientation=euler_to_quaternion((0.0, 0.0, 0.5)),
    )
    world = PoseState()
    fallback = PoseState(position=[9.0, 9.0, 9.0])

    resolved = resolve_pose_override(
        config,
        fallback,
        anchor,
        {"anchor": anchor, PoseReference.WORLD: world},
    )

    np.testing.assert_allclose(resolved.position[0], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(
        resolved.orientation[0],
        euler_to_quaternion((0.1, -0.2, 0.3)),
        atol=1e-12,
    )


def test_pose_override_component_reference_applies_one_rigid_transform() -> None:
    config = PoseOverrideConfig.model_validate(
        {
            "reference": "world",
            "position": {
                "reference": "anchor",
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
            },
            "orientation": {
                "reference": "anchor",
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        }
    )
    anchor = PoseState(
        position=[1.0, 2.0, 3.0],
        orientation=euler_to_quaternion((0.0, 0.0, np.pi / 2.0)),
    )

    resolved = resolve_pose_override(
        config,
        PoseState(),
        PoseState(),
        {PoseReference.WORLD: PoseState(), "anchor": anchor},
    )

    np.testing.assert_allclose(resolved.position[0], [1.0, 3.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(
        resolved.orientation[0],
        anchor.orientation[0],
        atol=1e-12,
    )


def test_pose_override_axis_reference_overrides_component_reference() -> None:
    config = PoseOverrideConfig.model_validate(
        {
            "reference": "anchor",
            "position": {
                "reference": "world",
                "x": 0.2,
                "y": {"value": 0.3, "reference": "other"},
                "z": 0.4,
            },
        }
    )
    world = PoseState()
    other = PoseState(position=[0.0, 10.0, 0.0])

    resolved = resolve_pose_override(
        config,
        PoseState(),
        PoseState(),
        {
            "anchor": PoseState(),
            PoseReference.WORLD: world,
            "other": other,
        },
    )

    np.testing.assert_allclose(resolved.position[0], [0.2, 10.3, 0.4], atol=1e-12)


def test_pose_override_axis_world_reference_overrides_only_selected_components() -> (
    None
):
    config = PoseOverrideConfig.model_validate(
        {
            "reference": "anchor",
            "position": {
                "x": 0.2,
                "y": {"value": 3.0, "reference": "world"},
            },
        }
    )
    anchor = PoseState(
        position=[10.0, 20.0, 30.0],
        orientation=euler_to_quaternion((0.0, 0.0, 0.0)),
    )
    world = PoseState()
    fallback = PoseState(position=[9.0, 9.0, 9.0], orientation=[0.0, 0.0, 0.0, 1.0])
    resolved = resolve_pose_override(
        config,
        fallback,
        anchor,
        {"anchor": anchor, PoseReference.WORLD: world},
    )

    np.testing.assert_allclose(resolved.position[0], [10.2, 3.0, 9.0])


def test_backend_initial_pose_resolves_axis_references_independently() -> None:
    handler = DummyObjectHandler(
        name="door",
        pose=PoseState(
            position=np.asarray([[9.0, 9.0, 9.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={},
        object_handlers={"door": handler},
        initial_poses={
            "door": PoseOverrideConfig.model_validate(
                {
                    "reference": "anchor",
                    "position": {
                        "x": 0.2,
                        "y": -0.3,
                        "z": {"value": -0.1, "reference": "world"},
                    },
                }
            )
        },
    )
    reference_poses = {
        "anchor": PoseState(position=[10.0, 20.0, 30.0]),
        PoseReference.WORLD: PoseState(),
    }
    backend._resolve_initial_reference_pose = (  # type: ignore[method-assign]
        lambda reference, _env_index, **_kwargs: reference_poses[reference]
    )

    backend._apply_initial_poses()

    np.testing.assert_allclose(handler.get_pose().position[0], [10.2, 19.7, -0.1])


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
