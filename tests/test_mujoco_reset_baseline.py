"""Regression tests for model- and operator-level reset baselines."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from auto_atom.backend.mjc.mujoco_backend import (
    MujocoOperatorHandler,
    MujocoTaskBackend,
)
from auto_atom.backend.mjc.mujoco_backend import MujocoObjectHandler
from auto_atom.basis.mjc.mujoco_basis import (
    CameraSpec,
    DataType,
    EnvConfig,
    MujocoBasis,
    OperatorBinding,
)
from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv
from auto_atom.framework import PoseOverrideConfig
from auto_atom.utils.pose import PoseState
from auto_atom.scene_composition import SceneConfig


def _write_xml(tmp_path: Path, xml: str) -> Path:
    path = tmp_path / "scene.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def _scene_config(path: Path) -> EnvConfig:
    return EnvConfig(
        scene=SceneConfig(base=path),
        enabled_sensors=set(),
        cameras=[CameraSpec(name="mounted_cam")],
    )


def test_low_level_reset_restores_model_body_and_camera_poses(tmp_path: Path) -> None:
    path = _write_xml(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="asset" pos="1 2 3">
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
            <body name="mount" pos="0.5 0.0 0.0">
              <camera name="mounted_cam" pos="0.2 0.0 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )
    env = MujocoBasis(_scene_config(path))
    try:
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "asset")
        camera_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_CAMERA, "mounted_cam"
        )
        body_pos = env.model.body_pos.copy()
        body_quat = env.model.body_quat.copy()
        cam_pos = env.model.cam_pos.copy()
        cam_quat = env.model.cam_quat.copy()

        env.model.body_pos[body_id] += [4.0, 5.0, 6.0]
        env.model.body_quat[body_id] = [0.70710678, 0.0, 0.70710678, 0.0]
        env.model.cam_pos[camera_id] += [1.0, 2.0, 3.0]
        env.model.cam_quat[camera_id] = [0.70710678, 0.70710678, 0.0, 0.0]
        env.reset()

        np.testing.assert_allclose(env.model.body_pos, body_pos)
        np.testing.assert_allclose(env.model.body_quat, body_quat)
        np.testing.assert_allclose(env.model.cam_pos, cam_pos)
        np.testing.assert_allclose(env.model.cam_quat, cam_quat)
    finally:
        env.close()


class _IdentityIK:
    def solve(self, target_pose_in_base, current_qpos):  # noqa: ANN001
        return np.asarray(current_qpos).copy()


def _identity_ik_factory(**_kwargs) -> _IdentityIK:
    return _IdentityIK()


def test_joint_operator_reset_restores_base_and_home_cache(tmp_path: Path) -> None:
    path = _write_xml(
        tmp_path,
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="robot" pos="1 0 0">
              <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
              <joint name="arm_joint" type="hinge" axis="0 0 1"/>
              <site name="eef" pos="0 0 1"/>
            </body>
          </worldbody>
          <actuator><position name="arm_act" joint="arm_joint"/></actuator>
        </mujoco>
        """,
    )
    config = EnvConfig(
        scene=SceneConfig(base=path),
        operators={
            "arm": OperatorBinding(
                arm_actuators=["arm_act"],
                pose_site="eef",
                root_body="robot",
                ik_factory=_identity_ik_factory,
            )
        },
        enabled_sensors={DataType.JOINT_POSITION},
    )
    env = UnifiedMujocoEnv(config)
    try:
        state = env._operator_states["arm"]
        baseline_base = state.base_pos.copy()
        baseline_tool = state.tool_offset_pos.copy()
        baseline_home = state.home_arm_qpos.copy()

        env.override_operator_base_pose(
            "arm",
            baseline_base + [2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        state.tool_offset_pos += 0.5
        state.home_arm_qpos += 0.25
        state.planned_joint_start_qpos = None
        state.planned_joint_target_qpos = None
        env.reset()

        np.testing.assert_allclose(state.base_pos, baseline_base)
        np.testing.assert_allclose(state.tool_offset_pos, baseline_tool)
        np.testing.assert_allclose(state.home_arm_qpos, baseline_home)
        np.testing.assert_allclose(state.planned_joint_start_qpos, baseline_home)
        np.testing.assert_allclose(state.planned_joint_target_qpos, baseline_home)
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        np.testing.assert_allclose(env.data.xpos[body_id], baseline_base, atol=1e-6)
    finally:
        env.close()


def test_mocap_operator_reset_restores_virtual_base_and_home(tmp_path: Path) -> None:
    path = _write_xml(
        tmp_path,
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="mocap" mocap="true">
              <geom type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0"/>
            </body>
            <body name="robot" pos="0 0 0">
              <freejoint name="robot_free"/>
              <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
              <site name="eef" pos="0 0 1"/>
            </body>
          </worldbody>
          <equality><weld body1="mocap" body2="robot"/></equality>
        </mujoco>
        """,
    )
    config = EnvConfig(
        scene=SceneConfig(base=path),
        operators={
            "arm": OperatorBinding(
                pose_site="eef",
                root_body="robot",
                mocap_body="mocap",
                freejoint="robot_free",
            )
        },
    )
    env = UnifiedMujocoEnv(config)
    try:
        state = env._operator_states["arm"]
        baseline_base = state.base_pos.copy()
        baseline_tool = state.tool_offset_pos.copy()
        baseline_mocap = state.home_mocap_pos.copy()

        env.set_operator_base_pose(
            "arm",
            [2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        state.tool_offset_pos += 0.5
        state.home_mocap_pos += 1.0
        env.reset()

        np.testing.assert_allclose(state.base_pos, baseline_base)
        np.testing.assert_allclose(state.tool_offset_pos, baseline_tool)
        np.testing.assert_allclose(state.home_mocap_pos, baseline_mocap)
    finally:
        env.close()


class _BatchObject:
    """Small object seam used to exercise masked baseline bookkeeping."""

    def __init__(self, pose: PoseState) -> None:
        self.name = "object"
        self.pose = pose

    def get_pose(self) -> PoseState:
        return self.pose

    def set_pose(self, pose: PoseState, env_mask: np.ndarray | None = None) -> None:
        mask = (
            np.ones(self.pose.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool)
        )
        position = self.pose.position.copy()
        orientation = self.pose.orientation.copy()
        position[mask] = pose.position[mask]
        orientation[mask] = pose.orientation[mask]
        self.pose = PoseState(position=position, orientation=orientation)


def test_masked_initial_pose_keeps_unselected_object_baseline() -> None:
    object_handler = _BatchObject(
        PoseState(
            position=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            orientation=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )
    )
    backend = MujocoTaskBackend(
        env=type("BatchEnv", (), {"batch_size": 2})(),
        operator_handlers={},
        object_handlers={"object": object_handler},
        initial_poses={
            "object": PoseOverrideConfig(position=[1.0, 2.0, 3.0]),
        },
    )

    backend._apply_initial_poses()
    baseline = backend._default_object_poses["object"]
    # Simulate an unselected row that was randomized during its prior episode.
    object_handler.pose.position[1] = [9.0, 9.0, 9.0]
    backend._apply_initial_poses(np.asarray([True, False]))

    np.testing.assert_allclose(
        backend._default_object_poses["object"].position[0],
        baseline.position[0],
    )
    np.testing.assert_allclose(
        backend._default_object_poses["object"].position[1],
        baseline.position[1],
    )


def test_initial_pose_references_are_applied_in_dependency_order() -> None:
    handlers = {
        "first": _BatchObject(
            PoseState(
                position=[[0.0, 0.0, 0.0]],
                orientation=[[0.0, 0.0, 0.0, 1.0]],
            )
        ),
        "second": _BatchObject(
            PoseState(
                position=[[0.0, 0.0, 0.0]],
                orientation=[[0.0, 0.0, 0.0, 1.0]],
            )
        ),
    }
    # Declare the dependent entry first to prove declaration order is not the
    # frame-resolution order.
    backend = MujocoTaskBackend(
        env=type("BatchEnv", (), {"batch_size": 1})(),
        operator_handlers={},
        object_handlers=handlers,
        initial_poses={
            "first": PoseOverrideConfig(reference="second", position=[1.0, 0.0, 0.0]),
            "second": PoseOverrideConfig(position=[2.0, 0.0, 0.0]),
        },
    )

    backend._apply_initial_poses()

    np.testing.assert_allclose(handlers["second"].pose.position[0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(handlers["first"].pose.position[0], [3.0, 0.0, 0.0])


def test_initial_pose_reference_cycles_are_rejected_before_mutation() -> None:
    handlers = {
        name: _BatchObject(
            PoseState(
                position=[[0.0, 0.0, 0.0]],
                orientation=[[0.0, 0.0, 0.0, 1.0]],
            )
        )
        for name in ("first", "second")
    }
    backend = MujocoTaskBackend(
        env=type("BatchEnv", (), {"batch_size": 1})(),
        operator_handlers={},
        object_handlers=handlers,
        initial_poses={
            "first": PoseOverrideConfig(reference="second", position=[1.0, 0.0, 0.0]),
            "second": PoseOverrideConfig(reference="first", position=[2.0, 0.0, 0.0]),
        },
    )

    with pytest.raises(ValueError, match="Circular initial pose reference"):
        backend._apply_initial_poses()

    np.testing.assert_allclose(handlers["first"].pose.position[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(handlers["second"].pose.position[0], [0.0, 0.0, 0.0])


def test_masked_initial_pose_keeps_unselected_camera_baseline() -> None:
    camera_pose = PoseState(
        position=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        orientation=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
    )
    backend = MujocoTaskBackend(
        env=type("BatchEnv", (), {"batch_size": 2})(),
        operator_handlers={},
        object_handlers={},
        camera_initial_poses={
            "camera": PoseOverrideConfig(position=[4.0, 5.0, 6.0]),
        },
    )

    def get_camera_pose(_name: str) -> PoseState:
        return camera_pose

    def set_camera_pose(
        _name: str,
        pose: PoseState,
        env_mask: np.ndarray,
    ) -> None:
        nonlocal camera_pose
        position = camera_pose.position.copy()
        orientation = camera_pose.orientation.copy()
        position[env_mask] = pose.position[env_mask]
        orientation[env_mask] = pose.orientation[env_mask]
        camera_pose = PoseState(position=position, orientation=orientation)

    backend._get_camera_pose = get_camera_pose  # type: ignore[method-assign]
    backend._set_camera_pose = set_camera_pose  # type: ignore[method-assign]
    backend._apply_camera_initial_poses()
    baseline = backend._default_camera_poses["camera"]
    camera_pose.position[1] = [9.0, 9.0, 9.0]
    backend._apply_camera_initial_poses(np.asarray([True, False]))

    np.testing.assert_allclose(
        backend._default_camera_poses["camera"].position[0],
        baseline.position[0],
    )
    np.testing.assert_allclose(
        backend._default_camera_poses["camera"].position[1],
        baseline.position[1],
    )


def test_shared_operator_pose_requires_row_zero_consistency() -> None:
    class SharedEnv:
        batch_size = 3
        _share_physics = True

        def __init__(self) -> None:
            physical = type(
                "Physical",
                (),
                {
                    "data": type("Data", (), {"ctrl": np.zeros(1)})(),
                    "model": type("Model", (), {"nu": 1})(),
                },
            )()
            self.envs = [physical, physical, physical]
            self.base_calls = 0
            self.eef_calls = 0

        def register_operator(self, *_args, **_kwargs) -> None:
            return None

        def override_operator_base_pose(self, *_args, **_kwargs) -> None:
            self.base_calls += 1

        def set_operator_home_eef_pose(self, *_args, **_kwargs) -> None:
            self.eef_calls += 1

    env = SharedEnv()
    handler = MujocoOperatorHandler(operator_name="arm", env=env)  # type: ignore[arg-type]
    differing = PoseState(
        position=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
        orientation=[[0.0, 0.0, 0.0, 1.0]] * 3,
    )

    with pytest.raises(ValueError, match="shared physics"):
        handler.set_pose(differing)
    with pytest.raises(ValueError, match="shared physics"):
        handler.set_home_end_effector_pose(differing, apply_home=False)
    assert env.base_calls == 0
    assert env.eef_calls == 0


def test_shared_operator_pose_partial_mask_still_checks_row_zero() -> None:
    class SharedEnv:
        batch_size = 3
        _share_physics = True

        def __init__(self) -> None:
            physical = type(
                "Physical",
                (),
                {
                    "data": type("Data", (), {"ctrl": np.zeros(1)})(),
                    "model": type("Model", (), {"nu": 1})(),
                },
            )()
            self.envs = [physical, physical, physical]

        def register_operator(self, *_args, **_kwargs) -> None:
            return None

        def override_operator_base_pose(self, *_args, **_kwargs) -> None:
            return None

    env = SharedEnv()
    handler = MujocoOperatorHandler(operator_name="arm", env=env)  # type: ignore[arg-type]
    differing = PoseState(
        position=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
        orientation=[[0.0, 0.0, 0.0, 1.0]] * 3,
    )

    with pytest.raises(ValueError, match="shared physics"):
        handler.set_pose(differing, env_mask=np.asarray([False, True, True]))


def test_shared_physics_rejects_divergent_object_pose_rows(tmp_path: Path) -> None:
    """Stateful pose writes must not silently let the last alias win."""
    path = _write_xml(
        tmp_path,
        """
        <mujoco>
          <worldbody>
            <body name="object">
              <freejoint name="object_joint"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )
    config = EnvConfig(scene=SceneConfig(base=path), enabled_sensors=set())
    physical = UnifiedMujocoEnv(config)
    try:
        env = type(
            "SharedBatch",
            (),
            {
                "batch_size": 2,
                "envs": [physical, physical],
                "_share_physics": True,
            },
        )()
        handler = MujocoObjectHandler(
            name="object",
            env=env,  # type: ignore[arg-type]
            body_name="object",
            freejoint_name="object_joint",
        )
        with pytest.raises(ValueError, match="shared physics"):
            handler.set_pose(
                PoseState(
                    position=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    orientation=[[0.0, 0.0, 0.0, 1.0]] * 2,
                )
            )
        # In shared mode row 0 is the canonical value even for a partial mask;
        # an equal row 0/row 1 pose is therefore applied exactly once.
        shared_pose = PoseState(
            position=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            orientation=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        )
        handler.set_pose(shared_pose, env_mask=np.asarray([False, True]))
        body_id = mujoco.mj_name2id(physical.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        np.testing.assert_allclose(physical.data.xpos[body_id], [1.0, 0.0, 0.0])
    finally:
        physical.close()
