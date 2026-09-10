"""Object-mounted cameras: ``role='object'`` mounting, following, and validation.

A camera with ``role='object'`` is re-parented onto its reference object body so
MuJoCo's own kinematics make it follow the object (``cam_bodyid`` drives
``cam_xpos``/``cam_xmat``), with the local pose staying the install offset in the
object's frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from auto_atom.backend.mjc.mujoco_backend import MujocoObjectHandler, MujocoTaskBackend
from auto_atom.basis.mjc.mujoco_basis import MujocoBasis
from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv
from auto_atom.config.env_config import (
    CameraCalibrationConfig,
    CameraExtrinsicsConfig,
    CameraSpec,
    DataType,
    EnvConfig,
)
from auto_atom.config.pose import PoseOverrideConfig
from auto_atom.config.randomization import (
    PoseRandomRange,
    RandomizationScopeConfig,
    ResolvedRandomizationConfig,
)
from auto_atom.scene_composition import SceneConfig
from auto_atom.utils.pose import PoseState

SCENE_XML = """
<mujoco>
  <worldbody>
    <body name="cup" pos="0.5 0 0.8">
      <freejoint name="cup_joint"/>
      <geom type="box" size="0.03 0.03 0.05"/>
      <body name="cup_rim" pos="0 0 0.05">
        <site name="cup_top" pos="0 0 0.02"/>
      </body>
    </body>
    <camera name="obj_cam" pos="0.1 0 0.3"/>
    <camera name="scene_cam" pos="1 0 1"/>
  </worldbody>
</mujoco>
"""

# The MJCF above places the mouth of the cup at (0.5, 0, 0.85); ``cup_rim``
# inherits that plus its own 0.05 z offset from ``cup``.


def _camera(name: str, **kwargs: object) -> CameraSpec:
    """A camera whose native outputs are off, so no renderer is allocated."""
    defaults: dict[str, object] = {"enable_color": False, "enable_depth": False}
    defaults.update(kwargs)
    return CameraSpec(name=name, **defaults)  # type: ignore[arg-type]


def _write_scene(tmp_path: Path, xml: str = SCENE_XML) -> Path:
    path = tmp_path / "scene.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def _basis(
    tmp_path: Path, cameras: list[CameraSpec], xml: str = SCENE_XML
) -> MujocoBasis:
    return MujocoBasis(
        EnvConfig(
            scene=SceneConfig(base=_write_scene(tmp_path, xml)),
            enabled_sensors={DataType.CAMERA},
            cameras=cameras,
        )
    )


class _SingleEnvBatch:
    """Minimal ``envs`` carrier for backend-construction tests."""

    def __init__(self, single: MujocoBasis) -> None:
        self.envs = [single]
        self.batch_size = 1


def _cam_id(model: mujoco.MjModel, name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    assert cam_id >= 0, f"camera {name!r} missing from the model"
    return int(cam_id)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"body {name!r} missing from the model"
    return int(body_id)


def _set_object_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position: list[float],
    yaw: float,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    address = int(model.jnt_qposadr[joint_id])
    data.qpos[address : address + 7] = [
        *position,
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    ]
    mujoco.mj_forward(model, data)


def test_object_camera_mounts_on_reference_body_and_survives_reset(
    tmp_path: Path,
) -> None:
    env = _basis(tmp_path, [_camera("obj_cam", role="object", parent_frame="cup")])
    try:
        cam_id = _cam_id(env.model, "obj_cam")
        cup_id = _body_id(env.model, "cup")

        assert int(env.model.cam_bodyid[cam_id]) == cup_id
        assert env.object_camera_names == frozenset({"obj_cam"})
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(
            env.data.cam_xpos[cam_id], [0.6, 0.0, 1.1], atol=1e-12
        )

        # The mount offset is model-level state: a mutated camera pose is
        # restored by reset, and the mounting itself is not undone.
        env.model.cam_pos[cam_id] += [1.0, 2.0, 3.0]
        env.reset()

        assert int(env.model.cam_bodyid[cam_id]) == cup_id
        np.testing.assert_allclose(
            env.model.cam_pos[cam_id], [0.1, 0.0, 0.3], atol=1e-12
        )
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(
            env.data.cam_xpos[cam_id], [0.6, 0.0, 1.1], atol=1e-12
        )
    finally:
        env.close()


def test_object_camera_follows_object_motion_keeping_relative_pose(
    tmp_path: Path,
) -> None:
    env = _basis(tmp_path, [_camera("obj_cam", role="object", parent_frame="cup")])
    try:
        cam_id = _cam_id(env.model, "obj_cam")
        cup_id = _body_id(env.model, "cup")

        _set_object_pose(env.model, env.data, "cup_joint", [1.0, 1.0, 0.5], math.pi / 2)

        # Offset [0.1, 0, 0.3] rotated by +90 deg about z becomes [0, 0.1, 0.3].
        np.testing.assert_allclose(
            env.data.cam_xpos[cam_id], [1.0, 1.1, 0.8], atol=1e-9
        )
        # The relative pose is the invariant: camera minus object, in the
        # object's own frame, is still the configured install offset.
        cup_rotation = np.asarray(env.data.xmat[cup_id]).reshape(3, 3)
        relative = cup_rotation.T @ (
            np.asarray(env.data.cam_xpos[cam_id]) - np.asarray(env.data.xpos[cup_id])
        )
        np.testing.assert_allclose(relative, [0.1, 0.0, 0.3], atol=1e-9)
    finally:
        env.close()


def test_object_camera_extrinsics_are_object_local(tmp_path: Path) -> None:
    env = _basis(
        tmp_path,
        [
            _camera(
                "obj_cam",
                role="object",
                parent_frame="cup",
                calibration=CameraCalibrationConfig(
                    extrinsics=CameraExtrinsicsConfig(position=(0.0, 0.0, 0.25))
                ),
            )
        ],
    )
    try:
        cam_id = _cam_id(env.model, "obj_cam")

        np.testing.assert_allclose(
            env.model.cam_pos[cam_id], [0.0, 0.0, 0.25], atol=1e-12
        )
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(
            env.data.cam_xpos[cam_id], [0.5, 0.0, 1.05], atol=1e-12
        )

        _set_object_pose(env.model, env.data, "cup_joint", [0.0, 0.0, 0.4], 0.0)
        np.testing.assert_allclose(
            env.data.cam_xpos[cam_id], [0.0, 0.0, 0.65], atol=1e-9
        )
    finally:
        env.close()


def test_object_camera_site_reference_mounts_on_site_body(tmp_path: Path) -> None:
    env = _basis(tmp_path, [_camera("obj_cam", role="object", parent_frame="cup_top")])
    try:
        cam_id = _cam_id(env.model, "obj_cam")

        assert int(env.model.cam_bodyid[cam_id]) == _body_id(env.model, "cup_rim")
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(
            env.data.cam_xpos[cam_id], [0.6, 0.0, 1.15], atol=1e-12
        )
    finally:
        env.close()


def test_object_camera_falls_back_to_gs_body_name(tmp_path: Path) -> None:
    xml = SCENE_XML.replace(
        'name="cup" pos="0.5 0 0.8"', 'name="cup_gs" pos="0.5 0 0.8"'
    )
    env = _basis(
        tmp_path,
        [_camera("obj_cam", role="object", parent_frame="cup")],
        xml=xml,
    )
    try:
        cam_id = _cam_id(env.model, "obj_cam")

        assert int(env.model.cam_bodyid[cam_id]) == _body_id(env.model, "cup_gs")
    finally:
        env.close()


def test_object_camera_without_mount_frame_rejects_world_body(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be mounted on a scene body"):
        _basis(tmp_path, [_camera("obj_cam", role="object")])


def test_object_camera_auto_detects_mjcf_parent_body(tmp_path: Path) -> None:
    xml = SCENE_XML.replace(
        '<camera name="obj_cam" pos="0.1 0 0.3"/>',
        "",
    ).replace(
        '<body name="cup_rim" pos="0 0 0.05">',
        '<body name="cup_rim" pos="0 0 0.05">\n        <camera name="obj_cam" pos="0.1 0 0.3"/>',
    )
    env = _basis(tmp_path, [_camera("obj_cam", role="object")], xml=xml)
    try:
        cam_id = _cam_id(env.model, "obj_cam")

        assert int(env.model.cam_bodyid[cam_id]) == _body_id(env.model, "cup_rim")
        assert env.object_camera_names == frozenset({"obj_cam"})
    finally:
        env.close()


def test_object_camera_requires_resolvable_parent_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found as site or body"):
        _basis(tmp_path, [_camera("obj_cam", role="object", parent_frame="missing")])


def test_object_camera_rejects_static_gs_background() -> None:
    with pytest.raises(ValidationError, match="is_static=True cannot be combined"):
        _camera("obj_cam", role="object", parent_frame="cup", is_static=True)


def test_scene_cameras_keep_their_mjcf_parent(tmp_path: Path) -> None:
    env = _basis(
        tmp_path,
        [
            _camera("obj_cam", role="object", parent_frame="cup"),
            _camera("scene_cam"),
        ],
    )
    try:
        assert int(env.model.cam_bodyid[_cam_id(env.model, "scene_cam")]) == 0
        assert env.object_camera_names == frozenset({"obj_cam"})
    finally:
        env.close()


def test_object_camera_rejects_camera_initial_pose(tmp_path: Path) -> None:
    env = _basis(tmp_path, [_camera("obj_cam", role="object", parent_frame="cup")])
    try:
        with pytest.raises(ValueError, match="object-mounted"):
            MujocoTaskBackend(
                env=_SingleEnvBatch(env),  # type: ignore[arg-type]
                operator_handlers={},
                object_handlers={},
                camera_initial_poses={
                    "obj_cam": PoseOverrideConfig(position=[0.0, 0.0, 0.0]),
                },
            )
    finally:
        env.close()


def _relative_camera_offset(
    env: MujocoBasis,
    camera_id: int,
    body_id: int,
) -> np.ndarray:
    """The camera position expressed in the object body's own frame."""
    mujoco.mj_forward(env.model, env.data)
    body_rotation = np.asarray(env.data.xmat[body_id]).reshape(3, 3)
    return body_rotation.T @ (
        np.asarray(env.data.cam_xpos[camera_id]) - np.asarray(env.data.xpos[body_id])
    )


def _camera_backend(
    env: MujocoBasis,
    cameras: dict[str, object],
    *,
    seed: int = 7,
) -> MujocoTaskBackend:
    """A backend whose only randomization is the given per-camera entries."""
    return MujocoTaskBackend(
        env=_SingleEnvBatch(env),  # type: ignore[arg-type]
        operator_handlers={},
        object_handlers={},
        randomization=ResolvedRandomizationConfig.from_scope_config(
            RandomizationScopeConfig(cameras=cameras)  # type: ignore[arg-type]
        ),
        random_seed=seed,
    )


def test_object_camera_randomization_samples_the_install_offset(
    tmp_path: Path,
) -> None:
    """An object camera randomizes its mount-frame offset, not a world pose."""
    env = UnifiedMujocoEnv(
        EnvConfig(
            scene=SceneConfig(base=_write_scene(tmp_path)),
            enabled_sensors={DataType.CAMERA},
            cameras=[_camera("obj_cam", role="object", parent_frame="cup")],
        )
    )
    try:
        backend = _camera_backend(env, {"obj_cam": PoseRandomRange(z=(-0.05, 0.05))})
        backend._record_default_poses()
        camera_id = _cam_id(env.model, "obj_cam")
        cup_id = _body_id(env.model, "cup")
        baseline = backend.get_camera_mount_pose("obj_cam").position[0].copy()

        backend.randomization_executor.apply_camera_randomization(
            np.asarray([True], dtype=bool)
        )

        sampled = backend.get_camera_mount_pose("obj_cam").position[0]
        # The mount is untouched, unspecified axes keep the install offset, and
        # the sampled axis is an additive offset in the mount frame.
        assert int(env.model.cam_bodyid[camera_id]) == cup_id
        np.testing.assert_allclose(sampled[:2], baseline[:2], atol=1e-12)
        assert -0.05 <= sampled[2] - baseline[2] <= 0.05
        assert sampled[2] != pytest.approx(baseline[2])

        # The sampled offset is what the object transport then carries, so the
        # randomization composes with motion instead of fighting it.
        handler = MujocoObjectHandler(
            name="cup",
            env=_SingleEnvBatch(env),  # type: ignore[arg-type]
            body_name="cup",
            freejoint_name="cup_joint",
        )
        yaw = 0.4
        handler.set_pose(
            PoseState(
                position=[[0.2, -0.1, 0.6]],
                orientation=[[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]],
            )
        )
        np.testing.assert_allclose(
            _relative_camera_offset(env, camera_id, cup_id), sampled, atol=1e-9
        )
    finally:
        env.close()


def test_object_camera_rejects_world_frame_randomization(tmp_path: Path) -> None:
    env = UnifiedMujocoEnv(
        EnvConfig(
            scene=SceneConfig(base=_write_scene(tmp_path)),
            enabled_sensors={DataType.CAMERA},
            cameras=[_camera("obj_cam", role="object", parent_frame="cup")],
        )
    )
    try:
        backend = _camera_backend(
            env,
            {
                "obj_cam": PoseRandomRange.model_validate(
                    {"reference": "absolute_world", "x": [0.0, 1.0]}
                )
            },
        )
        assert backend.object_camera_names() == {"obj_cam"}

        with pytest.raises(ValueError, match="install offset in the mount frame"):
            backend.randomization_executor.apply_camera_randomization(
                np.asarray([True], dtype=bool)
            )
    finally:
        env.close()


def test_fixed_camera_randomization_keeps_world_frame_modes(tmp_path: Path) -> None:
    env = UnifiedMujocoEnv(
        EnvConfig(
            scene=SceneConfig(base=_write_scene(tmp_path)),
            enabled_sensors={DataType.CAMERA},
            cameras=[_camera("scene_cam")],
        )
    )
    try:
        backend = _camera_backend(
            env,
            {
                "scene_cam": PoseRandomRange.model_validate(
                    {"reference": "absolute_world", "x": [2.0, 2.0]}
                )
            },
        )
        backend._record_default_poses()

        backend.randomization_executor.apply_camera_randomization(
            np.asarray([True], dtype=bool)
        )

        world = backend.get_camera_pose("scene_cam").position[0]
        np.testing.assert_allclose(world[0], 2.0, atol=1e-9)
    finally:
        env.close()


def test_object_camera_follows_object_only_transport(tmp_path: Path) -> None:
    """The object-only transport path keeps the camera-object relative pose.

    ``object_only`` moves the object kinematically through
    :meth:`MujocoObjectHandler.set_pose`; the camera is a child body, so its
    install offset survives arbitrary transport waypoints.
    """
    env = UnifiedMujocoEnv(
        EnvConfig(
            scene=SceneConfig(base=_write_scene(tmp_path)),
            enabled_sensors={DataType.CAMERA},
            cameras=[_camera("obj_cam", role="object", parent_frame="cup")],
        )
    )
    try:
        handler = MujocoObjectHandler(
            name="cup",
            env=_SingleEnvBatch(env),  # type: ignore[arg-type]
            body_name="cup",
            freejoint_name="cup_joint",
        )
        camera_id = _cam_id(env.model, "obj_cam")
        cup_id = _body_id(env.model, "cup")

        waypoints = (
            ([0.0, 0.0, 0.5], math.pi / 2),
            ([0.3, -0.4, 0.9], -math.pi / 3),
        )
        for position, yaw in waypoints:
            handler.set_pose(
                PoseState(
                    position=[position],
                    orientation=[[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]],
                )
            )
            np.testing.assert_allclose(
                _relative_camera_offset(env, camera_id, cup_id),
                [0.1, 0.0, 0.3],
                atol=1e-9,
            )
    finally:
        env.close()
