"""Config-declared cameras: the composed scene creates what it does not author.

A camera that the MJCF does not define used to be a hard error.  Now the config
declaration itself is enough: the mount frame, mount pose and optics it carries
are materialized onto the editable spec before the model is compiled, so the
scene XML only has to describe the scene.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from auto_atom.basis.mjc.mujoco_basis import MujocoBasis
from auto_atom.config.env_config import (
    CameraCalibrationConfig,
    CameraExtrinsicsConfig,
    CameraSpec,
    DataType,
    EnvConfig,
)
from auto_atom.scene_composition import SceneConfig

SCENE_XML = """
<mujoco>
  <worldbody>
    <body name="cup" pos="0.5 0 0.8">
      <freejoint name="cup_joint"/>
      <geom type="box" size="0.03 0.03 0.05"/>
      <body name="cup_rim" pos="0 0 0.05">
        <site name="cup_top" pos="0.02 0 0.03" quat="0.9238795 0 0.3826834 0"/>
      </body>
    </body>
    <camera name="authored_cam" pos="1 0 1"/>
  </worldbody>
</mujoco>
"""

# Gaussian-splatting scenes name an object's body ``<name>_gs``.
GS_SCENE_XML = """
<mujoco>
  <worldbody>
    <body name="cup_gs" pos="0.5 0 0.8">
      <geom type="box" size="0.03 0.03 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

# Rotation about Z by 90 degrees, in both conventions the stack uses:
# ``calibration.extrinsics.orientation`` is xyzw, an MJCF ``quat`` is wxyz.
ORIENTATION_XYZW = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
ORIENTATION_WXYZ = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
SITE_ORIENTATION_WXYZ = (0.9238795, 0.0, 0.3826834, 0.0)
SITE_POSITION = (0.02, 0.0, 0.03)


def _camera(
    name: str,
    *,
    parent_frame: str = "",
    position: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    fovy_deg: float | None = None,
    **kwargs: object,
) -> CameraSpec:
    """A declared camera with native outputs off, so no renderer is allocated."""

    calibration = None
    if position is not None or orientation is not None or fovy_deg is not None:
        calibration = CameraCalibrationConfig(
            fovy_deg=fovy_deg,
            extrinsics=(
                CameraExtrinsicsConfig(position=position, orientation=orientation)
                if position is not None or orientation is not None
                else None
            ),
        )
    defaults: dict[str, object] = {
        "enable_color": False,
        "enable_depth": False,
        "parent_frame": parent_frame,
        "calibration": calibration,
    }
    defaults.update(kwargs)
    return CameraSpec(name=name, **defaults)  # type: ignore[arg-type]


def _basis(
    directory: Path,
    cameras: list[CameraSpec],
    xml: str = SCENE_XML,
    enabled_sensors: set[DataType] | None = None,
) -> MujocoBasis:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scene.xml"
    path.write_text(xml, encoding="utf-8")
    return MujocoBasis(
        EnvConfig(
            scene=SceneConfig(base=path),
            enabled_sensors=(
                {DataType.CAMERA} if enabled_sensors is None else enabled_sensors
            ),
            cameras=cameras,
        )
    )


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"{object_type.name} {name!r} missing from the model"
    return int(object_id)


def _world_camera_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)
    return (
        np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy(),
        np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def _quat_to_matrix(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def _quat_multiply(
    left_wxyz: tuple[float, float, float, float],
    right_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    result = np.zeros(4)
    mujoco.mju_mulQuat(
        result, np.asarray(left_wxyz, dtype=np.float64), np.asarray(right_wxyz)
    )
    return tuple(float(value) for value in result)


def test_declared_camera_is_created_on_its_mount_body(tmp_path: Path) -> None:
    """A camera no MJCF defines is built from the declaration and follows it."""

    env = _basis(
        tmp_path,
        [
            _camera(
                "declared_cam",
                parent_frame="cup",
                position=(0.1, 0.2, 0.3),
                orientation=ORIENTATION_XYZW,
                fovy_deg=70.0,
            )
        ],
    )
    try:
        cam_id = _id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "declared_cam")
        cup_id = _id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        assert int(env.model.cam_bodyid[cam_id]) == cup_id
        np.testing.assert_allclose(env.model.cam_pos[cam_id], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(env.model.cam_quat[cam_id], ORIENTATION_WXYZ)
        assert float(env.model.cam_fovy[cam_id]) == pytest.approx(70.0)

        # Riding the body is MuJoCo's job: move the object and the camera's
        # world pose follows while the install offset stays put.
        joint = _id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint")
        address = int(env.model.jnt_qposadr[joint])
        env.data.qpos[address : address + 7] = [0.2, 0.1, 0.9, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)
        body_rotation = np.asarray(env.data.xmat[cup_id]).reshape(3, 3)
        offset = body_rotation.T @ (env.data.cam_xpos[cam_id] - env.data.xpos[cup_id])
        np.testing.assert_allclose(offset, [0.1, 0.2, 0.3], atol=1.0e-9)
    finally:
        env.close()


def test_declared_camera_site_frame_matches_manual_composition(tmp_path: Path) -> None:
    """Sites mount the camera on their owning body and compose their own pose."""

    offset_in_site = np.array([0.01, 0.0, 0.02])
    site_env = _basis(
        tmp_path / "site",
        [
            _camera(
                "site_cam",
                parent_frame="cup_top",
                position=tuple(float(value) for value in offset_in_site),
                orientation=ORIENTATION_XYZW,
            )
        ],
    )
    # The same mount expressed directly against the body that owns the site:
    # position = site.pos + R(site.quat) @ offset, quat = site ⊗ camera.
    composed_wxyz = _quat_multiply(SITE_ORIENTATION_WXYZ, ORIENTATION_WXYZ)
    body_env = _basis(
        tmp_path / "body",
        [
            _camera(
                "site_cam",
                parent_frame="cup_rim",
                position=tuple(
                    float(value)
                    for value in (
                        np.asarray(SITE_POSITION)
                        + _quat_to_matrix(SITE_ORIENTATION_WXYZ) @ offset_in_site
                    )
                ),
                orientation=(
                    float(composed_wxyz[1]),
                    float(composed_wxyz[2]),
                    float(composed_wxyz[3]),
                    float(composed_wxyz[0]),
                ),
            )
        ],
    )
    try:
        site_id = _id(site_env.model, mujoco.mjtObj.mjOBJ_CAMERA, "site_cam")
        body_id = _id(body_env.model, mujoco.mjtObj.mjOBJ_CAMERA, "site_cam")
        # The site resolves to the body it lives in, not to the scene root.
        assert int(site_env.model.cam_bodyid[site_id]) == _id(
            site_env.model, mujoco.mjtObj.mjOBJ_BODY, "cup_rim"
        )
        assert int(body_env.model.cam_bodyid[body_id]) == _id(
            body_env.model, mujoco.mjtObj.mjOBJ_BODY, "cup_rim"
        )
        site_position, site_rotation = _world_camera_pose(
            site_env.model, site_env.data, site_id
        )
        body_position, body_rotation = _world_camera_pose(
            body_env.model, body_env.data, body_id
        )
        np.testing.assert_allclose(site_position, body_position, atol=1.0e-6)
        np.testing.assert_allclose(site_rotation, body_rotation, atol=1.0e-6)
    finally:
        site_env.close()
        body_env.close()


def test_declared_camera_reuses_an_authored_element(tmp_path: Path) -> None:
    """Declaring a camera the scene authors must not create a second one."""

    env = _basis(tmp_path, [_camera("authored_cam")])
    try:
        cam_id = _id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "authored_cam")
        assert env.model.ncam == 1
        np.testing.assert_allclose(env.model.cam_pos[cam_id], [1.0, 0.0, 1.0])
        assert int(env.model.cam_bodyid[cam_id]) == 0
    finally:
        env.close()


def test_object_declared_camera_resolves_the_gs_body_alias(tmp_path: Path) -> None:
    """An object camera may name the object while the body is ``<name>_gs``."""

    env = _basis(
        tmp_path,
        [
            _camera(
                "obj_cam",
                role="object",
                parent_frame="cup",
                position=(0.25, 0.0, 0.0),
                orientation=(0.5, 0.5, 0.5, 0.5),
            )
        ],
        xml=GS_SCENE_XML,
    )
    try:
        cam_id = _id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "obj_cam")
        assert int(env.model.cam_bodyid[cam_id]) == _id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "cup_gs"
        )
        np.testing.assert_allclose(env.model.cam_pos[cam_id], [0.25, 0.0, 0.0])
    finally:
        env.close()


def test_declared_camera_is_inert_without_the_camera_sensor(tmp_path: Path) -> None:
    """A camera that produces no stream needs no element, as before."""

    env = _basis(
        tmp_path,
        [_camera("declared_cam", parent_frame="cup")],
        enabled_sensors=set(),
    )
    try:
        assert env.model.ncam == 1
        assert (
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "declared_cam") < 0
        )
    finally:
        env.close()


def test_declared_camera_without_mount_pose_fails_closed(tmp_path: Path) -> None:
    """A camera that has to be created must say where it is mounted."""

    with pytest.raises(ValueError, match=r"declared_cam.*calibration\.extrinsics"):
        _basis(tmp_path, [_camera("declared_cam", parent_frame="cup")])


def test_declared_camera_with_unknown_parent_frame_fails_closed(tmp_path: Path) -> None:
    """An unresolvable mount frame is a configuration bug, not a world mount."""

    with pytest.raises(ValueError, match="parent_frame 'cup_handle'"):
        _basis(
            tmp_path,
            [
                _camera(
                    "declared_cam",
                    parent_frame="cup_handle",
                    position=(0.0, 0.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                )
            ],
        )
