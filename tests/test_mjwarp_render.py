"""Observation-capture tests for the MJWarp environment.

Rendering is where MJWarp differs most from MuJoCo: it ray-traces where MuJoCo
rasterises, so pixel-exact RGB agreement is not achievable and asserting it would
be wrong. These tests therefore pin what *must* hold -- the observation contract
(keys, shapes, dtypes), metric depth, and the segmentation-derived masks that
serve as training targets -- and record the RGB divergence as a measured
tolerance rather than treating it as a defect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv  # noqa: E402
from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv  # noqa: E402
from auto_atom.config.env_config import EnvConfig  # noqa: E402
from auto_atom.contracts import ObservationEnvProtocol  # noqa: E402
from auto_atom.execution_config import (  # noqa: E402
    prepare_task_config_for_instantiation,
)
from auto_atom.runtime import ComponentRegistry  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CAMERAS = ("rack_camera_front", "plate_cam")


def _config(batch_size: int, *, enable_mask: bool = False) -> EnvConfig:
    overrides = [
        "execution.mode=object_only",
        f"env.batch_size={batch_size}",
        "env.viewer=null",
    ]
    if enable_mask:
        # A top-level interpolation variable, not an env.* field.
        overrides.append("enable_mask=true")
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "aao_configs"), version_base=None
    ):
        cfg = compose(config_name="rack_plate_p7_v4_umi_v3", overrides=overrides)
    node = OmegaConf.to_container(
        prepare_task_config_for_instantiation(cfg).env, resolve=True
    )
    node.pop("_target_", None)
    return EnvConfig.model_validate(node)


def _capture(config: EnvConfig, *, interests=(["object"], ["pick"])):
    """Capture from both backends on the same config, cleanly registered."""
    ComponentRegistry.clear()
    warp_env = MjWarpObjectOnlyEnv(config, njmax=512)
    warp_env.set_interest_objects_and_operations(*interests)
    warp_obs = warp_env.capture_observation()

    ComponentRegistry.clear()
    native_env = BatchedUnifiedMujocoEnv(config)
    native_env.set_interest_objects_and_operations(*interests)
    native_obs = native_env.capture_observation()
    return warp_env, warp_obs, native_env, native_obs


@pytest.fixture(scope="module")
def captures():
    config = _config(1, enable_mask=True)
    warp_env, warp_obs, native_env, native_obs = _capture(config)
    yield warp_env, warp_obs, native_env, native_obs
    warp_env.close()
    native_env.close()


def test_satisfies_observation_protocol(captures):
    warp_env = captures[0]
    assert isinstance(warp_env, ObservationEnvProtocol)


def test_observation_keys_match_the_native_env(captures):
    """The runtime keys observations by name, so the key sets must be identical."""
    _, warp_obs, _, native_obs = captures
    assert sorted(warp_obs) == sorted(native_obs)


def test_payload_shapes_and_dtypes_match_the_native_env(captures):
    """Consumers index a leading batch axis and assume the native dtypes."""
    _, warp_obs, _, native_obs = captures

    for key in sorted(native_obs):
        warp_data = np.asarray(warp_obs[key]["data"])
        native_data = np.asarray(native_obs[key]["data"])
        assert warp_data.shape == native_data.shape, key
        assert warp_data.dtype == native_data.dtype, key
        assert np.asarray(warp_obs[key]["t"]).shape == (1,), key


def test_batch_axis_covers_every_world():
    """MJWarp renders all worlds in one pass, so the axis is not a stack."""
    config = _config(2)
    ComponentRegistry.clear()
    env = MjWarpObjectOnlyEnv(config, njmax=512)
    try:
        env.set_interest_objects_and_operations(["object", "object"], ["pick", "place"])
        observation = env.capture_observation()
        for key, entry in observation.items():
            assert np.asarray(entry["data"]).shape[0] == 2, key
            assert np.asarray(entry["t"]).shape == (2,), key
    finally:
        env.close()


@pytest.mark.parametrize("camera", _CAMERAS)
def test_depth_is_metric_and_tracks_native(captures, camera):
    """Depth must be metres, not MJWarp's normalised [0, 1] output.

    get_depth returns clamp(value / depth_scale, 0, 1), so a naive scale of 1.0
    would cap every distance at one metre. Comparing against native on
    foreground pixels is what catches that: the cap shows up as a large error,
    while a correct conversion agrees to millimetres.
    """
    warp_env, warp_obs, _, native_obs = captures
    key = f"{camera}/aligned_depth_to_color/image_raw"
    warp_depth = np.asarray(warp_obs[key]["data"])[0]
    native_depth = np.asarray(native_obs[key]["data"])[0]
    model = warp_env.get_camera_model(camera)

    # Nothing is normalised: real distances exceed 1 m in this scene.
    assert warp_depth.max() > 1.0
    assert model.near <= warp_depth.min()
    assert warp_depth.max() <= model.far + 1e-3

    # Compare where both renderers see geometry rather than background, since a
    # silhouette pixel can legitimately be foreground for one and sky for the
    # other and would dominate a whole-image metric.
    foreground = (warp_depth < model.far * 0.99) & (native_depth < model.far * 0.99)
    assert foreground.sum() > 1000
    error = np.abs(warp_depth[foreground] - native_depth[foreground])
    assert error.mean() < 5e-3
    assert np.percentile(error, 99) < 5e-2


@pytest.mark.parametrize("camera", _CAMERAS)
def test_background_agrees_with_native(captures, camera):
    """A ray that hits nothing must read as the far plane, not zero.

    MJWarp reports 0.0 for a miss, which would otherwise read as "closer than
    everything in the scene" -- the opposite of the truth.
    """
    warp_env, warp_obs, _, native_obs = captures
    key = f"{camera}/aligned_depth_to_color/image_raw"
    far = warp_env.get_camera_model(camera).far
    warp_background = np.asarray(warp_obs[key]["data"])[0] >= far * 0.99
    native_background = np.asarray(native_obs[key]["data"])[0] >= far * 0.99

    assert warp_background.any()
    assert (warp_background == native_background).mean() > 0.999


@pytest.mark.parametrize("camera", _CAMERAS)
def test_masks_match_native_almost_exactly(captures, camera):
    """Masks are training targets, so they have to agree, and they do.

    These come from the segmentation pass rather than the shading model, which is
    why they agree where RGB cannot: measured IoU is 1.0 on three of the four
    camera/stream combinations and 0.9966 on the fourth, where four silhouette
    pixels differ.
    """
    _, warp_obs, _, native_obs = captures
    key = f"{camera}/mask/image_raw"
    warp_mask = np.asarray(warp_obs[key]["data"])[0]
    native_mask = np.asarray(native_obs[key]["data"])[0]

    assert warp_mask.any()
    intersection = np.logical_and(warp_mask > 0, native_mask > 0).sum()
    union = np.logical_or(warp_mask > 0, native_mask > 0).sum()
    assert intersection / union > 0.99


@pytest.mark.parametrize("camera", _CAMERAS)
def test_heat_map_channels_match_native(captures, camera):
    """Channel layout follows heatmap_operations on both backends."""
    warp_env, warp_obs, _, native_obs = captures
    key = f"{camera}/mask/heat_map"
    warp_heat = np.asarray(warp_obs[key]["data"])[0]
    native_heat = np.asarray(native_obs[key]["data"])[0]

    assert warp_heat.shape[-1] == len(warp_env.config.heatmap_operations)
    intersection = np.logical_and(warp_heat > 0, native_heat > 0).sum()
    union = np.logical_or(warp_heat > 0, native_heat > 0).sum()
    assert intersection / union > 0.99
    # 'pick' was the declared interest, so that channel carries the marks.
    pick = list(warp_env.config.heatmap_operations).index("pick")
    assert warp_heat[..., pick].any()


def test_heat_map_marks_each_world_from_its_own_stage():
    """Worlds run independently, so each marks the channel it is working on.

    The runtime passes one entry per world, so a world on 'place' must not be
    marked in the 'pick' channel just because another world is picking.
    """
    config = _config(2, enable_mask=True)
    ComponentRegistry.clear()
    env = MjWarpObjectOnlyEnv(config, njmax=512)
    try:
        env.set_interest_objects_and_operations(["object", "object"], ["pick", "place"])
        heat = np.asarray(env.capture_observation()["plate_cam/mask/heat_map"]["data"])
        operations = list(config.heatmap_operations)
        pick, place = operations.index("pick"), operations.index("place")

        assert heat[0, ..., pick].any() and not heat[0, ..., place].any()
        assert heat[1, ..., place].any() and not heat[1, ..., pick].any()
    finally:
        env.close()


def test_rgb_geometry_agrees_while_shading_differs(captures):
    """RGB cannot match pixel-exactly, and this records why.

    MJWarp ray-traces with its own lighting model; MuJoCo rasterises through
    OpenGL. No colour-space transform reconciles them -- sRGB-encoding or
    linearising either side measurably worsens the fit -- so the difference is
    the shading model, not a conversion this code should be applying. What must
    hold is that the *geometry* agrees: a structural mismatch would show up as
    large differences over a big fraction of the image.
    """
    _, warp_obs, _, native_obs = captures
    key = "rack_camera_front/color/image_raw"
    warp_rgb = np.asarray(warp_obs[key]["data"])[0].astype(np.int32)
    native_rgb = np.asarray(native_obs[key]["data"])[0].astype(np.int32)

    difference = np.abs(warp_rgb - native_rgb)
    assert np.median(difference) <= 8
    # Gross disagreement stays rare; a wrong camera pose or flipped image would
    # push this far higher.
    assert (difference > 96).mean() < 0.02


def test_capture_is_empty_when_the_camera_sensor_is_disabled():
    """No camera sensor means no rendering, and no context is ever built."""
    config = _config(1).model_copy(update={"enabled_sensors": set()})
    ComponentRegistry.clear()
    env = MjWarpObjectOnlyEnv(config, njmax=512)
    try:
        assert env.capture_observation() == {}
        assert env._renderer is None
    finally:
        env.close()


def test_unknown_mask_object_is_rejected():
    """An unnoticed empty mask channel is worse than a construction failure."""
    config = _config(1, enable_mask=True).model_copy(
        update={"mask_objects": ["no_such_body"]}
    )
    ComponentRegistry.clear()
    env = MjWarpObjectOnlyEnv(config, njmax=512)
    try:
        with pytest.raises(ValueError, match="no_such_body"):
            env.capture_observation()
    finally:
        env.close()


def test_interest_list_length_is_validated():
    config = _config(2)
    ComponentRegistry.clear()
    env = MjWarpObjectOnlyEnv(config, njmax=512)
    try:
        with pytest.raises(ValueError, match="same length"):
            env.set_interest_objects_and_operations(["object"], ["pick", "place"])
        with pytest.raises(ValueError, match="broadcast length 1 or per-env"):
            env.set_interest_objects_and_operations(["object"] * 3, ["pick"] * 3)
        with pytest.raises(ValueError, match="not configured in operations"):
            env.set_interest_objects_and_operations(["object"], ["fly"])
    finally:
        env.close()
