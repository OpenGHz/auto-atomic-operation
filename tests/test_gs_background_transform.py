from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import load_ply, save_ply

from auto_atom.basis.mjc.gs_mujoco_env import (
    BatchedGSUnifiedMujocoEnv,
    GaussianRenderConfig,
    GSUnifiedMujocoEnv,
    _materialize_transformed_background_ply,
    _normalize_background_pose,
    _sample_env_background_indices,
)
from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv, UnifiedMujocoEnv


def _write_dummy_ply(path) -> None:
    save_ply(
        GaussianData(
            xyz=np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=np.float32),
            rot=np.array(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32
            ),
            scale=np.ones((2, 3), dtype=np.float32),
            opacity=np.full((2,), 0.5, dtype=np.float32),
            sh=np.zeros((2, 3), dtype=np.float32),
        ),
        path,
    )


# --- _normalize_background_pose ---


def test_normalize_background_pose_xyz():
    pos, quat = _normalize_background_pose([0.1, 0.2, 0.3])
    assert pos == (0.1, 0.2, 0.3)
    assert quat == (0.0, 0.0, 0.0, 1.0)


def test_normalize_background_pose_xyzw():
    pos, quat = _normalize_background_pose([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert pos == (0.1, 0.2, 0.3)
    np.testing.assert_allclose(quat, (0.0, 0.0, 0.0, 1.0))


def test_normalize_background_pose_xyzw_normalizes():
    pos, quat = _normalize_background_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(quat, (0.0, 0.0, 0.0, 1.0))


def test_normalize_background_pose_passthrough():
    bp = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert _normalize_background_pose(bp) is bp


def test_normalize_background_pose_bad_length():
    with pytest.raises(ValueError, match="length 3.*or 7"):
        _normalize_background_pose([1.0, 2.0])


# --- GaussianRenderConfig ---


def test_resolved_background_transform_by_stem_xyz():
    cfg = GaussianRenderConfig(
        background_ply="assets/gs/backgrounds/discover-lab2.ply",
        background_transforms={
            "discover-lab2": [0.01, -0.02, 0.03],
        },
    )
    pos, quat = cfg.resolved_background_transform()
    assert pos == (0.01, -0.02, 0.03)
    assert quat == (0.0, 0.0, 0.0, 1.0)


def test_resolved_background_transform_by_stem_xyzw():
    cfg = GaussianRenderConfig(
        background_ply="assets/gs/backgrounds/door_bg.ply",
        background_transforms={
            "door_bg": [0.1, 0.2, 0.3, 0.0, 0.0, 0.7071068, 0.7071068],
        },
    )
    pos, quat = cfg.resolved_background_transform()
    assert pos == (0.1, 0.2, 0.3)
    np.testing.assert_allclose(quat, (0.0, 0.0, 0.7071068, 0.7071068), atol=1e-6)


def test_explicit_background_transform_takes_precedence():
    cfg = GaussianRenderConfig(
        background_ply="assets/gs/backgrounds/discover-lab2.ply",
        background_transform=[0.4, 0.5, 0.6],
        background_transforms={
            "discover-lab2": [0.01, -0.02, 0.03],
        },
    )
    pos, quat = cfg.resolved_background_transform()
    assert pos == (0.4, 0.5, 0.6)
    assert quat == (0.0, 0.0, 0.0, 1.0)


# --- _materialize_transformed_background_ply ---


def test_materialize_xyz_only(tmp_path):
    src = tmp_path / "background_0.ply"
    _write_dummy_ply(src)

    pose = ((0.25, -0.5, 1.0), (0.0, 0.0, 0.0, 1.0))
    shifted_path = _materialize_transformed_background_ply(str(src), pose)
    shifted = load_ply(shifted_path)

    np.testing.assert_allclose(
        shifted.xyz,
        np.array([[0.25, -0.4, 1.2], [1.25, 0.6, 2.2]], dtype=np.float32),
    )


def test_materialize_with_rotation(tmp_path):
    src = tmp_path / "background_rot.ply"
    _write_dummy_ply(src)

    # 90-degree rotation around Z: (qx, qy, qz, qw) = (0, 0, sin(45), cos(45))
    quat_xyzw = (0.0, 0.0, 0.7071068, 0.7071068)
    pose = ((0.0, 0.0, 0.0), quat_xyzw)
    result_path = _materialize_transformed_background_ply(str(src), pose)
    result = load_ply(result_path)

    # 90-deg Z rotation maps (x,y,z) -> (-y,x,z)
    expected_xyz = np.array([[-0.1, 0.0, 0.2], [-1.1, 1.0, 1.2]], dtype=np.float32)
    np.testing.assert_allclose(result.xyz, expected_xyz, atol=1e-4)


def test_materialize_identity_returns_original():
    path = "assets/gs/backgrounds/example.ply"
    pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert _materialize_transformed_background_ply(path, pose) == path


def test_materialize_none_returns_none():
    pose = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert _materialize_transformed_background_ply(None, pose) is None


def test_sample_env_background_indices_unique_when_backgrounds_cover_batch():
    rng = np.random.default_rng(123)
    indices = _sample_env_background_indices(
        batch_size=3,
        num_backgrounds=5,
        rng=rng,
    )

    assert indices.shape == (3,)
    assert len(np.unique(indices)) == 3
    assert np.all(indices >= 0)
    assert np.all(indices < 5)


def test_sample_env_background_indices_uses_all_backgrounds_when_counts_match():
    rng = np.random.default_rng(123)
    indices = _sample_env_background_indices(
        batch_size=4,
        num_backgrounds=4,
        rng=rng,
    )

    assert indices.shape == (4,)
    np.testing.assert_array_equal(np.sort(indices), np.arange(4, dtype=np.int64))


def test_sample_env_background_indices_allows_duplicates_when_backgrounds_insufficient():
    rng = np.random.default_rng(123)
    indices = _sample_env_background_indices(
        batch_size=5,
        num_backgrounds=2,
        rng=rng,
    )

    assert indices.shape == (5,)
    assert np.all(indices >= 0)
    assert np.all(indices < 2)
    assert len(np.unique(indices)) <= 2


def test_single_gs_reset_reassigns_multi_backgrounds_when_enabled(monkeypatch):
    monkeypatch.setattr(UnifiedMujocoEnv, "reset", lambda self: None)
    env = object.__new__(GSUnifiedMujocoEnv)
    env._is_multi_bg = True
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(randomize_background_on_reset=True)
    )
    env._randomize_active_bg = Mock()

    GSUnifiedMujocoEnv.reset(env)

    env._randomize_active_bg.assert_called_once_with()


def test_single_gs_reset_keeps_multi_background_when_disabled(monkeypatch):
    monkeypatch.setattr(UnifiedMujocoEnv, "reset", lambda self: None)
    env = object.__new__(GSUnifiedMujocoEnv)
    env._is_multi_bg = True
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(randomize_background_on_reset=False)
    )
    env._randomize_active_bg = Mock()

    GSUnifiedMujocoEnv.reset(env)

    env._randomize_active_bg.assert_not_called()


def test_batched_gs_reset_reassigns_multi_backgrounds_when_enabled(monkeypatch):
    monkeypatch.setattr(
        BatchedUnifiedMujocoEnv,
        "reset",
        lambda self, env_mask=None: None,
    )
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env._is_multi_bg = True
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(randomize_background_on_reset=True)
    )
    env._randomize_env_bg_assignment = Mock()
    env_mask = np.array([True, False])

    BatchedGSUnifiedMujocoEnv.reset(env, env_mask)

    env._randomize_env_bg_assignment.assert_called_once_with()


def test_batched_gs_reset_keeps_multi_backgrounds_when_disabled(monkeypatch):
    monkeypatch.setattr(
        BatchedUnifiedMujocoEnv,
        "reset",
        lambda self, env_mask=None: None,
    )
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env._is_multi_bg = True
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(randomize_background_on_reset=False)
    )
    env._randomize_env_bg_assignment = Mock()
    env_mask = np.array([True, False])

    BatchedGSUnifiedMujocoEnv.reset(env, env_mask)

    env._randomize_env_bg_assignment.assert_not_called()
