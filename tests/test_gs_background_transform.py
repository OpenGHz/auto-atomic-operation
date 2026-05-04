from pathlib import Path
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
    _merge_background_plys,
    _normalize_background_pose,
    _sample_combinations,
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


def test_batched_gs_is_updated_broadcasts_shared_physics_result():
    class _FakeEnv:
        def __init__(self) -> None:
            self.calls = 0

        def is_updated(self) -> bool:
            self.calls += 1
            return self.calls == 1

    fake = _FakeEnv()
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env._share_physics = True
    env.batch_size = 3
    env.envs = [fake, fake, fake]

    np.testing.assert_array_equal(
        BatchedGSUnifiedMujocoEnv.is_updated(env),
        np.array([True, True, True]),
    )
    np.testing.assert_array_equal(
        BatchedGSUnifiedMujocoEnv.is_updated(env),
        np.array([False, False, False]),
    )


# --- dict (parts) form of background_ply ---


def _write_dummy_ply_with_offset(path, offset: float) -> None:
    save_ply(
        GaussianData(
            xyz=np.array([[offset, 0.0, 0.0]], dtype=np.float32),
            rot=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            scale=np.ones((1, 3), dtype=np.float32),
            opacity=np.full((1,), 0.5, dtype=np.float32),
            sh=np.zeros((1, 3), dtype=np.float32),
        ),
        path,
    )


def test_dict_background_is_multi_background():
    cfg = GaussianRenderConfig(
        background_ply={"wall": "a.ply", "inside": "b.ply"},
    )
    assert cfg.is_multi_background() is True


def test_merge_background_plys_concatenates(tmp_path):
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    _write_dummy_ply_with_offset(a, 1.0)
    _write_dummy_ply_with_offset(b, 2.0)

    merged_path = _merge_background_plys([str(a), str(b)])
    merged = load_ply(merged_path)

    assert merged.xyz.shape == (2, 3)
    np.testing.assert_allclose(np.sort(merged.xyz[:, 0]), [1.0, 2.0])


def test_merge_background_plys_single_returns_input(tmp_path):
    a = tmp_path / "a.ply"
    _write_dummy_ply_with_offset(a, 1.0)
    assert _merge_background_plys([str(a)]) == str(a)


def test_merge_background_plys_cached(tmp_path, monkeypatch):
    import auto_atom.basis.mjc.gs_mujoco_env as gs_mod

    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    _write_dummy_ply_with_offset(a, 1.0)
    _write_dummy_ply_with_offset(b, 2.0)

    first = _merge_background_plys([str(a), str(b)])

    save_calls: list = []
    real_save = gs_mod.save_ply

    def _spy(*args, **kwargs):
        save_calls.append((args, kwargs))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(gs_mod, "save_ply", _spy)
    second = _merge_background_plys([str(a), str(b)])

    assert second == first
    assert save_calls == []


def test_sample_combinations_full_product_when_uncapped():
    parts = [["w1", "w2"], ["i1"]]
    combos = _sample_combinations(parts, cap=None, rng=np.random.default_rng(0))
    assert combos == [("w1", "i1"), ("w2", "i1")]


def test_sample_combinations_caps_without_replacement():
    parts = [["a", "b", "c", "d"], ["1", "2"]]
    combos = _sample_combinations(parts, cap=3, rng=np.random.default_rng(0))
    assert len(combos) == 3
    assert len(set(combos)) == 3  # no duplicates
    for combo in combos:
        assert combo[0] in {"a", "b", "c", "d"}
        assert combo[1] in {"1", "2"}


def test_sample_combinations_empty_part_raises():
    with pytest.raises(ValueError, match="empty part"):
        _sample_combinations([["a"], []], cap=None, rng=np.random.default_rng(0))


def test_resolved_background_plys_dict_returns_merged_combos(tmp_path):
    w1 = tmp_path / "wall1.ply"
    w2 = tmp_path / "wall2.ply"
    inside = tmp_path / "inside10.ply"
    _write_dummy_ply_with_offset(w1, 1.0)
    _write_dummy_ply_with_offset(w2, 2.0)
    _write_dummy_ply_with_offset(inside, 10.0)

    cfg = GaussianRenderConfig(
        background_ply={
            "wall": str(tmp_path / "wall*.ply"),
            "inside": str(inside),
        }
    )
    plys = cfg.resolved_background_plys()
    assert len(plys) == 2  # full product 2x1

    seen_walls = set()
    for path in plys:
        gd = load_ply(path)
        # Each merged PLY has 2 gaussians (one wall + one inside).
        assert gd.xyz.shape == (2, 3)
        wall_x = float(min(gd.xyz[:, 0]))
        inside_x = float(max(gd.xyz[:, 0]))
        assert inside_x == 10.0
        seen_walls.add(wall_x)
    assert seen_walls == {1.0, 2.0}


def test_resolved_background_plys_dict_max_combinations(tmp_path):
    walls = [tmp_path / f"wall{i}.ply" for i in range(4)]
    insides = [tmp_path / f"inside{i}.ply" for i in range(2)]
    for i, p in enumerate(walls):
        _write_dummy_ply_with_offset(p, float(i))
    for i, p in enumerate(insides):
        _write_dummy_ply_with_offset(p, 10.0 + i)

    cfg = GaussianRenderConfig(
        background_ply={
            "wall": str(tmp_path / "wall*.ply"),
            "inside": str(tmp_path / "inside*.ply"),
        }
    )
    plys = cfg.resolved_background_plys(
        max_combinations=3, rng=np.random.default_rng(42)
    )
    assert len(plys) == 3
    # Cap < total product (4*2=8): must sample without replacement.
    assert len(set(plys)) == 3


def test_resolved_part_plys_part_name_default(tmp_path):
    w0 = tmp_path / "wall0.ply"
    w1 = tmp_path / "wall1.ply"
    inside = tmp_path / "inside10.ply"
    for p in (w0, w1, inside):
        _write_dummy_ply_with_offset(p, 0.0)

    cfg = GaussianRenderConfig(
        background_ply={
            "wall": str(tmp_path / "wall*.ply"),
            "inside": str(inside),
        },
        background_transforms={
            "wall": [5.0, 0.0, 0.0],  # applies to every wall* PLY
            "inside": [0.0, 7.0, 0.0],  # applies to every inside PLY
        },
    )
    parts = cfg._resolved_part_plys()
    for path in parts["wall"]:
        gd = load_ply(path)
        np.testing.assert_allclose(gd.xyz[0], [5.0, 0.0, 0.0])
    for path in parts["inside"]:
        gd = load_ply(path)
        np.testing.assert_allclose(gd.xyz[0], [0.0, 7.0, 0.0])


def test_resolved_part_plys_per_ply_overrides_part_name(tmp_path):
    w0 = tmp_path / "wall0.ply"
    w1 = tmp_path / "wall1.ply"
    for p in (w0, w1):
        _write_dummy_ply_with_offset(p, 0.0)

    cfg = GaussianRenderConfig(
        background_ply={"wall": str(tmp_path / "wall*.ply")},
        background_transforms={
            "wall": [5.0, 0.0, 0.0],  # part-level default for the whole part
            "wall1": [9.0, 0.0, 0.0],  # per-PLY override wins for wall1
        },
    )
    parts = cfg._resolved_part_plys()
    by_stem = {Path(p).stem.split("__")[0]: p for p in parts["wall"]}
    np.testing.assert_allclose(load_ply(by_stem["wall0"]).xyz[0], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(load_ply(by_stem["wall1"]).xyz[0], [9.0, 0.0, 0.0])


def test_resolved_part_plys_part_name_overrides_global_default(tmp_path):
    w0 = tmp_path / "wall0.ply"
    inside = tmp_path / "inside10.ply"
    for p in (w0, inside):
        _write_dummy_ply_with_offset(p, 0.0)

    cfg = GaussianRenderConfig(
        background_ply={"wall": str(w0), "inside": str(inside)},
        background_transform=[1.0, 1.0, 1.0],  # global default
        background_transforms={
            "wall": [5.0, 0.0, 0.0],  # only overrides wall part
        },
    )
    parts = cfg._resolved_part_plys()
    np.testing.assert_allclose(load_ply(parts["wall"][0]).xyz[0], [5.0, 0.0, 0.0])
    # inside falls back to the global default.
    np.testing.assert_allclose(load_ply(parts["inside"][0]).xyz[0], [1.0, 1.0, 1.0])


def test_position_randomization_validator_rejects_bad_axis():
    with pytest.raises(ValueError, match="unknown axis"):
        GaussianRenderConfig(
            background_transform_randomization={
                "wall": {"w": [-0.1, 0.1]},
            },
        )


def test_position_randomization_validator_rejects_inverted_range():
    with pytest.raises(ValueError, match="low <= high"):
        GaussianRenderConfig(
            background_transform_randomization={
                "wall": {"x": [0.1, -0.1]},
            },
        )


def test_position_randomization_validator_rejects_wrong_length():
    with pytest.raises(ValueError, match=r"\[low, high\]"):
        GaussianRenderConfig(
            background_transform_randomization={
                "wall": {"x": [0.1, 0.2, 0.3]},
            },
        )


def test_dict_position_randomization_produces_distinct_offsets(tmp_path):
    """1 wall × 1 inside = 1 file combo, batch=4 → phase-1 yields 1
    deterministic env, phase-2 yields 3 randomized overflow envs.
    All four merged PLYs should be distinct cache entries."""
    wall = tmp_path / "wall0.ply"
    inside = tmp_path / "inside0.ply"
    _write_dummy_ply_with_offset(wall, 0.0)
    _write_dummy_ply_with_offset(inside, 0.0)

    cfg = GaussianRenderConfig(
        background_ply={"wall": str(wall), "inside": str(inside)},
        background_transform_randomization={
            "wall": {"x": [-0.5, 0.5]},
            "inside": {"y": [-0.3, 0.3]},
        },
    )
    assert cfg.has_position_randomization() is True

    rng = np.random.default_rng(0)
    plys = cfg.resolved_background_plys(max_combinations=4, rng=rng)
    assert len(plys) == 4
    # Phase-1 deterministic + 3 phase-2 randomized → 4 distinct cache files
    # (random samples are unique with prob 1 over a continuous range).
    assert len(set(plys)) == 4

    # All 4 merged PLYs satisfy the configured ranges (the deterministic
    # combo trivially: x=0 ∈ [-0.5, 0.5], y=0 ∈ [-0.3, 0.3]).
    for path in plys:
        gd = load_ply(path)
        assert gd.xyz.shape == (2, 3)
        for row in gd.xyz:
            x, y, _z = row
            if y == 0.0:
                assert -0.5 <= x <= 0.5
            else:
                assert x == 0.0
                assert -0.3 <= y <= 0.3


def test_position_randomization_seed_is_reproducible(tmp_path):
    wall = tmp_path / "wall0.ply"
    inside = tmp_path / "inside0.ply"
    _write_dummy_ply_with_offset(wall, 0.0)
    _write_dummy_ply_with_offset(inside, 0.0)
    cfg = GaussianRenderConfig(
        background_ply={"wall": str(wall), "inside": str(inside)},
        background_transform_randomization={
            "wall": {"x": [-0.5, 0.5]},
            "inside": {"y": [-0.3, 0.3]},
        },
    )
    paths_a = cfg.resolved_background_plys(
        max_combinations=3, rng=np.random.default_rng(42)
    )
    paths_b = cfg.resolved_background_plys(
        max_combinations=3, rng=np.random.default_rng(42)
    )
    assert paths_a == paths_b


def test_position_randomization_per_ply_key_overrides_part_key(tmp_path):
    w0 = tmp_path / "wall0.ply"
    w1 = tmp_path / "wall1.ply"
    _write_dummy_ply_with_offset(w0, 0.0)
    _write_dummy_ply_with_offset(w1, 0.0)
    cfg = GaussianRenderConfig(
        background_ply={"wall": str(tmp_path / "wall*.ply")},
        background_transform_randomization={
            "wall": {"x": [-1.0, 1.0]},  # part-level default
            "wall0": {"x": [10.0, 10.0]},  # exact PLY override (delta = 10)
        },
    )
    rng = np.random.default_rng(0)
    plys = cfg.resolved_background_plys(max_combinations=20, rng=rng)
    # 2 file combos available; 2 distinct (no offset, x=0) + 18 overflow
    # (with offset). Phase-1 gives at most one wall0 and one wall1 with x=0;
    # phase-2 entries respect the per-PLY vs part-level range precedence.
    seen_wall0_xs: list[float] = []
    seen_wall1_xs: list[float] = []
    for path in plys:
        gd = load_ply(path)
        assert gd.xyz.shape == (1, 3)
        x = float(gd.xyz[0, 0])
        if "wall0" in Path(path).name:
            seen_wall0_xs.append(x)
        else:
            seen_wall1_xs.append(x)
    for x in seen_wall0_xs:
        assert x == 0.0 or x == 10.0  # phase-1 deterministic OR phase-2 override
    for x in seen_wall1_xs:
        assert x == 0.0 or -1.0 <= x <= 1.0  # phase-1 OR part-level fallback


def test_position_randomization_skipped_when_files_cover_batch(tmp_path):
    """Phase-1 only: 2 walls × 2 insides = 4 combos, batch_size=4 → no
    overflow → no random offsets applied even though the config has them."""
    walls = [tmp_path / f"wall{i}.ply" for i in range(2)]
    insides = [tmp_path / f"inside{i}.ply" for i in range(2)]
    for p in walls + insides:
        _write_dummy_ply_with_offset(p, 0.0)

    cfg = GaussianRenderConfig(
        background_ply={
            "wall": str(tmp_path / "wall*.ply"),
            "inside": str(tmp_path / "inside*.ply"),
        },
        background_transform_randomization={
            "wall": {"x": [100.0, 100.0]},  # if applied this would be obvious
            "inside": {"y": [-50.0, 50.0]},
        },
    )
    plys = cfg.resolved_background_plys(
        max_combinations=4, rng=np.random.default_rng(7)
    )
    assert len(plys) == 4
    # All 4 envs must use the deterministic transforms only — pos still 0.
    for path in plys:
        gd = load_ply(path)
        np.testing.assert_allclose(gd.xyz, np.zeros_like(gd.xyz))


def test_position_randomization_overflow_only(tmp_path):
    """Phase-1 + phase-2: 1 wall × 1 inside = 1 combo, batch_size=4 → 1
    deterministic env + 3 randomized overflow envs."""
    wall = tmp_path / "wall0.ply"
    inside = tmp_path / "inside0.ply"
    _write_dummy_ply_with_offset(wall, 0.0)
    _write_dummy_ply_with_offset(inside, 0.0)

    cfg = GaussianRenderConfig(
        background_ply={"wall": str(wall), "inside": str(inside)},
        background_transform_randomization={
            "wall": {"x": [50.0, 50.0]},  # deterministic +50 when applied
            "inside": {"y": [-30.0, -30.0]},  # deterministic −30 when applied
        },
    )
    plys = cfg.resolved_background_plys(
        max_combinations=4, rng=np.random.default_rng(0)
    )
    assert len(plys) == 4

    # Find the merged PLY whose wall point sits at x=0 (phase-1 deterministic).
    n_deterministic = 0
    n_overflow = 0
    for path in plys:
        gd = load_ply(path)
        # Wall point: y=0 (only wall has the x range). Inside point: x=0.
        wall_row = gd.xyz[gd.xyz[:, 1] == 0.0][0]
        inside_row = gd.xyz[gd.xyz[:, 0] == 0.0][0]
        if wall_row[0] == 0.0 and inside_row[1] == 0.0:
            n_deterministic += 1
        else:
            assert wall_row[0] == 50.0
            assert inside_row[1] == -30.0
            n_overflow += 1
    assert n_deterministic == 1
    assert n_overflow == 3


def test_list_mode_position_randomization(tmp_path):
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    _write_dummy_ply_with_offset(a, 0.0)
    _write_dummy_ply_with_offset(b, 0.0)
    cfg = GaussianRenderConfig(
        background_ply=[str(a), str(b)],
        background_transform_randomization={
            "a": {"x": [5.0, 5.0]},  # deterministic +5 on a
            "b": {"y": [7.0, 7.0]},  # deterministic +7 on b
        },
    )
    plys = cfg.resolved_background_plys(rng=np.random.default_rng(0))
    assert len(plys) == 2

    by_stem = {Path(p).name.split("__")[0]: p for p in plys}
    a_gd = load_ply(by_stem["a"])
    b_gd = load_ply(by_stem["b"])
    np.testing.assert_allclose(a_gd.xyz[0], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(b_gd.xyz[0], [0.0, 7.0, 0.0])


def test_resolved_part_plys_applies_per_ply_transform(tmp_path):
    wall = tmp_path / "wall1.ply"
    inside = tmp_path / "inside10.ply"
    _write_dummy_ply_with_offset(wall, 0.0)
    _write_dummy_ply_with_offset(inside, 0.0)

    cfg = GaussianRenderConfig(
        background_ply={"wall": str(wall), "inside": str(inside)},
        background_transforms={
            "wall1": [5.0, 0.0, 0.0],  # translate wall by +5x
            "inside10": [0.0, 7.0, 0.0],  # translate inside by +7y
        },
    )
    parts = cfg._resolved_part_plys()
    wall_gd = load_ply(parts["wall"][0])
    inside_gd = load_ply(parts["inside"][0])

    np.testing.assert_allclose(wall_gd.xyz[0], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(inside_gd.xyz[0], [0.0, 7.0, 0.0])

    # The merged combo reflects both transforms.
    merged_paths = cfg.resolved_background_plys()
    assert len(merged_paths) == 1
    merged = load_ply(merged_paths[0])
    xs = sorted(merged.xyz[:, 0].tolist())
    ys = sorted(merged.xyz[:, 1].tolist())
    assert xs == [0.0, 5.0]
    assert ys == [0.0, 7.0]
