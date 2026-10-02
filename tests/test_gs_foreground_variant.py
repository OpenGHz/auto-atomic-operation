"""Tests for the ``foreground_variant`` FG-grouped round-robin sampling mode."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

# Gaussian splatting lives in the pixi `gs` environment (`pixi run -e gs test`).
pytest.importorskip("gaussian_renderer")

import auto_atom.basis.mjc.gs_mujoco_env as gs_env
from auto_atom.basis.mjc.gs_mujoco_env import (
    BatchedGSUnifiedMujocoEnv,
    GSUnifiedMujocoEnv,
    GaussianRenderConfig,
    _FGBGCombinationCursor,
    _fg_variant_at,
    _fg_variant_count,
    _fg_variant_lengths,
)
from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv


# --- variant helpers ---


def test_fg_variant_lengths_mixed_str_and_list():
    sizes = _fg_variant_lengths({"door": "door.ply", "knob": ["k0.ply", "k1.ply"]})
    assert sizes == {"door": 1, "knob": 2}


def test_fg_variant_lengths_empty_dict():
    assert _fg_variant_lengths({}) == {}


def test_fg_variant_lengths_empty_list_raises():
    with pytest.raises(ValueError, match="empty list"):
        _fg_variant_lengths({"door": []})


def test_fg_variant_count_cartesian():
    bg = {
        "door": ["d0.ply", "d1.ply", "d2.ply"],
        "knob": ["k0.ply", "k1.ply"],
    }
    assert _fg_variant_count(bg) == 6


def test_fg_variant_count_single_strings():
    assert _fg_variant_count({"door": "d.ply", "knob": "k.ply"}) == 1


def test_fg_variant_count_empty_dict_is_zero():
    assert _fg_variant_count({}) == 0


def test_fg_variant_at_decodes_via_unravel():
    bg = {
        "door": ["d0", "d1", "d2"],
        "knob": ["k0", "k1"],
    }
    # 3 * 2 = 6 variants. unravel_index uses C-order (row-major).
    assert _fg_variant_at(bg, 0) == {"door": "d0", "knob": "k0"}
    assert _fg_variant_at(bg, 1) == {"door": "d0", "knob": "k1"}
    assert _fg_variant_at(bg, 2) == {"door": "d1", "knob": "k0"}
    assert _fg_variant_at(bg, 5) == {"door": "d2", "knob": "k1"}


def test_fg_variant_at_preserves_str_entries_across_variants():
    bg = {"door": "constant.ply", "knob": ["k0", "k1"]}
    assert _fg_variant_at(bg, 0) == {"door": "constant.ply", "knob": "k0"}
    assert _fg_variant_at(bg, 1) == {"door": "constant.ply", "knob": "k1"}


def test_fg_variant_at_out_of_range_raises():
    bg = {"door": ["d0", "d1"]}
    with pytest.raises(IndexError):
        _fg_variant_at(bg, 2)
    with pytest.raises(IndexError):
        _fg_variant_at(bg, -1)


# --- _FGBGCombinationCursor ---


def test_cursor_rejects_num_bg_below_batch_size():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="num_bg >= batch_size"):
        _FGBGCombinationCursor(num_fg=2, num_bg=3, batch_size=4, rng=rng)


def test_cursor_rejects_zero_or_negative_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _FGBGCombinationCursor(num_fg=0, num_bg=4, batch_size=2, rng=rng)
    with pytest.raises(ValueError):
        _FGBGCombinationCursor(num_fg=2, num_bg=0, batch_size=2, rng=rng)
    with pytest.raises(ValueError):
        _FGBGCombinationCursor(num_fg=2, num_bg=4, batch_size=0, rng=rng)


def test_cursor_returns_distinct_bgs_per_batch():
    rng = np.random.default_rng(0)
    cursor = _FGBGCombinationCursor(num_fg=3, num_bg=8, batch_size=2, rng=rng)
    for _ in range(12):  # exhaust the round (4 batches per FG * 3 FGs) and overflow
        fg, bgs = cursor.next_batch()
        assert len(bgs) == 2
        assert len(set(bgs)) == 2  # distinct within the batch
        assert all(0 <= b < 8 for b in bgs)
        assert 0 <= fg < 3


def test_cursor_holds_fg_constant_across_one_round_until_exhausted():
    rng = np.random.default_rng(1)
    M, N, B = 3, 6, 2
    cursor = _FGBGCombinationCursor(num_fg=M, num_bg=N, batch_size=B, rng=rng)
    seen: list[tuple[int, tuple[int, ...]]] = []
    fg_to_bgs: dict[int, list[int]] = {}
    for _ in range(M * (N // B)):  # one full round
        fg, bgs = cursor.next_batch()
        seen.append((fg, tuple(bgs)))
        fg_to_bgs.setdefault(fg, []).extend(bgs)

    # Per-FG: collected B*(N//B) backgrounds and all are distinct.
    for fg, bgs in fg_to_bgs.items():
        assert len(bgs) == B * (N // B) == 6  # full N consumed per FG
        assert len(set(bgs)) == len(bgs)

    # All M foregrounds appear in this round.
    assert set(fg_to_bgs) == set(range(M))

    # FGs are visited in *contiguous runs* of (N // B) batches each.
    fg_run_lengths = []
    last_fg: int | None = None
    run_len = 0
    for fg, _bgs in seen:
        if fg != last_fg:
            if last_fg is not None:
                fg_run_lengths.append(run_len)
            run_len = 1
            last_fg = fg
        else:
            run_len += 1
    fg_run_lengths.append(run_len)
    assert all(r == N // B for r in fg_run_lengths)


def test_cursor_drops_remainder_when_bg_pool_not_multiple_of_batch():
    # N=5, B=2 → 2 batches per FG (4 BGs) + 1 remainder skipped.
    rng = np.random.default_rng(2)
    M, N, B = 2, 5, 2
    cursor = _FGBGCombinationCursor(num_fg=M, num_bg=N, batch_size=B, rng=rng)
    # One round = M * (N // B) = 4 batches.
    seen: list[tuple[int, tuple[int, ...]]] = []
    for _ in range(4):
        fg, bgs = cursor.next_batch()
        seen.append((fg, tuple(bgs)))
    fgs = [f for f, _ in seen]
    # Two FGs, each with 2 consecutive batches.
    counts = Counter(fgs)
    assert set(counts.values()) == {2}
    # Verify no FG-batch consumed more than ``N // B * B = 4`` distinct BGs.
    by_fg: dict[int, set[int]] = {}
    for fg, bgs in seen:
        by_fg.setdefault(fg, set()).update(bgs)
    for fg, bgs in by_fg.items():
        assert len(bgs) == 4


def test_cursor_reshuffles_after_round_exhausted():
    rng = np.random.default_rng(3)
    M, N, B = 2, 4, 2
    cursor = _FGBGCombinationCursor(num_fg=M, num_bg=N, batch_size=B, rng=rng)
    # First round = M * (N // B) = 4 batches.
    round1 = [cursor.next_batch() for _ in range(4)]
    # Second round starts fresh: per-FG BG order is reshuffled, FG order may
    # differ too.
    round2 = [cursor.next_batch() for _ in range(4)]

    # Both rounds cover the full M*N space exactly once.
    def _flatten(round_batches):
        out = set()
        for fg, bgs in round_batches:
            for bg in bgs:
                out.add((fg, bg))
        return out

    assert _flatten(round1) == {(fg, bg) for fg in range(M) for bg in range(N)}
    assert _flatten(round2) == {(fg, bg) for fg in range(M) for bg in range(N)}


def test_cursor_is_reproducible_under_same_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    cursor1 = _FGBGCombinationCursor(num_fg=3, num_bg=5, batch_size=2, rng=rng1)
    cursor2 = _FGBGCombinationCursor(num_fg=3, num_bg=5, batch_size=2, rng=rng2)
    for _ in range(10):
        assert cursor1.next_batch() == cursor2.next_batch()


# --- GaussianRenderConfig.resolved_body_gaussians(variant_idx=…) ---


def test_resolved_body_gaussians_variant_idx_no_merging(monkeypatch):
    """When ``variant_idx`` is given, list values are NOT merged."""
    merged_calls: list = []

    def fake_merge(*args, **kwargs):
        merged_calls.append((args, kwargs))
        return "should_not_be_called"

    monkeypatch.setattr(gs_env, "_merge_background_plys", fake_merge)

    cfg = GaussianRenderConfig(
        body_gaussians={
            "door": ["d0.ply", "d1.ply"],
            "knob": "knob.ply",
        }
    )
    v0 = cfg.resolved_body_gaussians(variant_idx=0)
    v1 = cfg.resolved_body_gaussians(variant_idx=1)
    assert v0 == {"door": "d0.ply", "knob": "knob.ply"}
    assert v1 == {"door": "d1.ply", "knob": "knob.ply"}
    assert merged_calls == []  # variant mode never merges


def test_resolved_body_gaussians_default_still_merges_list(monkeypatch):
    merge_seen: list = []

    def fake_merge(paths, cache_subdir="gs_body_combos"):
        merge_seen.append((list(paths), cache_subdir))
        return f"merged({len(paths)})"

    monkeypatch.setattr(gs_env, "_merge_background_plys", fake_merge)

    cfg = GaussianRenderConfig(body_gaussians={"door": ["d0.ply", "d1.ply"]})
    out = cfg.resolved_body_gaussians()  # variant_idx=None → merge mode
    assert out == {"door": "merged(2)"}
    assert merge_seen == [(["d0.ply", "d1.ply"], "gs_body_combos")]


# --- Setup-time validation in BatchedGSUnifiedMujocoEnv ---


def _make_batched_for_setup(
    *,
    body_gaussians,
    background_ply,
    batch_size: int,
    foreground_variant: bool = True,
    randomize: bool = True,
):
    """Build a partially-initialized BatchedGSUnifiedMujocoEnv to exercise
    ``_setup_gs_render_state`` without touching the real env constructor."""
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env.batch_size = batch_size
    env._bg_rng = np.random.default_rng(0)
    env._bg_cache = {}
    env._pending_gs_config = None
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(
            body_gaussians=body_gaussians,
            background_ply=background_ply,
            foreground_variant=foreground_variant,
            randomize_background_on_reset=randomize,
        )
    )
    env.envs = [SimpleNamespace(model=None)]
    env.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)
    return env


def test_foreground_variant_requires_multi_bg(monkeypatch, tmp_path):
    body_ply = tmp_path / "door.ply"
    body_ply.write_bytes(b"")  # any non-empty placeholder; load_ply mocked
    env = _make_batched_for_setup(
        body_gaussians={"door": [str(body_ply)]},
        background_ply=str(body_ply),  # single-bg → not multi
        batch_size=2,
    )
    with pytest.raises(ValueError, match="multi-valued"):
        BatchedGSUnifiedMujocoEnv._setup_gs_render_state(env)


def test_foreground_variant_rejects_empty_body_gaussians(tmp_path):
    a = tmp_path / "bg0.ply"
    b = tmp_path / "bg1.ply"
    a.write_bytes(b"")
    b.write_bytes(b"")
    env = _make_batched_for_setup(
        body_gaussians={},
        background_ply=[str(a), str(b)],
        batch_size=2,
    )
    with pytest.raises(ValueError, match="non-empty body_gaussians"):
        BatchedGSUnifiedMujocoEnv._setup_gs_render_state(env)


def test_foreground_variant_rejects_when_bg_pool_below_batch_size(
    monkeypatch, tmp_path
):
    a = tmp_path / "bg0.ply"
    b = tmp_path / "bg1.ply"
    door = tmp_path / "door.ply"
    for p in (a, b, door):
        p.write_bytes(b"")
    # Two backgrounds but batch_size=4 → fail.
    env = _make_batched_for_setup(
        body_gaussians={"door": str(door)},
        background_ply=[str(a), str(b)],
        batch_size=4,
    )
    # Stub the body / bg renderer factories so we never touch real GPU code
    # before the size check fires.
    monkeypatch.setattr(
        BatchedGSUnifiedMujocoEnv,
        "_make_bg_renderer",
        lambda self, p: Mock(name=f"bg_renderer({p})"),
    )
    monkeypatch.setattr(
        BatchedGSUnifiedMujocoEnv,
        "_build_shared_mask_renderers",
        lambda self, body: {},
    )
    monkeypatch.setattr(
        gs_env, "MjxBatchSplatRenderer", lambda *a, **kw: Mock(name="MjxBatchSplat")
    )
    with pytest.raises(ValueError, match="background pool"):
        BatchedGSUnifiedMujocoEnv._setup_gs_render_state(env)


# --- Reset behavior: cursor advance vs. legacy randomize ---


def test_reset_advances_combination_cursor_in_variant_mode(monkeypatch):
    monkeypatch.setattr(
        BatchedUnifiedMujocoEnv,
        "reset",
        lambda self, env_mask=None: None,
    )
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env._is_multi_bg = True
    env._foreground_variant_mode = True
    env._pending_gs_config = None
    env._share_physics = False
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(randomize_background_on_reset=True)
    )
    env._advance_combination_cursor = Mock()
    env._randomize_env_bg_assignment = Mock()

    BatchedGSUnifiedMujocoEnv.reset(env, env_mask=None)

    env._advance_combination_cursor.assert_called_once_with()
    env._randomize_env_bg_assignment.assert_not_called()


def test_reset_no_cursor_advance_when_randomize_disabled(monkeypatch):
    monkeypatch.setattr(
        BatchedUnifiedMujocoEnv,
        "reset",
        lambda self, env_mask=None: None,
    )
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env._is_multi_bg = True
    env._foreground_variant_mode = True
    env._pending_gs_config = None
    env._share_physics = False
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(randomize_background_on_reset=False)
    )
    env._advance_combination_cursor = Mock()
    env._randomize_env_bg_assignment = Mock()

    BatchedGSUnifiedMujocoEnv.reset(env, env_mask=None)

    env._advance_combination_cursor.assert_not_called()
    env._randomize_env_bg_assignment.assert_not_called()


# --- Lazy BG renderer cache ---


def test_lookup_bg_renderer_lazy_caches_in_variant_mode():
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env._foreground_variant_mode = True
    env._bg_renderer_cache = {}
    env._bg_source_plys = ["bg0.ply", "bg1.ply", "bg2.ply"]
    calls: list[str] = []

    def fake_make_bg(p):
        calls.append(p)
        return f"renderer({p})"

    env._make_bg_renderer = fake_make_bg

    r0 = BatchedGSUnifiedMujocoEnv._lookup_bg_renderer(env, 0)
    r0b = BatchedGSUnifiedMujocoEnv._lookup_bg_renderer(env, 0)
    r2 = BatchedGSUnifiedMujocoEnv._lookup_bg_renderer(env, 2)

    assert r0 == "renderer(bg0.ply)"
    assert r0 is r0b  # cached, not rebuilt
    assert r2 == "renderer(bg2.ply)"
    # bg0 built once, bg2 built once, bg1 never built.
    assert calls == ["bg0.ply", "bg2.ply"]


def test_lookup_bg_renderer_indexes_eager_list_in_legacy_mode():
    env = object.__new__(BatchedGSUnifiedMujocoEnv)
    env._foreground_variant_mode = False
    env._bg_gs_renderers = ["r0", "r1", "r2"]

    assert BatchedGSUnifiedMujocoEnv._lookup_bg_renderer(env, 0) == "r0"
    assert BatchedGSUnifiedMujocoEnv._lookup_bg_renderer(env, 2) == "r2"


# --- Single-env class rejects foreground_variant ---


def test_single_env_rejects_foreground_variant(monkeypatch, tmp_path):
    bg0 = tmp_path / "bg0.ply"
    bg1 = tmp_path / "bg1.ply"
    door = tmp_path / "door.ply"
    for p in (bg0, bg1, door):
        p.write_bytes(b"")

    env = object.__new__(GSUnifiedMujocoEnv)
    env._bg_rng = np.random.default_rng(0)
    env._pending_gs_config = None
    env.config = SimpleNamespace(
        gaussian_render=GaussianRenderConfig(
            body_gaussians={"door": str(door)},
            background_ply=[str(bg0), str(bg1)],
            foreground_variant=True,
        )
    )

    with pytest.raises(ValueError, match="only supported on BatchedGSUnifiedMujocoEnv"):
        GSUnifiedMujocoEnv._setup_gs_render_state(env)
