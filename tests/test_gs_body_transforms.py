from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# Gaussian splatting lives in the pixi `gs` environment (`pixi run -e gs test`).
pytest.importorskip("gaussian_renderer")

import auto_atom.basis.mjc.gs_mujoco_env as gs_env
from auto_atom.basis.mjc.gs_mujoco_env import GaussianRenderConfig
from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import load_ply, save_ply


def test_body_transforms_apply_before_body_mirrors_and_preserve_shared_centers(
    monkeypatch,
) -> None:
    centroids = {
        "door_src.ply": np.asarray(
            [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]], dtype=np.float32
        ),
        "knob_src.ply": np.asarray(
            [[8.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32
        ),
        "door_xf.ply": np.asarray(
            [[11.0, 0.0, 0.0], [13.0, 0.0, 0.0]], dtype=np.float32
        ),
        "knob_xf.ply": np.asarray(
            [[14.0, 0.0, 0.0], [16.0, 0.0, 0.0]], dtype=np.float32
        ),
    }
    calls: list[tuple] = []

    def fake_load_ply(path: str):
        return SimpleNamespace(xyz=centroids[path])

    def fake_transform(src_ply: str, pose, center):
        calls.append(
            (
                "transform",
                src_ply,
                pose,
                None if center is None else tuple(float(v) for v in center),
            )
        )
        return src_ply.replace("_src", "_xf")

    def fake_mirror(src_ply: str, axis, center, post_pose):
        calls.append(
            (
                "mirror",
                src_ply,
                tuple(float(v) for v in axis),
                tuple(float(v) for v in center),
            )
        )
        return src_ply.replace(".ply", "__mirrored.ply")

    monkeypatch.setattr(gs_env, "load_ply", fake_load_ply)
    monkeypatch.setattr(gs_env, "_materialize_transformed_body_ply", fake_transform)
    monkeypatch.setattr(gs_env, "_materialize_mirrored_body_ply", fake_mirror)

    cfg = gs_env.GaussianRenderConfig(
        body_gaussians={
            "door": "door_src.ply",
            "knob": "knob_src.ply",
        },
        body_transforms={
            "door": gs_env.BodyTransformSpec(
                position=[1.0, 0.0, 0.0],
                center=[2.0, 1.0, 1.0],
            ),
            "knob": gs_env.BodyTransformSpec(
                position=[1.0, 0.0, 0.0],
                share_center_with="door",
            ),
        },
        body_mirrors={
            "door": gs_env.BodyMirrorSpec(axis=[1.0, 0.0, 0.0]),
            "knob": gs_env.BodyMirrorSpec(
                axis=[1.0, 0.0, 0.0],
                share_center_with="door",
            ),
        },
    )

    resolved = cfg.resolved_body_gaussians()

    assert resolved == {
        "door": "door_xf__mirrored.ply",
        "knob": "knob_xf__mirrored.ply",
    }
    assert calls[0] == (
        "transform",
        "door_src.ply",
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (2.0, 1.0, 1.0),
    )
    assert calls[1] == (
        "transform",
        "knob_src.ply",
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (2.0, 1.0, 1.0),
    )
    assert calls[2] == (
        "mirror",
        "door_xf.ply",
        (1.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
    )
    assert calls[3] == (
        "mirror",
        "knob_xf.ply",
        (1.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
    )


# --- list-valued body_gaussians ---


def _write_dummy_body_ply(path, offset: float) -> None:
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


def test_list_valued_body_gaussians_merges_into_single_ply(tmp_path):
    a = tmp_path / "door_panel.ply"
    b = tmp_path / "door_trim.ply"
    _write_dummy_body_ply(a, 1.0)
    _write_dummy_body_ply(b, 2.0)

    cfg = GaussianRenderConfig(body_gaussians={"door_body": [str(a), str(b)]})
    resolved = cfg.resolved_body_gaussians()

    assert set(resolved) == {"door_body"}
    merged = load_ply(resolved["door_body"])
    assert merged.xyz.shape == (2, 3)
    np.testing.assert_allclose(np.sort(merged.xyz[:, 0]), [1.0, 2.0])
    # Merged PLY lives under the body-specific cache subdir.
    assert "gs_body_combos" in resolved["door_body"]


def test_list_valued_body_gaussians_single_entry_passthrough(tmp_path):
    a = tmp_path / "panel.ply"
    _write_dummy_body_ply(a, 1.0)

    cfg = GaussianRenderConfig(body_gaussians={"door_body": [str(a)]})
    resolved = cfg.resolved_body_gaussians()

    # Single-item list short-circuits to the source path (no cache write).
    assert resolved == {"door_body": str(a)}


def test_list_valued_body_gaussians_rejects_empty_list():
    cfg = GaussianRenderConfig(body_gaussians={"door_body": []})
    try:
        cfg.resolved_body_gaussians()
    except ValueError as exc:
        assert "empty list" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty list value")


def test_list_valued_body_gaussians_runs_transforms_on_merged(monkeypatch, tmp_path):
    a = tmp_path / "panel.ply"
    b = tmp_path / "trim.ply"
    _write_dummy_body_ply(a, 1.0)
    _write_dummy_body_ply(b, 2.0)

    seen: list[str] = []

    def fake_transform(src_ply: str, pose, center):
        seen.append(src_ply)
        return src_ply + ".xf"

    monkeypatch.setattr(gs_env, "_materialize_transformed_body_ply", fake_transform)

    cfg = GaussianRenderConfig(
        body_gaussians={"door_body": [str(a), str(b)]},
        body_transforms={
            "door_body": gs_env.BodyTransformSpec(position=[1.0, 0.0, 0.0])
        },
    )
    resolved = cfg.resolved_body_gaussians()

    # body_transforms saw the *merged* PLY, not the raw source list.
    assert len(seen) == 1
    assert "gs_body_combos" in seen[0]
    assert resolved["door_body"] == seen[0] + ".xf"


def test_list_valued_body_gaussians_share_center_with_resolves(monkeypatch, tmp_path):
    a = tmp_path / "door_panel.ply"
    b = tmp_path / "door_trim.ply"
    c = tmp_path / "handle.ply"
    _write_dummy_body_ply(a, 0.0)
    _write_dummy_body_ply(b, 2.0)  # door centroid = mean = 1.0
    _write_dummy_body_ply(c, 5.0)

    captured: list[tuple] = []

    def fake_transform(src_ply: str, pose, center):
        captured.append((src_ply, tuple(float(v) for v in center)))
        return src_ply + ".xf"

    monkeypatch.setattr(gs_env, "_materialize_transformed_body_ply", fake_transform)

    cfg = GaussianRenderConfig(
        body_gaussians={
            "door_body": [str(a), str(b)],
            "handle": str(c),
        },
        body_transforms={
            "door_body": gs_env.BodyTransformSpec(
                position=[0.0, 0.0, 0.0], center=[10.0, 0.0, 0.0]
            ),
            "handle": gs_env.BodyTransformSpec(
                position=[0.0, 0.0, 0.0], share_center_with="door_body"
            ),
        },
    )
    cfg.resolved_body_gaussians()

    by_name = {entry[0]: entry[1] for entry in captured}
    door_src = next(p for p in by_name if "gs_body_combos" in p)
    handle_src = str(c)
    # door's explicit center used directly; handle reuses door's explicit center.
    assert by_name[door_src] == (10.0, 0.0, 0.0)
    assert by_name[handle_src] == (10.0, 0.0, 0.0)


def test_body_transforms_unknown_body_still_errors_with_list_form(tmp_path):
    a = tmp_path / "panel.ply"
    _write_dummy_body_ply(a, 1.0)

    cfg = GaussianRenderConfig(
        body_gaussians={"door_body": [str(a)]},
        body_transforms={"ghost": gs_env.BodyTransformSpec(position=[1.0, 0.0, 0.0])},
    )
    try:
        cfg.resolved_body_gaussians()
    except ValueError as exc:
        assert "ghost" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown body")
