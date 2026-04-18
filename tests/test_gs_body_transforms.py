from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import auto_atom.basis.mjc.gs_mujoco_env as gs_env


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

    def fake_mirror(src_ply: str, axis, center):
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
