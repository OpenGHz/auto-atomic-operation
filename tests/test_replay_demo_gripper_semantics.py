from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_replay_demo_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "replay_demo.py"
    spec = importlib.util.spec_from_file_location("replay_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gripper_distance_is_mapped_with_opening_semantics_preserved():
    replay_demo = _load_replay_demo_module()

    src = np.array([0.0, 0.045, 0.09], dtype=np.float32)
    mapped = replay_demo._rescale_gripper_distance_to_ctrl(
        src,
        src_range=[0.0, 0.09],
        ctrl_range=[0.0, 0.02],
    )

    np.testing.assert_allclose(mapped, [0.02, 0.01, 0.0], atol=1e-6)
