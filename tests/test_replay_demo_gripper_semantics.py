from __future__ import annotations

import numpy as np

from auto_atom.runner.data_replay import _rescale_gripper_distance_to_ctrl


def test_gripper_distance_is_mapped_with_opening_semantics_preserved():
    src = np.array([0.0, 0.045, 0.09], dtype=np.float32)
    mapped = _rescale_gripper_distance_to_ctrl(
        src,
        src_range=[0.0, 0.09],
        ctrl_range=[0.0, 0.02],
    )

    np.testing.assert_allclose(mapped, [0.02, 0.01, 0.0], atol=1e-6)
