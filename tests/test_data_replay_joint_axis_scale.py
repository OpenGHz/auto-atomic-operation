import numpy as np
import pytest

from auto_atom.runner.data_replay import DataReplayConfig, _apply_joint_axis_scale


def test_joint_axis_scale_accepts_tuple_config() -> None:
    cfg = DataReplayConfig.model_validate({"joint_axis_scale": (1, 1, -1)})
    assert cfg.joint_axis_scale == [1.0, 1.0, -1.0]


def test_apply_joint_axis_scale_scales_prefix_without_mutating_input() -> None:
    data = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]],
        ],
        dtype=np.float32,
    )

    scaled = _apply_joint_axis_scale(
        data,
        [1.0, -1.0, 0.5],
        joint_names=["j1", "j2", "j3", "j4"],
        label="test replay",
    )

    expected = np.array(
        [
            [[1.0, -2.0, 1.5, 4.0], [5.0, -6.0, 3.5, 8.0]],
            [[-1.0, 2.0, -1.5, -4.0], [-5.0, 6.0, -3.5, -8.0]],
        ],
        dtype=np.float32,
    )

    assert np.allclose(scaled, expected)
    assert np.array_equal(
        data,
        np.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_apply_joint_axis_scale_rejects_scale_longer_than_action_dim() -> None:
    with pytest.raises(ValueError, match="joint_axis_scale has length 3"):
        _apply_joint_axis_scale(np.ones((2, 2), dtype=np.float32), [1.0, -1.0, 1.0])
