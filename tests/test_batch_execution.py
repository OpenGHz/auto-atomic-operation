from __future__ import annotations

import numpy as np
import pytest

from auto_atom.basis.mjc.batch_execution import (
    BatchExecutionAdapter,
    BatchExecutionMode,
)


class _FakeEnv:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[object] = []
        self.updated = False

    def capture_observation(self):
        self.calls.append("capture")
        return {
            "array": {"data": np.asarray([len(self.name)]), "t": len(self.calls)},
            "structured": {"data": {"name": self.name}, "t": len(self.calls)},
        }

    def is_updated(self) -> bool:
        self.calls.append("probe")
        value = not self.updated
        self.updated = True
        return value


def test_mask_and_row_validation() -> None:
    adapter = BatchExecutionAdapter([_FakeEnv("a"), _FakeEnv("b")], 2)

    np.testing.assert_array_equal(adapter.normalize_mask(), [True, True])
    with pytest.raises(ValueError, match="env_mask"):
        adapter.normalize_mask([True])
    with pytest.raises(ValueError, match="leading dimension"):
        adapter.broadcast_rows(np.zeros((3, 4)), label="action")


def test_replicated_dispatch_broadcasts_and_masks_rows() -> None:
    envs = [_FakeEnv("a"), _FakeEnv("b")]
    adapter = BatchExecutionAdapter(envs, 2)
    rows = adapter.broadcast_rows(np.asarray([1.0, 2.0]), label="action")

    adapter.dispatch_rows(
        lambda env, index, row: env.calls.append((index, row.tolist())),
        (rows,),
        [False, True],
    )

    assert envs[0].calls == []
    assert envs[1].calls == [(1, [1.0, 2.0])]


def test_shared_dispatch_uses_row_zero_once_when_any_row_is_active() -> None:
    env = _FakeEnv("shared")
    adapter = BatchExecutionAdapter(
        [env, env, env],
        3,
        BatchExecutionMode.SHARED,
    )
    rows = adapter.broadcast_rows(
        np.asarray([[1.0], [2.0], [3.0]]),
        label="action",
    )

    adapter.dispatch_rows(
        lambda physical, index, row: physical.calls.append((index, float(row[0]))),
        (rows,),
        [False, True, False],
    )
    adapter.dispatch(
        lambda physical, _index: physical.calls.append("unexpected"), [False] * 3
    )

    assert env.calls == [(0, 1.0)]


def test_shared_pure_map_still_processes_each_logical_row() -> None:
    env = _FakeEnv("shared")
    adapter = BatchExecutionAdapter([env, env], 2, BatchExecutionMode.SHARED)
    rows = adapter.broadcast_rows(np.asarray([[1.0], [4.0]]), label="pose")

    result = adapter.map_rows(
        lambda _env, index, row: float(row[0]) + index,
        (rows,),
    )

    assert result == [1.0, 5.0]


def test_shared_pure_map_accepts_one_physical_env() -> None:
    env = _FakeEnv("shared")
    adapter = BatchExecutionAdapter([env], 2, BatchExecutionMode.SHARED)
    rows = adapter.broadcast_rows(np.asarray([[1.0], [4.0]]), label="pose")

    result = adapter.map_rows(
        lambda physical, index, row: (physical.name, index, float(row[0])),
        (rows,),
    )

    assert result == [("shared", 0, 1.0), ("shared", 1, 4.0)]


def test_scalar_broadcast_is_available_for_unshaped_values() -> None:
    adapter = BatchExecutionAdapter([_FakeEnv("a"), _FakeEnv("b")], 2)

    np.testing.assert_array_equal(
        adapter.broadcast_rows(3.0, label="scalar"),
        np.asarray([3.0, 3.0]),
    )


def test_observation_stack_and_shared_broadcast() -> None:
    replicated = BatchExecutionAdapter([_FakeEnv("a"), _FakeEnv("bb")], 2)
    observation = replicated.capture_observation()
    np.testing.assert_array_equal(observation["array"]["data"], [[1], [2]])
    assert observation["structured"]["data"] == [{"name": "a"}, {"name": "bb"}]

    shared_env = _FakeEnv("shared")
    shared = BatchExecutionAdapter(
        [shared_env, shared_env],
        2,
        BatchExecutionMode.SHARED,
    )
    broadcast = shared.capture_observation()
    np.testing.assert_array_equal(broadcast["array"]["data"], [[6], [6]])
    assert shared_env.calls == ["capture"]


def test_shared_probe_and_physical_envs_deduplicate_aliases() -> None:
    env = _FakeEnv("shared")
    adapter = BatchExecutionAdapter([env, env, env], 3, BatchExecutionMode.SHARED)

    np.testing.assert_array_equal(
        adapter.probe_bool(lambda item: item.is_updated()), [True] * 3
    )
    assert adapter.physical_envs == [env]
    assert env.calls == ["probe"]
