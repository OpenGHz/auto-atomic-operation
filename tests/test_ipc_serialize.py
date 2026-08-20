from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.ipc.serialize import (
    deserialize_execution_summary,
    deserialize_task_update,
    deserialize_value,
    serialize_execution_summary,
    serialize_task_update,
    serialize_value,
)
from auto_atom.runtime import ExecutionSummary, StageExecutionStatus, TaskUpdate


def test_ndarray_binary_roundtrip_preserves_shape_and_dtype() -> None:
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)

    wire = serialize_value(value)

    assert wire["__ndarray__"] is True
    assert isinstance(wire["data"], bytes)
    assert wire["dtype"] == "float32"
    assert wire["shape"] == [3, 2, 4]

    restored = deserialize_value(wire)

    assert isinstance(restored, np.ndarray)
    assert restored.shape == (3, 2, 4)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, value)


def test_ndarray_deserialize_remains_compatible_with_legacy_list_format() -> None:
    wire = {
        "__ndarray__": True,
        "data": [[1, 2, 3], [4, 5, 6]],
        "dtype": "int64",
    }

    restored = deserialize_value(wire)

    assert isinstance(restored, np.ndarray)
    assert restored.shape == (2, 3)
    assert restored.dtype == np.int64
    np.testing.assert_array_equal(restored, np.array(wire["data"], dtype=np.int64))


def test_execution_summary_roundtrip_preserves_simulation_times() -> None:
    summary = ExecutionSummary(
        total_stages=1,
        max_updates=5,
        updates_used=1,
        timed_updates=1,
        completed_stage_count=np.asarray([1, 1], dtype=np.int64),
        final_stage_index=np.asarray([0, 0], dtype=np.int64),
        final_stage_name=["move", "move"],
        final_status=np.asarray(
            [StageExecutionStatus.SUCCEEDED, StageExecutionStatus.SUCCEEDED],
            dtype=object,
        ),
        final_done=np.asarray([True, True], dtype=bool),
        final_success=np.asarray([True, True], dtype=bool),
        sim_time_sec=1.0,
        env_completion_sim_time_sec=np.asarray([0.25, 1.0], dtype=np.float64),
    )

    restored = deserialize_execution_summary(serialize_execution_summary(summary))

    assert restored.timed_updates == 1
    assert restored.sim_time_sec == 1.0
    np.testing.assert_array_equal(
        restored.final_success,
        np.asarray([True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        restored.env_completion_sim_time_sec,
        np.asarray([0.25, 1.0], dtype=np.float64),
    )


def test_execution_summary_deserialize_defaults_timed_updates_for_legacy_data() -> None:
    summary = ExecutionSummary(
        total_stages=1,
        max_updates=5,
        updates_used=1,
        completed_stage_count=np.asarray([1], dtype=np.int64),
        final_stage_index=np.asarray([0], dtype=np.int64),
        final_stage_name=["move"],
        final_status=np.asarray([StageExecutionStatus.SUCCEEDED], dtype=object),
        final_done=np.asarray([True], dtype=bool),
        final_success=np.asarray([True], dtype=bool),
    )
    wire = serialize_execution_summary(summary)
    del wire["timed_updates"]

    restored = deserialize_execution_summary(wire)

    assert restored.timed_updates is None


def test_task_update_roundtrip_preserves_batched_success() -> None:
    update = TaskUpdate(
        stage_index=np.asarray([0, 1], dtype=np.int64),
        stage_name=["pick", "place"],
        status=np.asarray(
            [StageExecutionStatus.RUNNING, StageExecutionStatus.SUCCEEDED],
            dtype=object,
        ),
        done=np.asarray([False, True], dtype=bool),
        success=np.asarray([False, True], dtype=bool),
        phase=["pre_move", "post_move"],
        phase_step=np.asarray([0, 1], dtype=np.int64),
    )

    restored = deserialize_task_update(serialize_task_update(update))

    np.testing.assert_array_equal(restored.success, update.success)
