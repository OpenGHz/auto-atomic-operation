"""Wire-format (de)serialization for PolicyEvaluator data types.

Every numpy array is encoded as ``{"__ndarray__": True, "data": ..., "dtype": ...}``
so the receiving side can reconstruct it without ``allow_pickle``.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..runtime import (
    ExecutionRecord,
    ExecutionSummary,
    StageExecutionStatus,
    TaskUpdate,
)

_NDARRAY_TAG = "__ndarray__"


# ── generic recursive helpers ────────────────────────────────────────────


def serialize_value(value: Any) -> Any:
    """Recursively convert *value* to a JSON-safe wire representation."""
    if isinstance(value, np.ndarray):
        return {_NDARRAY_TAG: True, "data": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: serialize_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [serialize_value(v) for v in value]
        return converted if isinstance(value, list) else tuple(converted)
    return value


def deserialize_value(value: Any) -> Any:
    """Recursively restore numpy arrays from their tagged-dict wire format."""
    if isinstance(value, dict):
        if value.get(_NDARRAY_TAG):
            return np.array(value["data"], dtype=value["dtype"])
        return {k: deserialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [deserialize_value(v) for v in value]
        return converted if isinstance(value, list) else tuple(converted)
    return value


# ── TaskUpdate ───────────────────────────────────────────────────────────


def serialize_task_update(update: TaskUpdate) -> Dict[str, Any]:
    return {
        "stage_index": serialize_value(update.stage_index),
        "stage_name": list(update.stage_name),
        "status": [
            s.value if isinstance(s, StageExecutionStatus) else s
            for s in (
                update.status.tolist()
                if isinstance(update.status, np.ndarray)
                else update.status
            )
        ],
        "done": serialize_value(update.done),
        "success": serialize_value(update.success),
        "details": serialize_value(update.details),
        "phase": list(update.phase),
        "phase_step": serialize_value(update.phase_step),
    }


def deserialize_task_update(data: Dict[str, Any]) -> TaskUpdate:
    return TaskUpdate(
        stage_index=_to_ndarray(data["stage_index"], "int64"),
        stage_name=data["stage_name"],
        status=np.asarray(
            [StageExecutionStatus(s) for s in data["status"]], dtype=object
        ),
        done=_to_ndarray(data["done"], "bool"),
        success=np.asarray(data["success"], dtype=object),
        details=deserialize_value(data["details"]),
        phase=data["phase"],
        phase_step=_to_ndarray(data["phase_step"], "int64"),
    )


# ── ExecutionRecord ──────────────────────────────────────────────────────


def serialize_execution_record(record: ExecutionRecord) -> Dict[str, Any]:
    return {
        "env_index": record.env_index,
        "stage_index": record.stage_index,
        "stage_name": record.stage_name,
        "operator": record.operator,
        "operation": record.operation,
        "target_object": record.target_object,
        "blocking": record.blocking,
        "status": record.status.value,
        "details": serialize_value(record.details),
    }


def deserialize_execution_record(data: Dict[str, Any]) -> ExecutionRecord:
    return ExecutionRecord(
        env_index=data["env_index"],
        stage_index=data["stage_index"],
        stage_name=data["stage_name"],
        operator=data["operator"],
        operation=data["operation"],
        target_object=data["target_object"],
        blocking=data["blocking"],
        status=StageExecutionStatus(data["status"]),
        details=deserialize_value(data.get("details", {})),
    )


# ── ExecutionSummary ─────────────────────────────────────────────────────


def serialize_execution_summary(summary: ExecutionSummary) -> Dict[str, Any]:
    return {
        "total_stages": summary.total_stages,
        "max_updates": summary.max_updates,
        "updates_used": summary.updates_used,
        "completed_stage_count": serialize_value(summary.completed_stage_count),
        "final_stage_index": serialize_value(summary.final_stage_index),
        "final_stage_name": list(summary.final_stage_name),
        "final_status": [
            s.value if isinstance(s, StageExecutionStatus) else s
            for s in (
                summary.final_status.tolist()
                if isinstance(summary.final_status, np.ndarray)
                else summary.final_status
            )
        ],
        "final_done": serialize_value(summary.final_done),
        "final_success": serialize_value(summary.final_success),
        "elapsed_time_sec": summary.elapsed_time_sec,
        "records": [serialize_execution_record(r) for r in summary.records],
    }


def deserialize_execution_summary(data: Dict[str, Any]) -> ExecutionSummary:
    return ExecutionSummary(
        total_stages=data["total_stages"],
        max_updates=data["max_updates"],
        updates_used=data["updates_used"],
        completed_stage_count=_to_ndarray(data["completed_stage_count"], "int64"),
        final_stage_index=_to_ndarray(data["final_stage_index"], "int64"),
        final_stage_name=data["final_stage_name"],
        final_status=np.asarray(
            [StageExecutionStatus(s) for s in data["final_status"]], dtype=object
        ),
        final_done=_to_ndarray(data["final_done"], "bool"),
        final_success=np.asarray(data["final_success"], dtype=object),
        elapsed_time_sec=data["elapsed_time_sec"],
        records=[deserialize_execution_record(r) for r in data.get("records", [])],
    )


# ── helpers ──────────────────────────────────────────────────────────────


def _to_ndarray(value: Any, dtype: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, dict) and value.get(_NDARRAY_TAG):
        return np.array(value["data"], dtype=value["dtype"])
    return np.asarray(value, dtype=dtype)
