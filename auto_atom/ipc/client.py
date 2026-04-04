"""Lightweight rpyc client that mirrors the :class:`PolicyEvaluator` API.

Only ``rpyc``, ``numpy``, and the ``auto_atom`` data-class imports are needed
on the client side — no simulation dependencies required.

Usage::

    from auto_atom.ipc import RemotePolicyEvaluator

    evaluator = RemotePolicyEvaluator("localhost", 18861)
    evaluator.from_config("press_three_buttons", overrides=["env.batch_size=1"])
    update = evaluator.reset()
    obs = evaluator.get_observation()
    update = evaluator.update(action)
    summary = evaluator.summarize()
    evaluator.close()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rpyc
from rpyc.utils.classic import obtain

from ..runtime import ExecutionRecord, ExecutionSummary, TaskUpdate
from .serialize import (
    deserialize_execution_record,
    deserialize_execution_summary,
    deserialize_task_update,
    deserialize_value,
    serialize_value,
)
from .service import build_server_config

logger = logging.getLogger(__name__)


class RemotePolicyEvaluator:
    """Drop-in remote replacement for :class:`PolicyEvaluator`."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 18861,
        connect_timeout_sec: float = 30.0,
        retry_interval_sec: float = 0.5,
    ) -> None:
        self._call_count: int = 0
        self._conn = self._connect_with_retry(
            host, port, connect_timeout_sec, retry_interval_sec
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def from_yaml(self, path: str) -> "RemotePolicyEvaluator":
        t0 = time.perf_counter()
        self._conn.root.from_yaml(str(path))
        logger.info("from_yaml(%s) done in %.3fs", path, time.perf_counter() - t0)
        return self

    def from_config(
        self,
        config_name: str,
        overrides: Optional[List[str]] = None,
    ) -> "RemotePolicyEvaluator":
        t0 = time.perf_counter()
        self._conn.root.from_config(config_name, overrides or [])
        logger.info(
            "from_config(%s, %s) done in %.3fs",
            config_name,
            overrides,
            time.perf_counter() - t0,
        )
        return self

    def close(self) -> None:
        logger.info("close() — total RPC calls made: %d", self._call_count)
        try:
            self._conn.root.close()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    def ping(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = obtain(self._conn.root.ping())
        logger.debug("ping() -> %s (%.3fs)", result, time.perf_counter() - t0)
        return result

    # ── core API ─────────────────────────────────────────────────────

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        self._call_count += 1
        t0 = time.perf_counter()
        wire_mask = serialize_value(env_mask) if env_mask is not None else None
        result = obtain(self._conn.root.reset(wire_mask))
        update = deserialize_task_update(result)
        logger.info(
            "reset() -> stage=%s done=%s (%.3fs)",
            list(update.stage_name),
            update.done.tolist(),
            time.perf_counter() - t0,
        )
        return update

    def get_observation(self) -> Any:
        self._call_count += 1
        t0 = time.perf_counter()
        result = obtain(self._conn.root.get_observation())
        obs = deserialize_value(result)
        dt = time.perf_counter() - t0
        if isinstance(obs, dict):
            logger.debug(
                "get_observation() -> dict keys=%s (%.3fs)", list(obs.keys()), dt
            )
        else:
            logger.debug("get_observation() -> %s (%.3fs)", type(obs).__name__, dt)
        return obs

    def update(self, action: Any, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        self._call_count += 1
        t0 = time.perf_counter()
        wire_action = serialize_value(action)
        wire_mask = serialize_value(env_mask) if env_mask is not None else None
        result = obtain(self._conn.root.update(wire_action, wire_mask))
        update = deserialize_task_update(result)
        dt = time.perf_counter() - t0
        logger.debug(
            "update() #%d -> stage=%s done=%s (%.3fs)",
            self._call_count,
            list(update.stage_name),
            update.done.tolist(),
            dt,
        )
        return update

    def summarize(
        self,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
        elapsed_time_sec: float = 0.0,
    ) -> ExecutionSummary:
        self._call_count += 1
        t0 = time.perf_counter()
        result = obtain(
            self._conn.root.summarize(
                max_updates=max_updates,
                updates_used=updates_used,
                elapsed_time_sec=elapsed_time_sec,
            )
        )
        summary = deserialize_execution_summary(result)
        logger.info(
            "summarize() -> completed=%s success=%s (%.3fs)",
            summary.completed_stage_count.tolist(),
            summary.final_success.tolist(),
            time.perf_counter() - t0,
        )
        return summary

    # ── read-only properties ─────────────────────────────────────────

    @property
    def records(self) -> List[ExecutionRecord]:
        raw = obtain(self._conn.root.get_records())
        return [deserialize_execution_record(r) for r in raw]

    @property
    def batch_size(self) -> int:
        return int(self._conn.root.get_batch_size())

    @property
    def stage_plans(self) -> List[Dict[str, Any]]:
        return obtain(self._conn.root.get_stage_plans_info())

    # ── connection ───────────────────────────────────────────────────

    @staticmethod
    def _connect_with_retry(
        host: str,
        port: int,
        connect_timeout_sec: float,
        retry_interval_sec: float,
    ) -> rpyc.Connection:
        deadline = time.monotonic() + connect_timeout_sec
        last_error: Optional[Exception] = None
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                conn = rpyc.connect(host, port, config=build_server_config())
                logger.info("Connected to %s:%d (attempt %d)", host, port, attempt)
                return conn
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "Connection attempt %d to %s:%d failed: %s",
                    attempt,
                    host,
                    port,
                    exc,
                )
                time.sleep(retry_interval_sec)
        raise RuntimeError(
            f"Failed to connect to PolicyEvaluator rpyc server at "
            f"{host}:{port} within {connect_timeout_sec}s"
        ) from last_error

    def __del__(self) -> None:
        conn = self.__dict__.get("_conn")
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
