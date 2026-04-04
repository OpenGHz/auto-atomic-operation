"""rpyc service that wraps a :class:`PolicyEvaluator` for remote access.

The server holds all simulation dependencies; clients only need
``rpyc`` and lightweight ``auto_atom`` imports.

Usage::

    from auto_atom.ipc import serve_policy_evaluator
    serve_policy_evaluator(host="0.0.0.0", port=18861)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import rpyc
from rpyc.utils.server import ThreadedServer

from ..policy_eval import PolicyEvaluator
from ..runtime import ExecutionContext, load_task_file, load_task_file_hydra
from .serialize import (
    deserialize_value,
    serialize_execution_record,
    serialize_execution_summary,
    serialize_task_update,
    serialize_value,
)


# ── default callbacks ────────────────────────────────────────────────────


def _default_action_applier(
    context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
) -> None:
    if action is None:
        return
    if isinstance(action, dict):
        context.backend.env.apply_pose_action(
            "arm",
            action["position"],
            action["orientation"],
            action["gripper"],
        )


def _default_observation_getter(context: ExecutionContext) -> Any:
    return context.backend.env.capture_observation()


# ── rpyc config ──────────────────────────────────────────────────────────


def build_server_config(allow_public_attrs: bool = True) -> Dict[str, Any]:
    return {
        "allow_all_attrs": True,
        "allow_public_attrs": allow_public_attrs,
        "allow_pickle": True,
        "sync_request_timeout": None,
    }


# ── service factory ──────────────────────────────────────────────────────


def create_service(
    *,
    action_applier: Optional[
        Callable[[ExecutionContext, Any, Optional[np.ndarray]], None]
    ] = None,
    observation_getter: Optional[Callable[[ExecutionContext], Any]] = None,
) -> type:
    """Return an rpyc ``Service`` class that wraps a :class:`PolicyEvaluator`.

    The evaluator is created lazily when the client calls
    ``from_yaml`` or ``from_config``.
    """
    _action_applier = action_applier or _default_action_applier
    _observation_getter = observation_getter or _default_observation_getter

    class PolicyEvaluatorService(rpyc.Service):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._evaluator: Optional[PolicyEvaluator] = None

        def _require_evaluator(self) -> PolicyEvaluator:
            if self._evaluator is None:
                raise RuntimeError(
                    "PolicyEvaluator not initialized. "
                    "Call from_yaml() or from_config() first."
                )
            return self._evaluator

        # ── lifecycle ────────────────────────────────────────────────

        def exposed_ping(self) -> Dict[str, Any]:
            return {"status": "ok", "initialized": self._evaluator is not None}

        def exposed_from_yaml(self, path: str) -> None:
            self._evaluator = PolicyEvaluator(
                action_applier=_action_applier,
                observation_getter=_observation_getter,
            ).from_yaml(path)

        def exposed_from_config(
            self,
            config_name: str,
            overrides: Optional[List[str]] = None,
        ) -> None:
            task_file = load_task_file_hydra(
                config_name, overrides=list(overrides or [])
            )
            self._evaluator = PolicyEvaluator(
                action_applier=_action_applier,
                observation_getter=_observation_getter,
            ).from_config(task_file)

        def exposed_close(self) -> None:
            if self._evaluator is not None:
                self._evaluator.close()
                self._evaluator = None

        # ── core API ─────────────────────────────────────────────────

        def exposed_reset(self, env_mask: Any = None) -> Dict[str, Any]:
            mask = _wire_to_mask(env_mask)
            update = self._require_evaluator().reset(mask)
            return serialize_task_update(update)

        def exposed_get_observation(self) -> Any:
            obs = self._require_evaluator().get_observation()
            return serialize_value(obs)

        def exposed_update(self, action: Any, env_mask: Any = None) -> Dict[str, Any]:
            deserialized_action = deserialize_value(action)
            mask = _wire_to_mask(env_mask)
            update = self._require_evaluator().update(deserialized_action, mask)
            return serialize_task_update(update)

        def exposed_summarize(
            self,
            max_updates: Optional[int] = None,
            updates_used: int = 0,
            elapsed_time_sec: float = 0.0,
        ) -> Dict[str, Any]:
            summary = self._require_evaluator().summarize(
                max_updates=max_updates,
                updates_used=updates_used,
                elapsed_time_sec=elapsed_time_sec,
            )
            return serialize_execution_summary(summary)

        # ── read-only properties ─────────────────────────────────────

        def exposed_get_records(self) -> List[Dict[str, Any]]:
            return [
                serialize_execution_record(r) for r in self._require_evaluator().records
            ]

        def exposed_get_batch_size(self) -> int:
            return self._require_evaluator().batch_size

        def exposed_get_stage_plans_info(self) -> List[Dict[str, Any]]:
            plans = self._require_evaluator().stage_plans
            return [
                {
                    "stage_index": p.stage_index,
                    "stage_name": p.stage_name,
                    "operator_name": p.operator_name,
                    "operation": p.stage.operation.value,
                    "object": p.stage.object,
                }
                for p in plans
            ]

    return PolicyEvaluatorService


# ── convenience launcher ─────────────────────────────────────────────────


def serve_policy_evaluator(
    host: str = "localhost",
    port: int = 18861,
    *,
    action_applier: Optional[
        Callable[[ExecutionContext, Any, Optional[np.ndarray]], None]
    ] = None,
    observation_getter: Optional[Callable[[ExecutionContext], Any]] = None,
    protocol_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Start a blocking rpyc server for :class:`PolicyEvaluator`."""
    service_cls = create_service(
        action_applier=action_applier,
        observation_getter=observation_getter,
    )
    server = ThreadedServer(
        service_cls,
        hostname=host,
        port=port,
        protocol_config=protocol_config or build_server_config(),
    )
    print(f"PolicyEvaluator rpyc server listening on {host}:{port}")
    server.start()


# ── helpers ──────────────────────────────────────────────────────────────


def _wire_to_mask(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(deserialize_value(value), dtype=bool)
