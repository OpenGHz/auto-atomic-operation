"""Episode producers.

:class:`EvaluatorEpisodeSource` is the producer: a :class:`PolicyEvaluator`
driven by a policy, :class:`ConfigDrivenDemoPolicy` by default, so demo data
and policy data take one code path. Each env slot runs its own episode; a
slot that finishes is reset on its own and starts the next attempt while the
others keep running.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

import numpy as np

from auto_atom.config_loader import load_task_file_hydra
from auto_atom.policy_eval import (
    ConfigDrivenDemoPolicy,
    PolicyEvaluator,
    call_policy,
    default_action_applier,
    default_observation_getter,
)
from auto_atom.randomization import RandomizationFailureError
from auto_atom.runtime import (
    ComponentRegistry,
    ExecutionContext,
    StageExecutionStatus,
    TaskUpdate,
)
from auto_atom.utils.pose import PoseState

from .attempts import AttemptOutcome, AttemptScheduler, EpisodeAttempt
from .config import DEMO_POLICY, StreamConfig
from .recording import (
    EpisodeRecorder,
    missing_observation_keys,
    policy_action_row,
    split_observation,
)
from .records import Episode

PolicyFactory = Callable[[], Any]
"""Builds a policy; must be picklable (a module-level callable) to reach a
spawned worker."""


class EpisodeSource(Protocol):
    """Something that produces episodes until it is exhausted or closed."""

    def episodes(self) -> Iterator[Episode]: ...

    def close(self) -> None: ...


def resolve_policy_factory(name: str) -> PolicyFactory:
    """The factory a ``StreamConfig.policy`` name refers to."""
    if name == DEMO_POLICY:
        return ConfigDrivenDemoPolicy
    module_name, _, attribute = name.partition(":")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"Policy factory {name!r} is not callable.")
    return target


@dataclass
class _SlotRun:
    """The episode an env slot is running."""

    attempt: EpisodeAttempt
    reset_index: int
    scene: Dict[str, Any]
    randomization: Dict[str, Any]
    started: float
    initial_observation: Dict[str, Any] = field(default_factory=dict)
    recorder: EpisodeRecorder = field(default_factory=EpisodeRecorder)
    steps: int = 0


class EvaluatorEpisodeSource:
    """Roll out episodes with a :class:`PolicyEvaluator`, one per env slot.

    The evaluator is built when :meth:`episodes` starts, in the calling
    thread, which therefore owns the simulator (and its GL context) until
    :meth:`close`.
    """

    def __init__(
        self,
        config: StreamConfig,
        *,
        scheduler: Optional[AttemptScheduler] = None,
        policy: Optional[PolicyFactory] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.scheduler = scheduler or AttemptScheduler(config)
        self._policy_factory = policy or resolve_policy_factory(config.policy)
        self._stop = stop_event or threading.Event()
        self._evaluator: Optional[PolicyEvaluator] = None
        self._policy: Any = None
        self._reset_policy: Callable[[np.ndarray], None] = lambda _mask: None
        self._slots: List[Optional[_SlotRun]] = []

    def close(self) -> None:
        evaluator, self._evaluator = self._evaluator, None
        try:
            if evaluator is not None:
                evaluator.close()
        finally:
            ComponentRegistry.clear()

    def episodes(self) -> Iterator[Episode]:
        evaluator = self._open()
        batch_size = evaluator.batch_size
        policy = self._policy
        reads_observation = not isinstance(policy, ConfigDrivenDemoPolicy)
        self._slots = [None] * batch_size
        update, observation = self._start_slots(range(batch_size))

        while update is not None and any(self._slots):
            if self._stop.is_set():
                return
            active = np.asarray([run is not None for run in self._slots])
            # Idle slots (shard exhausted) read as done, so the policy leaves
            # them alone instead of drawing their first stage's actions.
            acted_on = dataclasses.replace(update, done=update.done | ~active)
            action = call_policy(
                policy, observation if reads_observation else {}, acted_on, evaluator
            )
            update = evaluator.update(action, active)

            finished: List[int] = []
            sampled: List[int] = []
            for slot in np.flatnonzero(active):
                run = self._slots[slot]
                assert run is not None
                run.steps += 1
                ended = bool(update.done[slot]) or run.steps >= self.config.max_updates
                if ended:
                    finished.append(int(slot))
                if ended or run.steps % self.config.sample_stride == 0:
                    sampled.append(int(slot))
            if sampled or reads_observation:
                observation = evaluator.get_observation()
            for slot in sampled:
                run = self._slots[slot]
                assert run is not None
                measured, commands, sim_time = split_observation(
                    observation, slot, observation_keys=self.config.observation_keys
                )
                run.recorder.append(
                    tick=run.steps - 1,
                    obs=measured,
                    action={**commands, **policy_action_row(action, slot, batch_size)},
                    update=acted_on,
                    env_index=slot,
                    sim_time=sim_time,
                )

            for slot in finished:
                run = self._slots[slot]
                self._slots[slot] = None
                assert run is not None
                episode = self._finish(slot, run, update)
                if episode is not None:
                    yield episode
                if self._stop.is_set():
                    return
            if finished:
                restarted, restart_observation = self._start_slots(finished)
                if restarted is not None:
                    update, observation = restarted, restart_observation

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def _open(self) -> PolicyEvaluator:
        if self._evaluator is not None:
            raise RuntimeError("EvaluatorEpisodeSource.episodes() is already running.")
        config = self.config
        ComponentRegistry.clear()
        task_file = load_task_file_hydra(
            config.task,
            config_dir=config.config_dir,
            overrides=[
                *config.overrides,
                f"++task.seed={config.base_seed}",
                f"++env.batch_size={config.batch_size}",
            ],
        )
        policy = self._policy_factory()
        evaluator = PolicyEvaluator(
            action_applier=getattr(policy, "action_applier", default_action_applier),
            observation_getter=getattr(
                policy, "observation_getter", default_observation_getter
            ),
        ).from_config(task_file)
        self._evaluator = evaluator
        self._policy = policy
        if evaluator.batch_size != config.batch_size:
            raise ValueError(
                f"The task built {evaluator.batch_size} env(s); the stream asked "
                f"for batch_size={config.batch_size} through env.batch_size."
            )
        self._reset_policy = _policy_resetter(policy, evaluator.batch_size)
        if config.determinism == "episode":
            raise NotImplementedError(
                "determinism='episode' needs a backend whose next reset can be "
                "given its number (AddressableResetHost); "
                f"{type(evaluator.context.backend).__name__} does not provide one. "
                "Use determinism='sequential'."
            )
        return evaluator

    # ------------------------------------------------------------------
    #  Slot lifecycle
    # ------------------------------------------------------------------

    def _start_slots(self, slots: Iterable[int]) -> Tuple[Optional[TaskUpdate], Any]:
        """Reset idle ``slots`` onto their next attempts.

        Sequential streams reset the slots together, as one batched reset;
        an attempt whose reset randomization fails is judged and its slot
        tries the next attempt. Returns the latest ``TaskUpdate`` and the
        observation after the resets, or ``(None, None)`` when no slot
        started.
        """
        evaluator = self._require_evaluator()
        waiting = list(slots)
        started_slots: List[int] = []
        latest: Optional[TaskUpdate] = None
        while waiting:
            assigned: List[Tuple[int, EpisodeAttempt]] = []
            for slot in waiting:
                attempt = self.scheduler.next_attempt()
                if attempt is None:
                    break
                assigned.append((slot, attempt))
            failed: List[int] = []
            for group in self._reset_groups(assigned):
                mask = np.zeros(evaluator.batch_size, dtype=bool)
                mask[[slot for slot, _ in group]] = True
                started = time.perf_counter()
                try:
                    update = evaluator.reset(mask)
                except RandomizationFailureError as error:
                    for slot, attempt in group:
                        self.scheduler.judge(
                            attempt,
                            AttemptOutcome(
                                kind="randomization_failed",
                                category="randomization_failed",
                                reason=str(error),
                            ),
                        )
                        failed.append(slot)
                    continue
                latest = update
                self._reset_policy(mask)
                for slot, attempt in group:
                    self._slots[slot] = _SlotRun(
                        attempt=attempt,
                        reset_index=_reset_index(evaluator.context),
                        scene=scene_ground_truth(evaluator.context, slot),
                        randomization={
                            **update.details[slot],
                            **evaluator.context.backend.get_reset_diagnostics(slot),
                        },
                        started=started,
                    )
                    started_slots.append(slot)
            waiting = failed
        if latest is None:
            return None, None
        observation = evaluator.get_observation()
        if self.config.observation_keys is not None:
            missing = missing_observation_keys(
                observation, self.config.observation_keys
            )
            if missing:
                raise KeyError(
                    f"observation_keys {missing} are not produced by the "
                    f"environment; it produces {sorted(observation)}."
                )
        for slot in started_slots:
            run = self._slots[slot]
            assert run is not None
            run.initial_observation, _, _ = split_observation(
                observation, slot, observation_keys=self.config.observation_keys
            )
        return latest, observation

    def _reset_groups(
        self, assigned: List[Tuple[int, EpisodeAttempt]]
    ) -> List[List[Tuple[int, EpisodeAttempt]]]:
        if not assigned:
            return []
        if self.config.determinism == "sequential":
            return [assigned]
        return [[item] for item in assigned]

    def _finish(
        self, slot: int, run: _SlotRun, update: TaskUpdate
    ) -> Optional[Episode]:
        """Judge a finished slot's attempt; its episode if it is yielded."""
        evaluator = self._require_evaluator()
        records = evaluator.pop_records(slot)
        done = bool(update.done[slot])
        details = update.details[slot] if update.details else {}
        if done and bool(update.success[slot]):
            kind, category, reason = "success", "succeeded", None
        elif done:
            kind = "failed"
            failed_details = next(
                (
                    record.details
                    for record in reversed(records)
                    if record.status == StageExecutionStatus.FAILED
                ),
                {},
            )
            reason = details.get("failure_reason") or failed_details.get(
                "failure_reason"
            )
            category = (
                details.get("failure_category")
                or failed_details.get("failure_category")
                or "task_failed"
            )
        else:
            kind, category = "truncated", "max_updates_reached"
            reason = (
                f"reached max_updates={self.config.max_updates} before task completion"
            )
        wall_time = time.perf_counter() - run.started
        sim_time = run.steps * float(evaluator.context.backend.dt_per_update)
        verdict = self.scheduler.judge(
            run.attempt,
            AttemptOutcome(
                kind=kind,
                category=category,
                reason=reason,
                steps=run.steps,
                wall_time=wall_time,
                sim_time=sim_time,
            ),
        )
        if not verdict.emit:
            return None
        return Episode(
            episode_index=run.attempt.episode_index,
            seed=self.config.base_seed,
            task=self.config.task,
            success=kind == "success",
            truncated=kind == "truncated",
            failure_reason=reason,
            transitions=run.recorder.arrays(),
            initial_observation=run.initial_observation,
            scene=run.scene,
            randomization=run.randomization,
            records=records,
            metadata={
                "config": self.config.model_dump(mode="json"),
                "retry": run.attempt.retry,
                "reset_index": run.reset_index,
                "worker_id": self.scheduler.worker_id,
                "num_workers": self.scheduler.num_workers,
                "slot": slot,
                "steps": run.steps,
                "wall_time": wall_time,
                "sim_time": sim_time,
                "outcome": kind,
                "failure_category": None if kind == "success" else category,
                "invalid_attempts": verdict.invalid_attempts,
            },
        )

    def _require_evaluator(self) -> PolicyEvaluator:
        if self._evaluator is None:
            raise RuntimeError("EvaluatorEpisodeSource is not running.")
        return self._evaluator


def _policy_resetter(policy: Any, batch_size: int) -> Callable[[np.ndarray], None]:
    """Reset the policy state of the slots in a mask.

    Slots reset one at a time, so with more than one slot ``policy.reset``
    must accept ``env_mask``; resetting every slot would clear the state of
    the episodes still running.
    """
    reset = getattr(policy, "reset", None)
    if reset is None:
        return lambda _mask: None
    if "env_mask" in inspect.signature(reset).parameters:
        return lambda mask: reset(env_mask=mask)
    if batch_size == 1:
        return lambda _mask: reset()
    raise TypeError(
        f"{type(policy).__name__}.reset() must accept env_mask when "
        f"batch_size={batch_size} > 1: stream slots reset independently."
    )


def _reset_index(context: ExecutionContext) -> int:
    """Number of the reset just performed: the backend's, else the runner's."""
    index = getattr(context.backend, "reset_index", None)
    return int(context.reset_count if index is None else index)


def scene_ground_truth(context: ExecutionContext, env_index: int) -> Dict[str, Any]:
    """Object, operator, and camera poses of one env, as float64 arrays."""
    backend = context.backend
    objects: Dict[str, Any] = {}
    for name in _object_names(context):
        try:
            handler = backend.get_object_handler(name)
        except KeyError:
            continue
        if handler is not None:
            objects[name] = _pose_arrays(handler.get_pose(), env_index)
    operators: Dict[str, Any] = {}
    for name in _operator_names(context):
        try:
            handler = backend.get_operator_handler(name)
        except KeyError:
            continue
        operators[name] = {
            "base": _pose_arrays(handler.get_base_pose(), env_index),
            "eef": _pose_arrays(handler.get_end_effector_pose(), env_index),
        }
    cameras = {
        name: _pose_arrays(pose, 0)
        for name, pose in backend.get_camera_poses(env_index).items()
    }
    return {"objects": objects, "operators": operators, "cameras": cameras}


def _object_names(context: ExecutionContext) -> List[str]:
    names = getattr(context.backend, "object_names", None)
    if names is not None:
        return sorted(names)
    found: Set[str] = {stage.object for stage in context.config.stages if stage.object}
    found.update(context.config.randomization.entities)
    return sorted(found)


def _operator_names(context: ExecutionContext) -> List[str]:
    names = getattr(context.backend, "operator_names", None)
    if names is not None:
        return sorted(names)
    found: Set[str] = {stage.operator for stage in context.config.stages}
    found.update(context.task_file.task_operators)
    return sorted(found)


def _pose_arrays(pose: PoseState, env_index: int) -> Dict[str, np.ndarray]:
    row = 0 if pose.batch_size == 1 else env_index
    return {
        "position": np.array(pose.position[row], dtype=np.float64),
        "orientation": np.array(pose.orientation[row], dtype=np.float64),
    }
