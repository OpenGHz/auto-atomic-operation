"""Episode producers.

:class:`EvaluatorEpisodeSource` is the producer: a :class:`PolicyEvaluator`
driven by the config-driven demo policy, the same actions ``aao-demo`` takes.
Other policies are not supported yet. Each env slot runs its own episode; a
slot that finishes is reset on its own and starts the next attempt while the
others keep running.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any,
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
from auto_atom.contracts import AddressableResetHost
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator
from auto_atom.randomization import RandomizationFailureError
from auto_atom.runtime import (
    ComponentRegistry,
    ExecutionContext,
    StageExecutionStatus,
    TaskUpdate,
)
from auto_atom.utils.pose import PoseState
from auto_atom.utils.seed import ResetAddress

from .attempts import AttemptOutcome, AttemptScheduler, EpisodeAttempt
from .config import StreamConfig
from .recording import EpisodeRecorder, missing_observation_keys, split_observation
from .records import Episode


class EpisodeSource(Protocol):
    """Something that produces episodes until it is exhausted or closed."""

    def episodes(self) -> Iterator[Episode]: ...

    def close(self) -> None: ...


@dataclass
class _SlotRun:
    """The episode an env slot is running."""

    attempt: EpisodeAttempt
    reset_index: int
    scene: Dict[str, Any]
    randomization: Dict[str, Any]
    started: float
    pending: Optional[Tuple[Dict[str, Any], float]] = None
    """Observation (and its time) the next tick decides on, if that tick is
    recorded."""
    final_observation: Dict[str, Any] = field(default_factory=dict)
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
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.scheduler = scheduler or AttemptScheduler(config)
        self._stop = stop_event or threading.Event()
        self._evaluator: Optional[PolicyEvaluator] = None
        self._policy = ConfigDrivenDemoPolicy()
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
        policy = self._policy
        self._slots = [None] * evaluator.batch_size
        update = self._start_slots(range(evaluator.batch_size))

        while update is not None and any(self._slots):
            if self._stop.is_set():
                return
            active = np.asarray([run is not None for run in self._slots])
            # Idle slots (shard exhausted) read as done, so the policy leaves
            # them alone instead of drawing their first stage's actions.
            acted_on = dataclasses.replace(update, done=update.done | ~active)
            # The demo policy does not read observations.
            update = evaluator.update(policy.act({}, acted_on, evaluator), active)

            finished: List[int] = []
            observed: List[int] = []
            recorded: List[int] = []
            for slot in np.flatnonzero(active):
                run = self._slots[slot]
                assert run is not None
                run.steps += 1
                ended = bool(update.done[slot]) or run.steps >= self.config.max_updates
                if ended:
                    finished.append(int(slot))
                # Observe for the final observation, or before the next
                # recorded tick (tick number run.steps).
                if ended or run.steps % self.config.sample_stride == 0:
                    observed.append(int(slot))
                if run.pending is not None:
                    recorded.append(int(slot))
            observation = None
            if observed:
                self._hold_noise_except(observed)
                observation = evaluator.get_observation()
            if recorded:
                # The tick's command, which only a capture after it reports.
                commands_capture = (
                    observation if observation is not None else self._capture_commands()
                )
                for slot in recorded:
                    run = self._slots[slot]
                    assert run is not None and run.pending is not None
                    _, commands, _ = split_observation(commands_capture, slot)
                    measured, sim_time = run.pending
                    run.recorder.append(
                        tick=run.steps - 1,
                        obs=measured,
                        action=commands,
                        update=acted_on,
                        env_index=slot,
                        sim_time=sim_time,
                    )
                    run.pending = None
            for slot in observed:
                run = self._slots[slot]
                assert run is not None and observation is not None
                measured, _, sim_time = split_observation(
                    observation, slot, observation_keys=self.config.observation_keys
                )
                if slot in finished:
                    run.final_observation = measured
                else:
                    run.pending = (measured, sim_time)

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
                restarted = self._start_slots(finished)
                if restarted is not None:
                    update = restarted

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
        evaluator = PolicyEvaluator(
            action_applier=self._policy.action_applier
        ).from_config(task_file)
        self._evaluator = evaluator
        if evaluator.batch_size != config.batch_size:
            raise ValueError(
                f"The task built {evaluator.batch_size} env(s); the stream asked "
                f"for batch_size={config.batch_size} through env.batch_size."
            )
        backend = evaluator.context.backend
        if config.determinism == "episode" and not isinstance(
            backend, AddressableResetHost
        ):
            raise TypeError(
                "determinism='episode' needs a backend whose next reset can be "
                f"given its number (AddressableResetHost); {type(backend).__name__} "
                "is not one. Use determinism='sequential'."
            )
        return evaluator

    # ------------------------------------------------------------------
    #  Slot lifecycle
    # ------------------------------------------------------------------

    def _start_slots(self, slots: Iterable[int]) -> Optional[TaskUpdate]:
        """Reset idle ``slots`` onto their next attempts.

        Sequential streams reset the slots together, as one batched reset;
        an attempt whose reset randomization fails is judged and its slot
        tries the next attempt. Returns the latest ``TaskUpdate``, or
        ``None`` when no slot started.
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
                    update = evaluator.reset(mask, address=self._address(group))
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
                # Only the reset slots: the others keep the actions they drew.
                self._policy.reset(env_mask=mask)
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
            return None
        self._hold_noise_except(started_slots)
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
            measured, _, sim_time = split_observation(
                observation, slot, observation_keys=self.config.observation_keys
            )
            run.pending = (measured, sim_time)
        return latest

    def _reset_groups(
        self, assigned: List[Tuple[int, EpisodeAttempt]]
    ) -> List[List[Tuple[int, EpisodeAttempt]]]:
        """Slots reset together: all of them, or one per reset when addressed."""
        if not assigned:
            return []
        if self.config.determinism == "sequential":
            return [assigned]
        return [[item] for item in assigned]

    def _address(
        self, group: List[Tuple[int, EpisodeAttempt]]
    ) -> Optional[ResetAddress]:
        """Reset number of an episode-deterministic attempt: its index + 1."""
        if self.config.determinism == "sequential":
            return None
        ((_, attempt),) = group
        return ResetAddress(attempt.episode_index + 1, attempt.retry)

    def _capture_commands(self) -> Dict[str, Dict[str, Any]]:
        """Every row's command channels, rendering nothing when the env can.

        Without ``capture_commands`` this is a full capture that holds every
        row's camera noise, so it costs a render but changes no episode.
        """
        evaluator = self._require_evaluator()
        # CommandObservationEnvProtocol, checked by attribute: isinstance()
        # would cache a class's answer.
        capture_commands = getattr(evaluator.get_env(), "capture_commands", None)
        if capture_commands is not None:
            return capture_commands()
        self._hold_noise_except([])
        return evaluator.get_observation()

    def _hold_noise_except(self, slots: List[int]) -> None:
        """Keep the next capture from advancing the camera noise of other slots.

        An episode-deterministic stream counts a slot's noise frames over the
        captures made for it (its first observation, the observation before
        each further recorded tick, and its final observation), not the ones
        made for other slots.
        """
        if self.config.determinism == "sequential":
            return
        evaluator = self._require_evaluator()
        hold = getattr(evaluator.get_env(), "hold_camera_noise", None)
        if hold is None:
            return
        mask = np.ones(evaluator.batch_size, dtype=bool)
        mask[list(slots)] = False
        if mask.any():
            hold(mask)

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
            final_observation=run.final_observation,
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
