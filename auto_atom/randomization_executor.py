"""Backend-neutral randomization execution.

The executor owns the *policy* half of randomization: the feasibility loop, its
attempt budget, cross-reset coverage history, the ``maximin`` candidate-group
selection, and the fail-closed vs. best-effort decision. A backend implements
:class:`RandomizationHost` — pose reads/writes, baselines, named-frame
resolution, camera models, and support geometry — and inherits the semantics
unchanged.

Reproducibility contract: the executor draws from the host's RNG in a fixed
order and derives only a ``sample_index`` from the host's reset counter, so a
given seed produces the same reset sequence in any backend that supplies the
same baselines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Container,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

import numpy as np

from auto_atom.config.randomization import (
    RandomizationConstraintConfig,
    RandomizationFailureConfig,
    RandomizationSelectorKind,
    RandomizationSpec,
)
from auto_atom.contracts import RandomizationConstraintReport
from auto_atom.randomization import (
    CollisionParticipant,
    RandomizationAction,
    RandomizationAncestors,
    RandomizationFailureError,
    distribution_uses_space_filling_history,
    find_collision_participant,
    history_clearance,
    maximin_select,
    reference_ancestors,
    resolve_collision_ancestors,
    resolve_collision_radius,
)
from auto_atom.utils.pose import PoseState

DEFAULT_ATTEMPT_BUDGET = RandomizationFailureConfig().max_attempts
"""Attempt budget for a component whose specs leave ``failure.max_attempts``
at its default. A configured spec may request any value; the tightest one wins."""


@dataclass
class PendingRandomizationAction:
    """One sampled pose waiting to be written into the scene.

    ``radius`` and ``ancestors`` may be batched (per-environment) because
    auto-resolved radii and reference ancestors can differ between
    environments. ``constraints`` is the hard-constraint set that this action's
    candidate must satisfy, if any.
    """

    kind: str
    owner: str
    label: str
    pose: PoseState
    radius: Any
    references: Tuple[Any, ...] = ()
    ancestors: RandomizationAncestors = field(default_factory=set)
    constraints: Optional[RandomizationConstraintConfig] = None


@runtime_checkable
class RandomizationHost(Protocol):
    """The simulator-specific surface the randomization executor needs."""

    @property
    def randomization_rng(self) -> np.random.Generator: ...

    @property
    def randomization_reset_index(self) -> int: ...

    @property
    def object_names(self) -> Container[str]: ...

    @property
    def operator_names(self) -> Container[str]: ...

    def action_specs(self) -> Mapping[str, RandomizationAction]: ...

    def sample_target(
        self,
        action: RandomizationAction,
        env_index: int,
        working_poses: Dict[str, PoseState],
        *,
        candidate_index: int,
    ) -> Tuple[Dict[str, PoseState], List[PendingRandomizationAction]]: ...

    def evaluate_constraints(
        self,
        candidate_poses: Mapping[str, PoseState],
        *,
        env_index: int,
        constraints: Optional[RandomizationConstraintConfig],
        ancestors: Optional[Mapping[str, RandomizationAncestors]],
        target_names: Optional[Any],
    ) -> RandomizationConstraintReport: ...

    def record_randomization_diagnostics(
        self,
        env_index: int,
        diagnostics: Mapping[str, Any],
    ) -> None: ...


class RandomizationExecutor:
    """Feasibility loop, coverage history, and candidate selection."""

    def __init__(
        self,
        host: RandomizationHost,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._host = host
        self._logger = logger or logging.getLogger(__name__)
        self._history: Dict[Tuple[str, ...], List[np.ndarray]] = {}

    @property
    def history(self) -> Dict[Tuple[str, ...], List[np.ndarray]]:
        """Accepted cross-reset samples, keyed by component.

        The history is intentionally *not* cleared per reset: it is the
        persistent sequence that makes non-IID generators cover the proposal
        volume across episodes instead of restarting every reset.
        """
        return self._history

    def clear_history(self) -> None:
        """Drop the cross-reset coverage history."""
        self._history.clear()

    def _record_history(self, history_key: Tuple[str, ...], vector: np.ndarray) -> None:
        """Record only the pose selected/applied for a reset component."""
        history = self._history.get(history_key, [])
        history = (history + [np.asarray(vector, dtype=np.float64).copy()])[-256:]
        self._history[history_key] = history

    def sample_component_for_env(
        self,
        component: List[str],
        env_index: int,
        accepted_sampled_poses: Dict[str, PoseState],
        accepted_participants: List[CollisionParticipant],
    ) -> tuple[
        Dict[str, PoseState],
        List[PendingRandomizationAction],
        Optional[tuple[str, str]],
    ]:
        accepted_env_poses = {
            name: pose.select(env_index)
            for name, pose in accepted_sampled_poses.items()
        }
        last_sampled_poses: Dict[str, PoseState] = {}
        last_actions: List[PendingRandomizationAction] = []
        last_failure: Optional[tuple[str, str]] = None
        last_violations: List[str] = []
        last_minimum_clearance = float("inf")
        best_failure: Optional[
            tuple[
                tuple[int, float],
                Dict[str, PoseState],
                List[PendingRandomizationAction],
                Optional[tuple[str, str]],
                List[str],
                float,
            ]
        ] = None
        action_specs = self._host.action_specs()
        configured_attempts = [
            int(action_specs[label].randomization.constraints.failure.max_attempts)
            for label in component
        ]
        # The budget is the tightest configured ``failure.max_attempts``; an
        # unconfigured component falls back to the config default.
        attempt_budget = min(configured_attempts, default=DEFAULT_ATTEMPT_BUDGET)
        # Candidate generation owns cross-reset coverage.  The selector only
        # decides how this reset's feasible candidate group is reduced to one
        # sample; it must not decide whether accepted history is consulted.
        history_enabled = any(
            distribution_uses_space_filling_history(
                action_specs[label].randomization.distribution
            )
            for label in component
        )
        collect_candidate_group = any(
            action_specs[label].randomization.distribution.selector
            == RandomizationSelectorKind.MAXIMIN
            for label in component
        )
        history_key = tuple(sorted(component))
        history = self._history.get(history_key, [])
        history_min_distance = max(
            (
                float(action_specs[label].randomization.distribution.spacing)
                for label in component
                if distribution_uses_space_filling_history(
                    action_specs[label].randomization.distribution
                )
            ),
            default=0.0,
        )
        candidate_limit = max(
            (
                int(action_specs[label].randomization.distribution.candidate_count)
                for label in component
            ),
            default=1,
        )
        valid_candidates: list[
            tuple[Dict[str, PoseState], List[PendingRandomizationAction], np.ndarray]
        ] = []
        stable_labels = {label: index for index, label in enumerate(sorted(component))}
        for attempt in range(attempt_budget):
            working_poses = dict(accepted_env_poses)
            env_sampled_poses: Dict[str, PoseState] = {}
            env_actions: List[PendingRandomizationAction] = []
            env_participants: List[CollisionParticipant] = []
            selected_ancestors: Dict[str, Set[str]] = {}
            failure: Optional[tuple[str, str]] = None
            violations: List[str] = []
            minimum_clearance = float("inf")
            for action_label in component:
                sampled_poses, actions = self._host.sample_target(
                    action_specs[action_label],
                    env_index,
                    working_poses,
                    candidate_index=(
                        self._host.randomization_reset_index * 1009
                        + attempt * 17
                        + stable_labels[action_label]
                    ),
                )
                for key, pose in sampled_poses.items():
                    working_poses[key] = pose
                    env_sampled_poses[key] = pose
                for action in actions:
                    if not action.references:
                        raise ValueError(
                            f"Sampled action '{action.label}' has no references"
                        )
                    action.ancestors = reference_ancestors(
                        action.references,
                        selected_ancestors,
                        operator_names=self._host.operator_names,
                    )
                    action_ancestors = set(action.ancestors)
                    selected_ancestors[action.label] = action_ancestors
                    if action.kind in ("object", "operator_base"):
                        selected_ancestors[action.owner] = action_ancestors
                    blocking = find_collision_participant(
                        owner_name=action.owner,
                        env_index=env_index,
                        candidate_pose=action.pose,
                        collision_radius=action.radius,
                        ancestors=action.ancestors,
                        collision_participants=accepted_participants + env_participants,
                    )
                    env_actions.append(action)
                    env_participants.append(
                        CollisionParticipant(
                            owner=action.owner,
                            label=action.label,
                            pose=action.pose,
                            radius=action.radius,
                            ancestors=set(action.ancestors),
                        )
                    )
                    if failure is None and blocking is not None:
                        failure = (action.label, blocking.label)
                        violations.append(f"{action.label}:collides:{blocking.label}")
                        candidate_row = 0 if action.pose.batch_size == 1 else env_index
                        other_row = 0 if blocking.pose.batch_size == 1 else env_index
                        candidate_position = np.asarray(
                            action.pose.position[candidate_row], dtype=np.float64
                        )
                        other_position = np.asarray(
                            blocking.pose.position[other_row], dtype=np.float64
                        )
                        minimum_clearance = min(
                            minimum_clearance,
                            float(
                                np.linalg.norm(candidate_position - other_position)
                                - float(action.radius)
                                - resolve_collision_radius(
                                    blocking.radius,
                                    env_index,
                                )
                            ),
                        )
            constraint_groups: dict[
                RandomizationConstraintConfig, list[PendingRandomizationAction]
            ] = {}
            for action in env_actions:
                if action.constraints is None or (
                    action.constraints.visible_in is None
                    and action.constraints.separated is None
                ):
                    continue
                constraint_groups.setdefault(action.constraints, []).append(action)
            for constraints, constrained_actions in constraint_groups.items():
                constrained_names = {action.owner for action in constrained_actions}
                candidate_poses = {
                    name: pose
                    for name, pose in accepted_env_poses.items()
                    if name in self._host.object_names
                }
                candidate_poses.update(
                    {
                        action.owner: action.pose
                        for action in env_actions
                        if action.kind == "object"
                    }
                )
                candidate_ancestors = {
                    participant.owner: resolve_collision_ancestors(
                        participant.ancestors,
                        env_index,
                    )
                    for participant in accepted_participants
                    if participant.owner in candidate_poses
                }
                candidate_ancestors.update(
                    {
                        action.owner: set(action.ancestors)
                        for action in env_actions
                        if action.kind == "object" and action.owner in candidate_poses
                    }
                )
                report = self._host.evaluate_constraints(
                    candidate_poses,
                    env_index=env_index,
                    constraints=constraints,
                    ancestors=candidate_ancestors,
                    target_names=constrained_names,
                )
                if not report.valid and failure is None:
                    failure = (constrained_actions[0].label, report.violations[0])
                if not report.valid:
                    violations.extend(report.violations)
                    minimum_clearance = min(
                        minimum_clearance,
                        float(report.minimum_clearance),
                    )
            vector: np.ndarray | None = None
            if failure is None and (history_enabled or collect_candidate_group):
                vector = np.concatenate(
                    [
                        np.asarray(
                            next(
                                action.pose
                                for action in env_actions
                                if action.label == label
                            ).position[0],
                            dtype=np.float64,
                        )
                        for label in sorted(component)
                    ]
                )
                if history_enabled:
                    # Distance from this candidate to the nearest sample
                    # accepted in an earlier reset.
                    nearest_history_gap = history_clearance(vector, history)
                    duplicate = nearest_history_gap <= 1e-12
                    if duplicate or nearest_history_gap < history_min_distance:
                        failure = (component[0], "history:min_distance")
                        violations.append(f"{component[0]}:history:min_distance")
                        required = max(history_min_distance, 1e-12)
                        minimum_clearance = min(
                            minimum_clearance,
                            nearest_history_gap - required,
                        )

            last_sampled_poses = env_sampled_poses
            last_actions = env_actions
            last_failure = failure
            last_violations = violations
            last_minimum_clearance = minimum_clearance
            if failure is None:
                if not collect_candidate_group:
                    if history_enabled and vector is not None:
                        self._record_history(history_key, vector)
                    return env_sampled_poses, env_actions, None
                if vector is None:
                    raise RuntimeError(
                        "Randomization candidate group was not vectorized"
                    )
                valid_candidates.append((env_sampled_poses, env_actions, vector))
                if len(valid_candidates) >= candidate_limit:
                    break
            else:
                unique_violations = list(dict.fromkeys(violations))
                score = (len(unique_violations), -minimum_clearance)
                if best_failure is None or score < best_failure[0]:
                    best_failure = (
                        score,
                        env_sampled_poses,
                        env_actions,
                        failure,
                        unique_violations,
                        minimum_clearance,
                    )

        if valid_candidates:
            vectors = np.vstack([candidate[2] for candidate in valid_candidates])
            distribution = action_specs[component[0]].randomization.distribution
            selected = maximin_select(
                vectors,
                count=1,
                min_distance=float(distribution.spacing),
                # History is prepared independently of the selector.  It is
                # the scoring baseline for maximin within this reset's group.
                seed_points=np.vstack(history) if history else None,
            )
            if len(selected) == 0:
                selected = maximin_select(
                    vectors,
                    count=1,
                    min_distance=0.0,
                    seed_points=np.vstack(history) if history else None,
                )
            selected_index = int(
                np.flatnonzero(np.all(np.isclose(vectors, selected[0]), axis=1))[0]
            )
            chosen_poses, chosen_actions, chosen_vector = valid_candidates[
                selected_index
            ]
            if history_enabled:
                self._record_history(history_key, chosen_vector)
            return chosen_poses, chosen_actions, None

        if best_failure is not None:
            (
                _,
                best_poses,
                best_actions,
                best_failure_reason,
                best_violations,
                best_clearance,
            ) = best_failure
            error_actions = [
                action_specs[label]
                for label in component
                if action_specs[label].randomization.constraints.failure.mode.value
                == "error"
            ]
            diagnostics = {
                "labels": list(component),
                "attempts": attempt_budget,
                "violations": best_violations,
                "minimum_clearance": best_clearance,
                "mode": "error" if error_actions else "best_effort",
            }
            self._host.record_randomization_diagnostics(env_index, diagnostics)
            if error_actions:
                raise RandomizationFailureError(
                    target=error_actions[0].label,
                    attempts=attempt_budget,
                    violations=diagnostics["violations"],
                    minimum_clearance=best_clearance,
                )
            self._logger.warning(
                "Randomization constraints exhausted for '%s' after %d attempts; "
                "applying best-effort candidate. violations=%s minimum_clearance=%s",
                component[0],
                attempt_budget,
                diagnostics["violations"],
                best_clearance,
            )
            if history_enabled:
                best_vector = np.concatenate(
                    [
                        np.asarray(
                            next(
                                action.pose
                                for action in best_actions
                                if action.label == label
                            ).position[0],
                            dtype=np.float64,
                        )
                        for label in sorted(component)
                    ]
                )
                self._record_history(history_key, best_vector)
            return best_poses, best_actions, best_failure_reason
        return last_sampled_poses, last_actions, last_failure
