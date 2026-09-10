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
    RandomizationGroupConfig,
    RandomizationSelectorKind,
    RandomizationSpec,
    RandomizationStrategy,
)
from auto_atom.contracts import RandomizationConstraintReport
from auto_atom.randomization import (
    CollisionParticipant,
    RandomizationAction,
    RandomizationAncestors,
    RandomizationFailureError,
    RandomizationPlan,
    copy_randomization_ancestors,
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

    def action_dependencies(self) -> Mapping[str, Set[str]]: ...

    def randomization_plan(self) -> RandomizationPlan: ...

    @property
    def batch_size(self) -> int: ...

    def template_pose(self, label: str) -> PoseState: ...

    def begin_randomization_episode(self) -> None: ...

    def apply_action(
        self,
        action: PendingRandomizationAction,
        env_mask: np.ndarray,
    ) -> None: ...

    def apply_camera_randomization(self, env_mask: np.ndarray) -> None: ...

    def run_visibility_preflight(self, env_mask: np.ndarray) -> None: ...

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
        action_specs = self._host.randomization_plan().actions
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

    def sample_hard_sphere_rsa_component_for_env(
        self,
        component: List[str],
        env_index: int,
        accepted_sampled_poses: Dict[str, PoseState],
        accepted_participants: List[CollisionParticipant],
        *,
        group_name: str,
        group: RandomizationGroupConfig | None = None,
    ) -> tuple[
        Dict[str, PoseState],
        List[PendingRandomizationAction],
        Optional[tuple[str, str]],
    ]:
        """Place group members sequentially with heterogeneous hard spheres.

        RSA is deliberately local: a rejected proposal is discarded, while
        already accepted members remain fixed for the current reset.  Member
        order is independently shuffled for every environment/reset, so the
        YAML declaration order does not become a spatial bias.
        """
        action_specs = self._host.randomization_plan().actions
        accepted_env_poses = {
            name: pose.select(env_index)
            for name, pose in accepted_sampled_poses.items()
        }
        working_poses = dict(accepted_env_poses)
        sampled_poses: Dict[str, PoseState] = {}
        actions: List[PendingRandomizationAction] = []
        local_participants: List[CollisionParticipant] = []
        dependencies = self._host.action_dependencies()
        remaining = set(component)
        order: list[str] = []
        while remaining:
            ready = [
                label
                for label in component
                if label in remaining
                and not (set(dependencies.get(label, ())) & remaining)
            ]
            if not ready:
                raise ValueError(
                    f"Circular randomization reference in RSA component {component!r}"
                )
            rng = self._host.randomization_rng
            if len(ready) > 1 and hasattr(rng, "permutation"):
                ready = [str(value) for value in rng.permutation(ready)]
            order.extend(ready)
            remaining.difference_update(ready)
        configured_attempts = [
            int(action_specs[label].randomization.constraints.failure.max_attempts)
            for label in component
        ]
        max_attempts = (
            int(group.failure.max_attempts)
            if group is not None
            else min(configured_attempts, default=DEFAULT_ATTEMPT_BUDGET)
        )
        failure_mode = (
            group.failure.mode.value
            if group is not None
            else (
                "error"
                if any(
                    action_specs[label].randomization.constraints.failure.mode.value
                    == "error"
                    for label in component
                )
                else "best_effort"
            )
        )
        selected_ancestors: Dict[str, Set[str]] = {}
        for member in order:
            action_spec = action_specs[member]
            best: tuple[float, PoseState, PendingRandomizationAction, str] | None = None
            accepted = False
            for attempt in range(max_attempts):
                candidate_poses, candidate_actions = self._host.sample_target(
                    action_spec,
                    env_index,
                    working_poses,
                    candidate_index=(
                        self._host.randomization_reset_index * 1009
                        + attempt * 17
                        + sum(ord(c) for c in member)
                    ),
                )
                candidate = next(
                    (item for item in candidate_actions if item.label == member),
                    None,
                )
                if candidate is None:
                    raise RuntimeError(
                        f"Randomization group '{group_name}' member '{member}' "
                        "did not produce a candidate"
                    )
                candidate.ancestors = reference_ancestors(
                    candidate.references,
                    selected_ancestors,
                    operator_names=self._host.operator_names,
                )
                blocking = find_collision_participant(
                    owner_name=candidate.owner,
                    env_index=env_index,
                    candidate_pose=candidate.pose,
                    collision_radius=candidate.radius,
                    ancestors=candidate.ancestors,
                    collision_participants=accepted_participants,
                )
                if blocking is None:
                    # Always-on hard-sphere collision uses the radius sum only.
                    # Optional ``separated`` clearance is enforced exactly once
                    # below through evaluate_randomization_constraints (which
                    # uses real support geometry), never folded into the radius
                    # collision test.
                    blocking = find_collision_participant(
                        owner_name=candidate.owner,
                        env_index=env_index,
                        candidate_pose=candidate.pose,
                        collision_radius=candidate.radius,
                        ancestors=candidate.ancestors,
                        collision_participants=local_participants,
                    )
                constraint_report: RandomizationConstraintReport | None = None
                if candidate.constraints is not None and (
                    candidate.constraints.visible_in is not None
                    or candidate.constraints.separated is not None
                ):
                    candidate_poses_for_constraints = dict(working_poses)
                    candidate_poses_for_constraints.update(candidate_poses)
                    candidate_poses_for_constraints[member] = candidate.pose
                    candidate_constraint_ancestors = {
                        participant.owner: resolve_collision_ancestors(
                            participant.ancestors,
                            env_index,
                        )
                        for participant in accepted_participants + local_participants
                    }
                    candidate_constraint_ancestors[candidate.owner] = set(
                        candidate.ancestors
                    )
                    constraint_report = self._host.evaluate_constraints(
                        candidate_poses_for_constraints,
                        env_index=env_index,
                        constraints=candidate.constraints,
                        ancestors=candidate_constraint_ancestors,
                        target_names={candidate.owner},
                    )
                violation = (
                    f"{candidate.label}:collides:{blocking.label}"
                    if blocking is not None
                    else (
                        constraint_report.violations[0]
                        if constraint_report is not None and not constraint_report.valid
                        else ""
                    )
                )
                candidate_row = 0 if candidate.pose.batch_size == 1 else env_index
                candidate_pos = np.asarray(
                    candidate.pose.position[candidate_row], dtype=np.float64
                )
                clearance = float("inf")
                if blocking is not None:
                    other_row = 0 if blocking.pose.batch_size == 1 else env_index
                    other_pos = np.asarray(
                        blocking.pose.position[other_row], dtype=np.float64
                    )
                    clearance = float(
                        np.linalg.norm(candidate_pos - other_pos)
                        - candidate.radius
                        - resolve_collision_radius(blocking.radius, env_index)
                    )
                elif constraint_report is not None and not constraint_report.valid:
                    clearance = float(constraint_report.minimum_clearance)
                if best is None or clearance > best[0]:
                    best = (clearance, candidate.pose, candidate, violation)
                if blocking is not None or (
                    constraint_report is not None and not constraint_report.valid
                ):
                    continue
                candidate_poses.update({member: candidate.pose})
                working_poses.update(candidate_poses)
                candidate.ancestors = set(candidate.ancestors)
                local_participants.append(
                    CollisionParticipant(
                        owner=candidate.owner,
                        label=candidate.label,
                        pose=candidate.pose,
                        radius=candidate.radius,
                        ancestors=set(candidate.ancestors),
                    )
                )
                selected_ancestors[member] = set(candidate.ancestors)
                if candidate.kind in ("object", "operator_base"):
                    selected_ancestors[candidate.owner] = set(candidate.ancestors)
                sampled_poses.update(candidate_poses)
                actions.append(candidate)
                accepted = True
                break
            if accepted:
                continue
            if best is None:
                raise RandomizationFailureError(
                    target=member,
                    attempts=max_attempts,
                    violations=[f"{member}:no_candidate"],
                    minimum_clearance=float("-inf"),
                )
            _, best_pose, best_action, violation = best
            diagnostics = {
                "group": group_name,
                "generator": (
                    group.distribution.generator.value
                    if group is not None
                    else "hard_sphere_rsa"
                ),
                "member": member,
                "attempts": max_attempts,
                "violations": [violation],
                "minimum_clearance": best[0],
                "mode": failure_mode,
            }
            self._host.record_randomization_diagnostics(env_index, diagnostics)
            if failure_mode == "error":
                raise RandomizationFailureError(
                    target=member,
                    attempts=max_attempts,
                    violations=diagnostics["violations"],
                    minimum_clearance=best[0],
                )
            self._logger.warning(
                "Hard-sphere RSA exhausted for '%s' after %d attempts; keeping "
                "the best-effort sample. violations=%s minimum_clearance=%s",
                member,
                max_attempts,
                diagnostics["violations"],
                best[0],
            )
            working_poses[member] = best_pose
            sampled_poses[member] = best_pose
            best_action.ancestors = set(best_action.ancestors)
            local_participants.append(
                CollisionParticipant(
                    owner=best_action.owner,
                    label=best_action.label,
                    pose=best_action.pose,
                    radius=best_action.radius,
                    ancestors=set(best_action.ancestors),
                )
            )
            actions.append(best_action)
        return sampled_poses, actions, None

    def apply_randomization(self, env_mask: np.ndarray) -> None:
        self._host.begin_randomization_episode()
        plan = self._host.randomization_plan()
        components = [list(component) for component in plan.components]
        action_specs = plan.actions
        hard_sphere_groups = {
            frozenset(group.members): (group_name, group)
            for group_name, group in plan.groups.items()
        }
        sampled_poses: Dict[str, PoseState] = {}
        collision_participants: List[CollisionParticipant] = []

        def apply_actions(actions: List[PendingRandomizationAction]) -> None:
            for action in actions:
                self._host.apply_action(action, env_mask)
                collision_participants.append(
                    CollisionParticipant(
                        owner=action.owner,
                        label=action.label,
                        pose=action.pose,
                        radius=action.radius,
                        ancestors=copy_randomization_ancestors(action.ancestors),
                    )
                )

        # Operators form the context for mounted cameras and object references.
        # Sampling them first makes the camera state final before visibility
        # constraints are evaluated for objects.
        operator_labels = {
            label for label, action in action_specs.items() if action.kind != "object"
        }
        for component in components:
            operator_component = [
                label for label in component if label in operator_labels
            ]
            if not operator_component:
                continue
            component_poses, component_actions = self.sample_component(
                operator_component,
                env_mask,
                sampled_poses,
                collision_participants,
            )
            apply_actions(component_actions)
            sampled_poses.update(component_poses)

        self._host.apply_camera_randomization(env_mask)

        # A ``visible_in`` object region whose whole position box is outside a
        # required camera frustum is deterministically infeasible — fail fast
        # with a diagnostic instead of exhausting the attempt loop.
        self._host.run_visibility_preflight(env_mask)

        # Object components retain reference-connected and separated joint
        # sampling, but now see the already-final operator/camera context.
        for component in components:
            object_component = [
                label for label in component if action_specs[label].kind == "object"
            ]
            if not object_component:
                continue
            hard_sphere_group = hard_sphere_groups.get(frozenset(object_component))
            component_poses, component_actions = self.sample_component(
                object_component,
                env_mask,
                sampled_poses,
                collision_participants,
                hard_sphere_rsa_group=hard_sphere_group,
                use_rsa=(
                    plan.strategy == RandomizationStrategy.RSA
                    and len(object_component) > 1
                ),
            )
            apply_actions(component_actions)
            sampled_poses.update(component_poses)

    def sample_component(
        self,
        component: List[str],
        env_mask: np.ndarray,
        accepted_sampled_poses: Dict[str, PoseState],
        accepted_participants: List[CollisionParticipant],
        hard_sphere_rsa_group: tuple[str, RandomizationGroupConfig] | None = None,
        use_rsa: bool = False,
    ) -> tuple[Dict[str, PoseState], List[PendingRandomizationAction]]:
        key_buffers = {name: self._host.template_pose(name) for name in component}
        action_buffers: Dict[str, PendingRandomizationAction] = {}
        action_order: List[str] = []

        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            if not use_rsa and hard_sphere_rsa_group is None:
                env_sampled_poses, env_actions, failure = self.sample_component_for_env(
                    component,
                    env_index,
                    accepted_sampled_poses,
                    accepted_participants,
                )
            else:
                env_sampled_poses, env_actions, failure = (
                    self.sample_hard_sphere_rsa_component_for_env(
                        component,
                        env_index,
                        accepted_sampled_poses,
                        accepted_participants,
                        group_name=(
                            hard_sphere_rsa_group[0]
                            if hard_sphere_rsa_group is not None
                            else f"component:{','.join(component)}"
                        ),
                        group=(
                            hard_sphere_rsa_group[1]
                            if hard_sphere_rsa_group is not None
                            else None
                        ),
                    )
                )
            if failure is not None:
                # Only the joint-rejection loop returns a failure tuple; the
                # RSA executor reports its own exhaustion and either raises or
                # applies the best-effort candidate itself.
                failed_label, blocking_label = failure
                self._logger.warning(
                    "Collision rejection exhausted for '%s'; keeping the "
                    "last overlapping sample against '%s'.",
                    failed_label,
                    blocking_label,
                )

            for name, pose in env_sampled_poses.items():
                if name not in key_buffers:
                    continue
                key_buffers[name].position[env_index] = pose.position[0]
                key_buffers[name].orientation[env_index] = pose.orientation[0]
            for action in env_actions:
                if action.label not in action_buffers:
                    template = self._host.template_pose(action.label)
                    buffered_radius: float | np.ndarray = float(action.radius)
                    action_ancestors = resolve_collision_ancestors(
                        action.ancestors,
                        env_index,
                    )
                    buffered_ancestors: RandomizationAncestors = set(action_ancestors)
                    if self._host.batch_size > 1:
                        buffered_radius = np.full(
                            self._host.batch_size,
                            float(action.radius),
                            dtype=np.float64,
                        )
                        buffered_ancestors = [
                            set(action_ancestors) for _ in range(self._host.batch_size)
                        ]
                    action_buffers[action.label] = PendingRandomizationAction(
                        kind=action.kind,
                        owner=action.owner,
                        label=action.label,
                        pose=template,
                        radius=buffered_radius,
                        ancestors=buffered_ancestors,
                        constraints=action.constraints,
                    )
                    action_order.append(action.label)
                action_buffers[action.label].pose.position[env_index] = (
                    action.pose.position[0]
                )
                action_buffers[action.label].pose.orientation[env_index] = (
                    action.pose.orientation[0]
                )
                if isinstance(action_buffers[action.label].radius, np.ndarray):
                    action_buffers[action.label].radius[env_index] = float(
                        action.radius
                    )
                else:
                    action_buffers[action.label].radius = float(action.radius)
                buffered_action_ancestors = action_buffers[action.label].ancestors
                env_action_ancestors = resolve_collision_ancestors(
                    action.ancestors,
                    env_index,
                )
                if isinstance(buffered_action_ancestors, list):
                    buffered_action_ancestors[env_index] = set(env_action_ancestors)
                else:
                    action_buffers[action.label].ancestors = set(env_action_ancestors)

        component_actions = [action_buffers[label] for label in action_order]
        return key_buffers, component_actions
