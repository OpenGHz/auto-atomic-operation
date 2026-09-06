"""Backend-neutral randomization planning and candidate-sequence helpers.

The module deliberately knows nothing about MuJoCo or a particular simulator.
Backends provide pose reads/writes and constraint evaluation; this module owns
the stable action graph and deterministic candidate-generation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .framework import (
    OperatorRandomizationConfig,
    RandomizationInput,
    RandomizationReference,
    RandomizationSequenceKind,
    RandomizationSpec,
    canonical_randomization_spec,
)


@dataclass(frozen=True)
class RandomizationAction:
    """One independently sampled object, operator, or operator-attribute pose."""

    kind: str
    owner: str
    label: str
    randomization: RandomizationSpec


@dataclass(frozen=True)
class RandomizationPlan:
    """Compiled action graph used by a backend reset."""

    actions: Mapping[str, RandomizationAction]
    dependencies: Mapping[str, frozenset[str]]
    order: Tuple[str, ...]
    components: Tuple[Tuple[str, ...], ...]


class RandomizationFailureError(RuntimeError):
    """Raised when a constrained randomization cannot produce a candidate."""

    def __init__(
        self,
        *,
        target: str,
        attempts: int,
        violations: Sequence[str],
        minimum_clearance: float,
    ) -> None:
        self.target = target
        self.attempts = int(attempts)
        self.violations = tuple(str(value) for value in violations)
        self.minimum_clearance = float(minimum_clearance)
        detail = ", ".join(self.violations) or "unknown constraint"
        super().__init__(
            f"Randomization for '{target}' failed after {attempts} attempts: "
            f"{detail}; minimum_clearance={minimum_clearance:.6g}"
        )


def _parse_entity_reference(reference: str) -> tuple[str, str | None]:
    if "." in reference:
        name, attribute = reference.split(".", 1)
        if attribute in {"base", "eef"}:
            return name, attribute
    return reference, None


def _references(spec: RandomizationSpec) -> Iterable[RandomizationReference | str]:
    for region in (
        spec.proposal.regions if hasattr(spec.proposal, "regions") else (spec.proposal,)
    ):
        yield from region.references()


def compile_randomization_plan(
    randomization: Mapping[str, RandomizationInput | OperatorRandomizationConfig],
    *,
    object_names: Set[str],
    operator_names: Set[str],
) -> RandomizationPlan:
    """Compile randomization entries into a deterministic dependency graph."""
    actions: Dict[str, RandomizationAction] = {}
    declaration_order: List[str] = []
    for owner, value in randomization.items():
        if owner in operator_names and isinstance(value, OperatorRandomizationConfig):
            for attribute, kind in (("base", "operator_base"), ("eef", "operator_eef")):
                nested = getattr(value, attribute)
                if nested is None:
                    continue
                label = f"{owner}.{attribute}"
                actions[label] = RandomizationAction(
                    kind=kind,
                    owner=owner,
                    label=label,
                    randomization=canonical_randomization_spec(nested),
                )
                declaration_order.append(label)
            continue
        if owner in object_names:
            actions[owner] = RandomizationAction(
                kind="object",
                owner=owner,
                label=owner,
                randomization=canonical_randomization_spec(value),
            )
            declaration_order.append(owner)

    dependencies: Dict[str, Set[str]] = {label: set() for label in actions}
    for label, action in actions.items():
        if action.kind == "operator_eef" and f"{action.owner}.base" in actions:
            dependencies[label].add(f"{action.owner}.base")
        for reference in _references(action.randomization):
            if isinstance(reference, RandomizationReference):
                continue
            bare, attribute = _parse_entity_reference(reference)
            dependency = f"{bare}.{attribute}" if attribute else bare
            if attribute is None and f"{bare}.base" in actions:
                dependency = f"{bare}.base"
            if dependency in actions:
                dependencies[label].add(dependency)

    declaration_index = {label: index for index, label in enumerate(declaration_order)}
    order: List[str] = []
    visited: Set[str] = set()
    visiting: Set[str] = set()

    def visit(label: str) -> None:
        if label in visited:
            return
        if label in visiting:
            raise ValueError(f"Circular randomization reference involving '{label}'")
        visiting.add(label)
        for dependency in sorted(
            dependencies[label], key=declaration_index.__getitem__
        ):
            visit(dependency)
        visiting.remove(label)
        visited.add(label)
        order.append(label)

    for label in declaration_order:
        visit(label)

    adjacency: Dict[str, Set[str]] = {label: set() for label in actions}
    for label, parents in dependencies.items():
        for parent in parents:
            adjacency[label].add(parent)
            adjacency[parent].add(label)
    # Separation is a joint constraint. Connect all constrained object actions
    # so the backend samples and validates them as one component rather than
    # introducing declaration-order bias.
    separated_objects = [
        label
        for label, action in actions.items()
        if action.kind == "object"
        and action.randomization.constraints.separated is not None
    ]
    if separated_objects:
        anchor = separated_objects[0]
        # A separation declaration defines the joint randomized-object set.
        # Include unconstrained object actions too, so a single declaration
        # protects its target from every other randomized object.
        joint_objects = [
            label for label, action in actions.items() if action.kind == "object"
        ]
        for label in joint_objects[1:]:
            adjacency[anchor].add(label)
            adjacency[label].add(anchor)
    order_index = {label: index for index, label in enumerate(order)}
    components: List[Tuple[str, ...]] = []
    seen: Set[str] = set()
    for label in order:
        if label in seen:
            continue
        stack = [label]
        component: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(tuple(sorted(component, key=order_index.__getitem__)))

    return RandomizationPlan(
        actions=actions,
        dependencies={
            label: frozenset(values) for label, values in dependencies.items()
        },
        order=tuple(order),
        components=tuple(components),
    )


_HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        index, remainder = divmod(index, base)
        value += remainder * factor
        factor /= base
    return value


def unit_candidate(
    *,
    rng: np.random.Generator,
    dimension: int,
    sequence: RandomizationSequenceKind,
    index: int,
    candidate_count: int,
) -> np.ndarray:
    """Return one deterministic or pseudo-random point in ``[0, 1]^dimension``."""
    if sequence == RandomizationSequenceKind.IID:
        return np.asarray(rng.random(dimension), dtype=np.float64)
    if sequence == RandomizationSequenceKind.STRATIFIED:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        point = np.empty(dimension, dtype=np.float64)
        for axis in range(dimension):
            strata = (index + axis * 9973) % candidate_count
            point[axis] = (strata + 0.5) / candidate_count
        return point
    if sequence in {
        RandomizationSequenceKind.LOW_DISCREPANCY,
        RandomizationSequenceKind.POISSON_DISK,
    }:
        return np.asarray(
            [
                _van_der_corput(index + 1, _HALTON_PRIMES[axis % len(_HALTON_PRIMES)])
                for axis in range(dimension)
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported randomization sequence: {sequence!r}")


def maximin_select(
    candidates: np.ndarray,
    *,
    count: int,
    min_distance: float = 0.0,
    seed_points: np.ndarray | None = None,
) -> np.ndarray:
    """Select a deterministic maximin subset from candidate points.

    The first point is the candidate with the smallest lexicographic index;
    subsequent points maximize distance to the selected set. A positive
    ``min_distance`` filters candidates that cannot satisfy the requested
    spacing. This helper is intentionally metric-only; backends may transform
    poses or use a different physical distance metric before calling it.
    """
    points = np.asarray(candidates, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("candidates must have shape (N, D)")
    if count <= 0 or len(points) == 0:
        return points[:0]
    seed = None
    if seed_points is not None:
        seed = np.asarray(seed_points, dtype=np.float64)
        if seed.size == 0:
            seed = None
        elif seed.ndim != 2 or seed.shape[1] != points.shape[1]:
            raise ValueError("seed_points must have shape (M, D)")
    if seed is None:
        selected = [0]
    else:
        distances = np.linalg.norm(points[:, None, :] - seed[None, :, :], axis=2)
        scores = distances.min(axis=1)
        first = int(np.argmax(scores))
        if scores[first] < min_distance:
            return points[:0]
        selected = [first]
    remaining = set(range(len(points))) - set(selected)
    while remaining and len(selected) < count:
        best_index = None
        best_score = -np.inf
        for index in sorted(remaining):
            distance = min(
                float(np.linalg.norm(points[index] - points[j])) for j in selected
            )
            if distance < min_distance:
                continue
            if distance > best_score:
                best_score = distance
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return points[np.asarray(selected, dtype=np.int64)]
