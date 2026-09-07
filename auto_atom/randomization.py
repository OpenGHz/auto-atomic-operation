"""Backend-neutral randomization planning and candidate-generator helpers.

The module deliberately knows nothing about MuJoCo or a particular simulator.
Backends provide pose reads/writes and constraint evaluation; this module owns
the stable action graph and deterministic candidate-generation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from .framework import (
    OperatorRandomizationConfig,
    RandomizationGeneratorConfig,
    RandomizationGeneratorInput,
    RandomizationGeneratorKind,
    RandomizationGroupConfig,
    RandomizationGroupDistributionConfig,
    RandomizationGroupGeneratorKind,
    RandomizationInput,
    RandomizationPoissonDiskConfig,
    RandomizationReference,
    RandomizationSelectorKind,
    RandomizationSpec,
    RandomizationStrategy,
    canonical_randomization_spec,
    pose_randomization_regions,
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
    groups: Mapping[str, RandomizationGroupConfig]
    """Validated joint-placement groups keyed by their task-level name."""


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


class PoissonDiskCandidateStream:
    """Persistent physical-coordinate stream backed by SciPy's Poisson disk.

    The bounds and radius are expressed in the same physical units as the
    pose proposal (metres for position axes).  The SciPy engine is kept alive
    across resets, so points rejected by a later feasibility check still
    consume the stream and cannot reappear during its lifetime.
    """

    def __init__(
        self,
        config: RandomizationPoissonDiskConfig,
        *,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        radius: float,
        seed: int,
    ) -> None:
        self._lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self._upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        if self._lower_bounds.ndim != 1 or self._lower_bounds.size == 0:
            raise ValueError("Poisson-disk position bounds must be non-empty")
        if self._upper_bounds.shape != self._lower_bounds.shape:
            raise ValueError("Poisson-disk bounds must have matching dimensions")
        if np.any(self._upper_bounds <= self._lower_bounds):
            raise ValueError("Poisson-disk position bounds must have positive extent")
        if radius <= 0.0:
            raise ValueError(
                "Poisson-disk randomization requires distribution.min_distance > 0"
            )
        if config.optimization == "lloyd":
            raise ValueError(
                "Poisson-disk optimization='lloyd' is incompatible with a "
                "persistent one-point stream; use optimization=null."
            )
        self._radius = float(radius)
        self._config = config
        self._seed = int(seed)
        self._engine = self._new_engine()

    def _new_engine(self) -> qmc.PoissonDisk:
        try:
            return qmc.PoissonDisk(
                d=int(self._lower_bounds.size),
                radius=self._radius,
                hypersphere=self._config.hypersphere,
                ncandidates=int(self._config.ncandidates),
                optimization=self._config.optimization,
                rng=self._seed,
                l_bounds=self._lower_bounds,
                u_bounds=self._upper_bounds,
            )
        except MemoryError as exc:
            raise ValueError(
                "Poisson-disk radius is too small for the physical proposal "
                "space; increase the proposal extent or reduce min_distance."
            ) from exc

    def next(self) -> np.ndarray:
        """Return the next point, failing explicitly when the space is full."""
        sample = self._engine.random(n=1)
        if len(sample) == 0:
            raise ValueError(
                "Poisson-disk proposal space is exhausted; reduce "
                "distribution.min_distance or enlarge the physical range."
            )
        return np.asarray(sample[0], dtype=np.float64).copy()


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


def _axis_interval(region: object, axis: str) -> tuple[float, float] | None:
    """Return a proposal interval, or ``None`` when the axis is unconstrained."""
    axis_range = getattr(region, "axis_range", None)
    if axis_range is None:
        return None
    value = axis_range(axis)
    if value is None:
        return None
    return float(value[0]), float(value[1])


def _regions_can_overlap(left: RandomizationSpec, right: RandomizationSpec) -> bool:
    """Conservatively detect whether two object proposal spaces can intersect."""
    for left_region in pose_randomization_regions(left):
        for right_region in pose_randomization_regions(right):
            overlaps = True
            for axis in ("x", "y", "z"):
                left_interval = _axis_interval(left_region, axis)
                right_interval = _axis_interval(right_region, axis)
                if left_interval is None or right_interval is None:
                    continue
                if (
                    left_interval[1] < right_interval[0]
                    or right_interval[1] < left_interval[0]
                ):
                    overlaps = False
                    break
            if overlaps:
                return True
    return False


def compile_randomization_plan(
    randomization: Mapping[str, RandomizationInput | OperatorRandomizationConfig],
    *,
    object_names: Set[str],
    operator_names: Set[str],
    randomization_groups: Mapping[str, RandomizationGroupConfig] | None = None,
    strategy: RandomizationStrategy = RandomizationStrategy.RSA,
) -> RandomizationPlan:
    """Compile entries and automatically derive placement components."""
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

    # Legacy groups are accepted only as an internal migration aid.  New
    # callers rely on proposal-space overlap and constraints to derive them.
    groups = dict(randomization_groups or {})
    group_members: Dict[str, str] = {}
    for group_name, group in groups.items():
        for member in group.members:
            previous_group = group_members.get(member)
            if previous_group is not None:
                raise ValueError(
                    f"Randomization group member '{member}' appears in both "
                    f"'{previous_group}' and '{group_name}'"
                )
            group_members[member] = group_name
            action = actions.get(member)
            if action is None:
                if member in operator_names:
                    raise ValueError(
                        f"Randomization group '{group_name}' member '{member}' "
                        "must be an object, not an operator"
                    )
                raise ValueError(
                    f"Randomization group '{group_name}' references unknown "
                    f"randomization member '{member}'"
                )
            if action.kind != "object":
                raise ValueError(
                    f"Randomization group '{group_name}' member '{member}' must "
                    "be an object, not an operator action"
                )
            if action.randomization.constraints.separated is not None:
                raise ValueError(
                    f"Randomization group '{group_name}' member '{member}' cannot "
                    "also declare constraints.separated"
                )
            if (
                action.randomization.distribution.selector
                != RandomizationSelectorKind.FIRST_FEASIBLE
            ):
                raise ValueError(
                    f"Randomization group '{group_name}' member '{member}' must "
                    "use distribution.selector=first_feasible; hard_sphere_rsa "
                    "owns group placement selection"
                )
            for region_index, region in enumerate(
                pose_randomization_regions(action.randomization)
            ):
                if float(region.collision_radius) <= 0.0:
                    raise ValueError(
                        f"Randomization group '{group_name}' member '{member}' "
                        f"region {region_index} requires collision_radius > 0"
                    )
                named_references = [
                    reference
                    for reference in region.references()
                    if not isinstance(reference, RandomizationReference)
                ]
                if named_references:
                    raise ValueError(
                        f"Randomization group '{group_name}' member '{member}' "
                        "cannot use named entity references"
                    )

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
    object_actions = [
        (label, action) for label, action in actions.items() if action.kind == "object"
    ]
    for index, (left_label, left_action) in enumerate(object_actions):
        for right_label, right_action in object_actions[index + 1 :]:
            if _regions_can_overlap(
                left_action.randomization, right_action.randomization
            ):
                adjacency[left_label].add(right_label)
                adjacency[right_label].add(left_label)
    for group in groups.values():
        anchor = group.members[0]
        for member in group.members[1:]:
            adjacency[anchor].add(member)
            adjacency[member].add(anchor)
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

    for group_name, group in groups.items():
        group_member_set = set(group.members)
        containing_component = next(
            (
                component
                for component in components
                if group_member_set <= set(component)
            ),
            None,
        )
        if (
            containing_component is None
            or set(containing_component) != group_member_set
        ):
            raise ValueError(
                f"Randomization group '{group_name}' must form an independent "
                "component; remove reference or separation links to non-members"
            )

    if strategy == RandomizationStrategy.RSA and not groups:
        generated_groups: Dict[str, RandomizationGroupConfig] = {}
        for component in components:
            if len(component) < 2 or not all(
                actions[label].kind == "object" for label in component
            ):
                continue
            generated_groups[f"component:{','.join(component)}"] = (
                RandomizationGroupConfig(
                    members=list(component),
                    distribution=RandomizationGroupDistributionConfig(
                        generator=RandomizationGroupGeneratorKind.HARD_SPHERE_RSA,
                    ),
                )
            )
        groups = generated_groups

    return RandomizationPlan(
        actions=actions,
        dependencies={
            label: frozenset(values) for label, values in dependencies.items()
        },
        order=tuple(order),
        components=tuple(components),
        groups=groups,
    )


def unit_candidate(
    *,
    rng: np.random.Generator,
    dimension: int,
    generator: RandomizationGeneratorInput,
    index: int,
    candidate_count: int,
    seed: int = 0,
    poisson_stream: PoissonDiskCandidateStream | None = None,
) -> np.ndarray:
    """Return one point from a SciPy QMC generator.

    Poisson disk candidates are generated by a physical-coordinate stream and
    therefore must be supplied through ``poisson_stream``. The other QMC
    generators continue to return normalized values in ``[0, 1]^dimension``.
    """
    poisson_config = (
        generator.poisson_disk
        if isinstance(generator, RandomizationGeneratorConfig)
        else None
    )
    generator_kind = (
        RandomizationGeneratorKind.POISSON_DISK
        if poisson_config is not None
        else generator
    )
    if generator_kind == RandomizationGeneratorKind.IID:
        return np.asarray(rng.random(dimension), dtype=np.float64)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if generator_kind == RandomizationGeneratorKind.LATIN_HYPERCUBE:
        engine = qmc.LatinHypercube(d=dimension, seed=seed)
        return np.asarray(
            engine.random(n=candidate_count)[index % candidate_count],
            dtype=np.float64,
        )
    if generator_kind == RandomizationGeneratorKind.HALTON:
        engine = qmc.Halton(d=dimension, scramble=True, seed=seed)
        if index:
            engine.fast_forward(index)
        return np.asarray(engine.random(n=1)[0], dtype=np.float64)
    if generator_kind == RandomizationGeneratorKind.SOBOL:
        engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
        if index:
            engine.fast_forward(index)
        return np.asarray(engine.random(n=1)[0], dtype=np.float64)
    if generator_kind == RandomizationGeneratorKind.POISSON_DISK:
        if poisson_stream is not None:
            return poisson_stream.next()
        raise ValueError(
            "Poisson-disk candidates require a physical PoissonDiskCandidateStream"
        )
    raise ValueError(f"Unsupported randomization generator: {generator_kind!r}")


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
    spacing. ``seed_points`` is an optional low-level scoring anchor; the
    backend's public selector contract keeps cross-reset history outside this
    helper. Backends may transform poses or use a different physical distance
    metric before calling it.
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
        distances = cdist(points, seed)
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
            distance = float(cdist(points[index : index + 1], points[selected]).min())
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
