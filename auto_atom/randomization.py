"""Backend-neutral randomization planning, candidate generation, and constraint
evaluation.

The module deliberately knows nothing about MuJoCo or a particular simulator.
Backends provide pose reads/writes, camera models, and support geometry; this
module owns the stable action graph, the deterministic candidate-generation
semantics, and the feasibility arithmetic (camera frustum projection and
inter-entity separation) used to accept or reject candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from auto_atom.config.randomization import (
    OperatorRandomizationConfig,
    PoseRandomizationSpec,
    PoseRandomRange,
    RandomizationConstraintConfig,
    RandomizationDistributionConfig,
    RandomizationFailureMode,
    RandomizationGeneratorConfig,
    RandomizationGeneratorInput,
    RandomizationGeneratorKind,
    RandomizationGroupConfig,
    RandomizationGroupDistributionConfig,
    RandomizationGroupGeneratorKind,
    RandomizationInput,
    RandomizationPoissonDiskConfig,
    RandomizationSelectorKind,
    RandomizationSpec,
    RandomizationStrategy,
    RandomizationVisibilityConfig,
    canonical_randomization_spec,
    pose_randomization_regions,
)
from auto_atom.config.reference import RandomizationReference
from auto_atom.contracts import (
    CameraModel,
    RandomizationConstraintReport,
    SupportGeometry,
)
from auto_atom.utils.pose import (
    PoseState,
    euler_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_rpy,
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
                "Poisson-disk randomization requires distribution.spacing > 0"
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
                "space; increase the proposal extent or reduce spacing."
            ) from exc

    def next(self) -> np.ndarray:
        """Return the next point, failing explicitly when the space is full."""
        sample = self._engine.random(n=1)
        if len(sample) == 0:
            raise ValueError(
                "Poisson-disk proposal space is exhausted; reduce "
                "distribution.spacing or enlarge the physical range."
            )
        return np.asarray(sample[0], dtype=np.float64).copy()


def parse_entity_reference(reference: str) -> tuple[str, str | None]:
    """Split an entity-name reference into ``(name, attribute)``.

    ``'arm.base'`` → ``('arm', 'base')``; ``'arm.eef'`` → ``('arm', 'eef')``;
    plain ``'vase'`` → ``('vase', None)``. Any other dotted form is returned
    unchanged, so it stays an opaque named-frame reference.
    """
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
                if float(region.collision_radius) == 0.0:
                    raise ValueError(
                        f"Randomization group '{group_name}' member '{member}' "
                        f"region {region_index} requires collision_radius > 0 "
                        "or an auto radius (< 0); 0 (exempt) has no radius to place"
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
            bare, attribute = parse_entity_reference(reference)
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


_RandomizationAncestors = Union[Set[str], List[Set[str]]]


@dataclass
class CollisionParticipant:
    """One already-accepted pose that later candidates must keep clear of.

    ``radius`` and ``ancestors`` may be scalar (shared by every environment) or
    batched (a per-environment value), because auto-resolved operator radii and
    per-environment reference ancestors differ between environments.
    """

    owner: str
    label: str
    pose: PoseState
    radius: Union[float, np.ndarray]
    ancestors: _RandomizationAncestors = field(default_factory=set)


def resolve_collision_radius(radius: Union[float, np.ndarray], env_index: int) -> float:
    """Resolve a scalar or batched collision radius for one environment."""
    if isinstance(radius, np.ndarray):
        values = np.asarray(radius, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return 0.0
        if values.size == 1:
            return float(values[0])
        return float(values[env_index])
    return float(radius)


def resolve_collision_ancestors(
    ancestors: _RandomizationAncestors,
    env_index: int,
) -> Set[str]:
    """Resolve scalar or batched reference ancestors for one environment."""
    if isinstance(ancestors, list):
        if not ancestors:
            return set()
        if len(ancestors) == 1:
            return ancestors[0]
        return ancestors[env_index]
    return ancestors


def find_collision_participant(
    *,
    owner_name: str,
    env_index: int,
    candidate_pose: PoseState,
    collision_radius: float,
    ancestors: Set[str],
    collision_participants: Sequence[CollisionParticipant],
    extra_clearance: float = 0.0,
) -> Optional[CollisionParticipant]:
    """Return the first accepted participant closer than the radius sum.

    The test is a pure radius sum: object size is carried by the participants'
    radii, so ``extra_clearance`` is a surface gap. A non-positive candidate
    radius means the candidate is exempt from collision rejection, and
    parent/child pairs are skipped so an articulated assembly is never treated
    as self-colliding.
    """
    candidate_radius = float(collision_radius)
    if candidate_radius <= 0.0:
        return None
    candidate_row = 0 if candidate_pose.batch_size == 1 else env_index
    candidate_pos = np.asarray(candidate_pose.position[candidate_row], dtype=np.float64)
    for participant in collision_participants:
        participant_radius = resolve_collision_radius(participant.radius, env_index)
        if participant_radius <= 0.0:
            continue
        if participant.owner == owner_name:
            continue
        participant_ancestors = resolve_collision_ancestors(
            participant.ancestors,
            env_index,
        )
        if participant.owner in ancestors or owner_name in participant_ancestors:
            continue
        other_row = 0 if participant.pose.batch_size == 1 else env_index
        other_pos = np.asarray(participant.pose.position[other_row], dtype=np.float64)
        if np.linalg.norm(candidate_pos - other_pos) < (
            candidate_radius + participant_radius + float(extra_clearance)
        ):
            return participant
    return None


def distribution_uses_space_filling_history(distribution: object) -> bool:
    """Return whether a generator participates in cross-reset coverage.

    ``selector`` is intentionally absent from this decision: selectors operate
    on the current candidate group only, while non-IID generators own the
    persistent sequence semantics that make accepted samples from earlier
    resets unavailable to later resets.
    """
    generator = getattr(distribution, "generator", RandomizationGeneratorKind.IID)
    return generator != RandomizationGeneratorKind.IID


def history_clearance(vector: np.ndarray, history: Sequence[np.ndarray]) -> float:
    """Return distance from a candidate to its nearest accepted sample."""
    if not history:
        return float("inf")
    points = np.vstack(history)
    return float(
        np.linalg.norm(
            np.asarray(vector, dtype=np.float64) - points,
            axis=1,
        ).min()
    )


def select_randomization_region(
    rng: np.random.Generator,
    spec: RandomizationInput,
    *,
    candidate_index: int = 0,
) -> PoseRandomRange:
    """Select one region from a possibly multi-region randomization spec.

    A wrapper's regions are selected once per sampling attempt so rejection
    retries naturally draw a fresh region. ``equal`` weighting is a uniform
    draw; ``volume`` weighting draws proportionally to the proposal box volume.
    The single ``PoseRandomRange`` path consumes no random value, which keeps
    legacy streams bit-identical.
    """
    canonical = canonical_randomization_spec(spec)
    regions = pose_randomization_regions(canonical)
    if not regions:
        raise ValueError("Randomization region lists must not be empty")
    if len(regions) == 1:
        return regions[0]
    if canonical.distribution.region_weighting == "volume":
        volumes = []
        for region in regions:
            volume = 1.0
            for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
                axis_range = region.axis_range(axis)
                if axis_range is not None:
                    volume *= max(float(axis_range[1]) - float(axis_range[0]), 0.0)
            volumes.append(volume)
        total_volume = float(sum(volumes))
        if total_volume > 0.0:
            sampled = float(rng.uniform(0.0, total_volume))
            cumulative = 0.0
            for region, volume in zip(regions, volumes):
                cumulative += volume
                if sampled <= cumulative:
                    return region
    sampled_index = int(rng.uniform(0.0, float(len(regions))))
    return regions[max(0, min(sampled_index, len(regions) - 1))]


def sample_pose_for_env(
    rng: np.random.Generator,
    *,
    base_pose: PoseState,
    rand_range: PoseRandomRange,
    env_index: int,
    batch_size: int,
    reference_poses: Optional[
        Mapping[Union[RandomizationReference, str], PoseState]
    ] = None,
    distribution: Optional[RandomizationDistributionConfig] = None,
    sample_index: int = 0,
    reset_index: int = 0,
    poisson_stream: Optional[PoissonDiskCandidateStream] = None,
) -> PoseState:
    """Sample one environment's pose from an axis range.

    Each axis either keeps its baseline (an unconfigured axis), receives an
    absolute value (``absolute_world`` / ``absolute_base``) or is added to the
    baseline (a relative or entity-name reference). The baseline per axis comes
    from that axis's own reference, so a single region can mix frames.

    Random draws happen in a fixed order — one per configured position axis
    (x, y, z), then one per configured rotation axis (roll, pitch, yaw) — so the
    stream stays reproducible across resets and refactors. Non-IID generators
    substitute low-discrepancy values for the uniform draws instead of
    consuming them.
    """
    base_pose = base_pose.broadcast_to(batch_size)
    pose_by_reference = {
        reference: pose.broadcast_to(batch_size)
        for reference, pose in (reference_poses or {}).items()
    }

    def _baseline(reference: Union[RandomizationReference, str]) -> PoseState:
        return pose_by_reference.get(reference, base_pose)

    generator = getattr(distribution, "generator", RandomizationGeneratorKind.IID)
    is_poisson_generator = generator == RandomizationGeneratorKind.POISSON_DISK or (
        isinstance(generator, RandomizationGeneratorConfig)
    )
    candidate_count = int(getattr(distribution, "candidate_count", 1))
    poisson_position_axes = (
        tuple(
            axis for axis in ("x", "y", "z") if rand_range.axis_range(axis) is not None
        )
        if poisson_stream is not None
        else ()
    )
    poisson_values = poisson_stream.next() if poisson_stream is not None else None
    qmc_values = None
    if generator != RandomizationGeneratorKind.IID:
        orientation_generator = (
            RandomizationGeneratorKind.SOBOL if is_poisson_generator else generator
        )
        qmc_values = unit_candidate(
            rng=rng,
            dimension=3 if is_poisson_generator else 6,
            generator=orientation_generator,
            index=sample_index,
            candidate_count=candidate_count,
            seed=int(reset_index * 1_000_003 + sample_index),
        )

    position = np.empty(3, dtype=np.float64)
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        reference = rand_range.axis_reference(axis_name)
        baseline = _baseline(reference)
        value = float(baseline.position[env_index, axis_index])
        rng_pair = rand_range.axis_range(axis_name)
        if rng_pair is not None:
            if poisson_values is not None:
                sampled = float(poisson_values[poisson_position_axes.index(axis_name)])
            elif qmc_values is None:
                sampled = float(rng.uniform(*rng_pair))
            else:
                sampled = float(
                    rng_pair[0] + qmc_values[axis_index] * (rng_pair[1] - rng_pair[0])
                )
            if reference in (
                RandomizationReference.ABSOLUTE_WORLD,
                RandomizationReference.ABSOLUTE_BASE,
            ):
                value = sampled
            else:
                value += sampled
        position[axis_index] = value

    rotation_axes = ("roll", "pitch", "yaw")
    rotation_references = tuple(
        rand_range.axis_reference(axis_name) for axis_name in rotation_axes
    )
    if (
        all(rand_range.axis_range(axis_name) is None for axis_name in rotation_axes)
        and len(set(rotation_references)) == 1
    ):
        orientation = np.asarray(
            _baseline(rotation_references[0]).orientation[env_index],
            dtype=np.float64,
        ).copy()
        return PoseState(position=position, orientation=orientation)

    rotation = np.empty(3, dtype=np.float64)
    for axis_index, axis_name in enumerate(rotation_axes):
        reference = rand_range.axis_reference(axis_name)
        baseline = _baseline(reference)
        baseline_rpy = quaternion_to_rpy(baseline.orientation[env_index])
        value = float(baseline_rpy[axis_index])
        rng_pair = rand_range.axis_range(axis_name)
        if rng_pair is not None:
            if qmc_values is None:
                sampled = float(rng.uniform(*rng_pair))
            else:
                sampled = float(
                    rng_pair[0]
                    + qmc_values[axis_index if is_poisson_generator else 3 + axis_index]
                    * (rng_pair[1] - rng_pair[0])
                )
            if reference in (
                RandomizationReference.ABSOLUTE_WORLD,
                RandomizationReference.ABSOLUTE_BASE,
            ):
                value = sampled
            else:
                value += sampled
        rotation[axis_index] = value
    orientation = np.asarray(euler_to_quaternion(tuple(rotation)), dtype=np.float64)
    return PoseState(position=position, orientation=orientation)


def sample_pose_batch(
    rng: np.random.Generator,
    *,
    base_pose: PoseState,
    rand_range: PoseRandomRange,
    env_mask: np.ndarray,
    batch_size: int,
    distribution: Optional[RandomizationDistributionConfig] = None,
    reset_index: int = 0,
) -> PoseState:
    """Sample one pose per enabled environment into a batched ``PoseState``."""
    base_pose = base_pose.broadcast_to(batch_size)
    position = base_pose.position.copy()
    orientation = base_pose.orientation.copy()
    for env_index, enabled in enumerate(env_mask):
        if not enabled:
            continue
        sampled = sample_pose_for_env(
            rng,
            base_pose=base_pose,
            rand_range=rand_range,
            env_index=env_index,
            batch_size=batch_size,
            distribution=distribution,
            sample_index=reset_index * 1009 + env_index,
            reset_index=reset_index,
        )
        position[env_index] = sampled.position[0]
        orientation[env_index] = sampled.orientation[0]
    return PoseState(position=position, orientation=orientation)


@dataclass(frozen=True)
class VisibilityInfeasibility:
    """A ``visible_in`` region that provably cannot be satisfied for one env.

    ``violations`` mirrors the message format used by the feasibility loop so a
    fail-fast diagnostic is indistinguishable from an exhausted-retry one apart
    from ``attempts == 0``.
    """

    target: str
    env_index: int
    violations: Tuple[str, ...]


def reference_ancestors(
    references: Sequence[Union[RandomizationReference, str]],
    selected_ancestors: Mapping[str, Set[str]],
    *,
    operator_names: Container[str],
) -> Set[str]:
    """Return the entity names a target's references depend on.

    A bare entity reference denotes an operator base when the name belongs to an
    operator, matching the shorthand used elsewhere in the randomization plan.
    """
    ancestors: Set[str] = set()
    for reference in references:
        if isinstance(reference, RandomizationReference):
            continue
        bare, attribute = parse_entity_reference(reference)
        if attribute is None and bare in operator_names:
            attribute = "base"
        reference_key = f"{bare}.{attribute}" if attribute is not None else bare
        ancestors.add(bare)
        ancestors.update(selected_ancestors.get(reference_key, ()))
    return ancestors


def validate_pose_randomization_spec(
    label: str,
    spec: PoseRandomizationSpec,
    *,
    allow_absolute_base: bool,
    object_names: Container[str],
    operator_names: Container[str],
) -> None:
    """Validate every region of one target against its reference context.

    ``absolute_base`` is only meaningful for an operator end effector, and a
    region must not mix it with other frames because the axes would then be
    resolved against unrelated origins. Named references must resolve to a known
    object or operator (``<operator>.base`` / ``<operator>.eef``).
    """
    for region_index, region in enumerate(pose_randomization_regions(spec)):
        references = region.references()
        if RandomizationReference.ABSOLUTE_BASE in references:
            if not allow_absolute_base:
                raise ValueError(
                    f"{label} randomization region {region_index} cannot use "
                    "'absolute_base' — only operator end-effector "
                    "randomization is defined in a base frame."
                )
            if any(
                reference != RandomizationReference.ABSOLUTE_BASE
                for reference in references
            ):
                raise ValueError(
                    f"{label} randomization region {region_index} cannot mix "
                    "'absolute_base' with references in other frames."
                )
        for reference in references:
            if isinstance(reference, RandomizationReference):
                continue
            bare, attribute = parse_entity_reference(reference)
            if attribute is not None and bare not in operator_names:
                raise ValueError(
                    f"{label} randomization region {region_index} reference "
                    f"'{reference}' uses '.{attribute}', but '{bare}' is not a "
                    "known operator."
                )
            if (
                attribute is None
                and bare not in object_names
                and bare not in operator_names
            ):
                raise ValueError(
                    f"{label} randomization region {region_index} reference "
                    f"'{reference}' is not a known object or operator."
                )


def validate_randomization_configuration(
    randomization: Mapping[str, RandomizationInput | OperatorRandomizationConfig],
    *,
    object_names: Container[str],
    operator_names: Container[str],
) -> Tuple[str, ...]:
    """Validate target-specific reference rules for every configured region.

    Returns the keys that match neither an object nor an operator so the caller
    can report them in its own logging channel instead of silently ignoring a
    typo.
    """
    unknown: List[str] = []
    for name, spec in randomization.items():
        if name in object_names:
            if isinstance(spec, OperatorRandomizationConfig):
                raise TypeError(
                    f"Object '{name}' randomization must use a direct "
                    "single- or multi-region pose specification, not an "
                    "operator randomization config."
                )
            validate_pose_randomization_spec(
                f"Object '{name}'",
                spec,
                allow_absolute_base=False,
                object_names=object_names,
                operator_names=operator_names,
            )
            continue
        if name not in operator_names:
            unknown.append(name)
            continue
        if not isinstance(spec, OperatorRandomizationConfig):
            raise TypeError(
                f"Operator '{name}' randomization must use the nested form "
                "with explicit `base:` and/or `eef:` sub-entries (i.e. an "
                "OperatorRandomizationConfig). Direct pose randomization "
                "specifications are not supported."
            )
        if spec.base is not None:
            validate_pose_randomization_spec(
                f"Operator '{name}' base",
                spec.base,
                allow_absolute_base=False,
                object_names=object_names,
                operator_names=operator_names,
            )
        if spec.eef is not None:
            validate_pose_randomization_spec(
                f"Operator '{name}' end effector",
                spec.eef,
                allow_absolute_base=True,
                object_names=object_names,
                operator_names=operator_names,
            )
    return tuple(unknown)


def camera_frustum_disjoint_box(
    camera: CameraModel,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> bool:
    """Return True when a world AABB lies entirely outside a camera frustum.

    Deterministic-infeasibility predicate for ``visible_in``: if every possible
    center position of an entity is outside a camera's view frustum, no sampled
    pose can ever be visible in that camera, so retrying the feasibility loop is
    futile. The test is conservative (it never reports a false infeasibility): a
    box is only declared fully outside when it lies beyond one frustum boundary,
    and pixel margins are deliberately ignored (a box outside the full frustum
    is certainly outside any margin-shrunk one).
    """
    lower = np.asarray(box_min, dtype=np.float64)
    upper = np.asarray(box_max, dtype=np.float64)
    camera_pos = np.asarray(camera.pose.position[0], dtype=np.float64)
    rotation = quaternion_to_rotation_matrix(camera.pose.orientation[0])
    # Camera frame: x right, y up, z backward (MuJoCo). A point is visible
    # when depth d = -z is within [near, far] and |x|, |y| <= d * tan(fov).
    corners = [
        rotation.T @ (np.asarray([ix, iy, iz], dtype=np.float64) - camera_pos)
        for ix in (lower[0], upper[0])
        for iy in (lower[1], upper[1])
        for iz in (lower[2], upper[2])
    ]
    cam_points = np.asarray(corners, dtype=np.float64)
    cam_min = cam_points.min(axis=0)
    cam_max = cam_points.max(axis=0)
    near = float(camera.near)
    far = float(camera.far)
    if near >= far:
        return True
    # Depth bounds (z_cam valid in [-far, -near]).
    if cam_min[2] > -near or cam_max[2] < -far:
        return True
    tan_hfy = np.tan(float(camera.fovy_radians) / 2.0)
    aspect = float(camera.width) / float(camera.height) if camera.height > 0 else 1.0
    tan_hfx = tan_hfy * aspect
    # Side bounds (valid x in [z*tan, -z*tan]; y analogously).
    if cam_max[0] - tan_hfx * cam_min[2] < 0.0:  # wholly left of frustum
        return True
    if cam_min[0] + tan_hfx * cam_min[2] > 0.0:  # wholly right
        return True
    if cam_max[1] - tan_hfy * cam_min[2] < 0.0:  # wholly below
        return True
    return bool(cam_min[1] + tan_hfy * cam_min[2] > 0.0)  # wholly above


def object_region_world_box(
    region: PoseRandomRange,
    default_world: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """World AABB of one region's possible center positions, or ``None``.

    Only world-axis-aligned position ranges (``absolute_world`` or ``relative``
    to the fixed default pose) are supported; entity-tracked or otherwise
    frame-dependent references return ``None`` so the caller falls back to the
    normal retry loop.
    """
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    for axis, index in (("x", 0), ("y", 1), ("z", 2)):
        reference = region.axis_reference(axis)
        if not isinstance(reference, RandomizationReference):
            return None  # entity-tracked reference
        if reference not in (
            RandomizationReference.RELATIVE,
            RandomizationReference.ABSOLUTE_WORLD,
        ):
            return None
        base = float(default_world[index])
        axis_range = region.axis_range(axis)
        if axis_range is None:
            lower[index] = base
            upper[index] = base
            continue
        low, high = float(axis_range[0]), float(axis_range[1])
        if reference == RandomizationReference.RELATIVE:
            low += base
            high += base
        lower[index] = min(low, high)
        upper[index] = max(low, high)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        return None
    return lower, upper


def find_visibility_infeasibility(
    targets: Mapping[str, RandomizationSpec],
    *,
    env_mask: np.ndarray,
    default_pose_of: Callable[[str], Optional[PoseState]],
    camera_names_of: Callable[[RandomizationVisibilityConfig, int], Sequence[str]],
    camera_model_of: Callable[[str, int], CameraModel],
) -> Optional[VisibilityInfeasibility]:
    """Return the first provably empty ``visible_in`` region, if any.

    A ``visible_in`` target is deterministic given fixed cameras and its own
    geometry: if the entity's whole possible position box is outside one of the
    required camera frustums, no candidate can ever satisfy the constraint.
    Detecting that up front avoids burning ``max_attempts`` on a futile search.
    Only world-axis-aligned position regions are checked (conservative); other
    reference modes fall through to the retry loop. Targets whose failure policy
    is not fail-closed are left to the loop, which may still apply a
    best-effort candidate.
    """
    for name, spec in targets.items():
        constraints = spec.constraints
        if constraints is None or constraints.visible_in is None:
            continue
        visible = constraints.visible_in
        if constraints.failure.mode != RandomizationFailureMode.ERROR:
            continue
        default_pose = default_pose_of(name)
        if default_pose is None:
            continue
        regions = pose_randomization_regions(spec)
        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            default_world = np.asarray(
                default_pose.select(env_index).position, dtype=np.float64
            ).reshape(3)
            camera_names = camera_names_of(visible, env_index)
            for region in regions:
                box = object_region_world_box(region, default_world)
                if box is None:
                    continue
                lower, upper = box
                for camera_name in camera_names:
                    camera = camera_model_of(camera_name, env_index)
                    if not camera_frustum_disjoint_box(camera, lower, upper):
                        continue
                    return VisibilityInfeasibility(
                        target=name,
                        env_index=env_index,
                        violations=(f"{name}:outside_view:{camera_name}",),
                    )
    return None


class RandomizationConstraintEvaluator:
    """Feasibility oracle for camera visibility and inter-entity separation.

    This owns every part of candidate acceptance that is not simulator-specific:    the frustum projection arithmetic, the separation clearance arithmetic, and
    the per-episode caches that keep the rejection loop cheap. A backend only
    supplies two reads — a camera model and a support geometry per entity — so a
    new backend inherits ``visible_in`` / ``separated`` semantics unchanged.

    The two reads are passed per call rather than bound at construction because
    a backend resolves them through its own (possibly monkeypatched) accessors.

    Camera models and support radii are pose-invariant within one episode, so
    they are resolved once per episode instead of once per candidate. Call
    :meth:`reset` from the backend's low-level reset so camera-pose
    randomization and reconfiguration are picked up.
    """

    def __init__(self) -> None:
        self._camera_models: MutableMapping[str, CameraModel] = {}
        self._support_radii: MutableMapping[str, float] = {}

    def reset(self) -> None:
        """Drop the per-episode caches."""
        self._camera_models.clear()
        self._support_radii.clear()

    def _camera_model(
        self,
        camera_name: str,
        camera_model_of: Callable[[str], CameraModel],
    ) -> CameraModel:
        cached = self._camera_models.get(camera_name)
        if cached is None:
            cached = camera_model_of(camera_name)
            self._camera_models[camera_name] = cached
        return cached

    def _support_radius(
        self,
        entity_name: str,
        support_geometry_of: Callable[[str], SupportGeometry],
    ) -> float:
        """Conservative support radius of one entity for the current episode.

        The bounding-sphere radius is measured around the entity's own geom
        centroid, so it is invariant to the proposed pose; caching it avoids a
        full geometry refresh on every candidate evaluation.
        """
        cached = self._support_radii.get(entity_name)
        if cached is None:
            cached = float(support_geometry_of(entity_name).radius)
            self._support_radii[entity_name] = cached
        return cached

    def evaluate(
        self,
        candidate_poses: Mapping[str, PoseState],
        constraints: Optional[RandomizationConstraintConfig] = None,
        *,
        camera_model_of: Callable[[str], CameraModel],
        support_geometry_of: Callable[[str], SupportGeometry],
        all_camera_names: Sequence[str] = (),
        ancestors: Optional[Mapping[str, Set[str]]] = None,
        target_names: Optional[Set[str]] = None,
    ) -> RandomizationConstraintReport:
        """Check one candidate set against the configured hard constraints.

        ``candidate_poses`` maps logical entity names to the *proposed* poses;
        only translation is used, because both the visibility bound and the
        separation test are sphere-based. ``target_names`` restricts the
        visibility check to the entity currently being sampled, ``ancestors``
        exempts articulated descendants from the separation test, and
        ``all_camera_names`` is the backend's camera set that
        ``visible_in.cameras: all`` refers to.
        """
        if constraints is None:
            return RandomizationConstraintReport(valid=True)

        violations: List[str] = []
        minimum_clearance = float("inf")

        if constraints.visible_in is not None:
            violations.extend(
                self._visibility_violations(
                    candidate_poses,
                    constraints.visible_in,
                    camera_model_of=camera_model_of,
                    support_geometry_of=support_geometry_of,
                    all_camera_names=all_camera_names,
                    target_names=target_names,
                )
            )

        separated = constraints.separated
        if separated is not None and separated.scope == "scene":
            raise NotImplementedError(
                "scene-scope separation requires geom-level broad-phase support, "
                "which is not implemented yet."
            )
        if separated is not None and separated.scope == "randomized":
            minimum_clearance = self._separation_violations(
                candidate_poses,
                separated.clearance,
                support_geometry_of=support_geometry_of,
                ancestors=ancestors,
                violations=violations,
            )

        return RandomizationConstraintReport(
            valid=not violations,
            violations=tuple(violations),
            minimum_clearance=minimum_clearance,
        )

    def _visibility_violations(
        self,
        candidate_poses: Mapping[str, PoseState],
        visible: Any,
        *,
        camera_model_of: Callable[[str], CameraModel],
        support_geometry_of: Callable[[str], SupportGeometry],
        all_camera_names: Sequence[str],
        target_names: Optional[Set[str]],
    ) -> List[str]:
        if visible.mode.value != "frustum":
            raise NotImplementedError(
                "segmentation visibility requires a renderer-backed "
                "implementation, which is not available yet."
            )
        if visible.geometry.value == "support_hull":
            raise NotImplementedError(
                "support-hull visibility requires backend-provided support "
                "points, which are not available yet."
            )
        if visible.cameras == "all":
            camera_names = list(all_camera_names)
            if not camera_names:
                raise ValueError(
                    "visible_in.cameras: all needs the backend's camera set, "
                    "but none was provided — refusing to skip the visibility "
                    "check silently."
                )
        else:
            camera_names = list(visible.cameras)
        candidates = (
            candidate_poses
            if target_names is None
            else {
                name: pose
                for name, pose in candidate_poses.items()
                if name in target_names
            }
        )
        # Camera models and projection constants depend only on the camera and
        # are fixed for the whole feasibility loop — resolve them once instead
        # of once per (entity, camera) pair.
        camera_data = []
        for camera_name in camera_names:
            camera = self._camera_model(camera_name, camera_model_of)
            cam_pos = np.asarray(camera.pose.position[0], dtype=np.float64)
            rotation = np.asarray(
                quaternion_to_rotation_matrix(camera.pose.orientation[0]),
                dtype=np.float64,
            )
            half_fovy = camera.fovy_radians / 2.0
            half_fovx = np.arctan(np.tan(half_fovy) * (camera.width / camera.height))
            camera_data.append(
                (
                    camera_name,
                    cam_pos,
                    rotation,
                    camera.width * 0.5 / np.tan(half_fovx),
                    camera.height * 0.5 / np.tan(half_fovy),
                    float(camera.width),
                    float(camera.height),
                    float(camera.near),
                    float(camera.far),
                    float(visible.margin_px),
                )
            )
        use_bounding_sphere = visible.geometry.value == "bounding_sphere"
        violations: List[str] = []
        for entity_name, pose in candidates.items():
            center = np.asarray(pose.position[0], dtype=np.float64)
            radius = (
                self._support_radius(entity_name, support_geometry_of)
                if use_bounding_sphere
                else 0.0
            )
            for (
                camera_name,
                cam_pos,
                rotation,
                scale_x,
                scale_y,
                cam_width,
                cam_height,
                near,
                far,
                margin,
            ) in camera_data:
                # Camera x-axis is right, y-axis up, z-axis backward, so the
                # depth is the negated camera-frame z coordinate.
                camera_point = rotation.T @ (center - cam_pos)
                depth = -float(camera_point[2])
                depth_margin = radius / max(depth, 1e-9)
                px = cam_width * 0.5 + camera_point[0] / max(depth, 1e-9) * scale_x
                py = cam_height * 0.5 - camera_point[1] / max(depth, 1e-9) * scale_y
                if depth <= near + radius or depth >= far - radius:
                    violations.append(f"{entity_name}:outside_depth:{camera_name}")
                if (
                    px - depth_margin * cam_width < margin
                    or px + depth_margin * cam_width > cam_width - margin
                    or py - depth_margin * cam_height < margin
                    or py + depth_margin * cam_height > cam_height - margin
                ):
                    violations.append(f"{entity_name}:outside_view:{camera_name}")
        return violations

    def _separation_violations(
        self,
        candidate_poses: Mapping[str, PoseState],
        clearance: float,
        *,
        support_geometry_of: Callable[[str], SupportGeometry],
        ancestors: Optional[Mapping[str, Set[str]]],
        violations: List[str],
    ) -> float:
        """Append pairwise separation violations and return the tightest gap."""
        names = list(candidate_poses)
        minimum_clearance = float("inf")
        for index, left_name in enumerate(names):
            left_center = np.asarray(
                candidate_poses[left_name].position[0], dtype=np.float64
            )
            left_radius = self._support_radius(left_name, support_geometry_of)
            for right_name in names[index + 1 :]:
                if ancestors and (
                    right_name in ancestors.get(left_name, set())
                    or left_name in ancestors.get(right_name, set())
                ):
                    continue
                right_center = np.asarray(
                    candidate_poses[right_name].position[0], dtype=np.float64
                )
                gap = (
                    float(np.linalg.norm(left_center - right_center))
                    - left_radius
                    - self._support_radius(right_name, support_geometry_of)
                    - float(clearance)
                )
                minimum_clearance = min(minimum_clearance, gap)
                if gap < 0.0:
                    violations.append(f"{left_name}:collides:{right_name}")
        return minimum_clearance
