"""Backend-neutral randomization execution.

The executor owns randomization: it compiles the task's config into a plan,
selects regions, resolves per-axis references, samples from the configured
generators, resolves auto collision radii, runs both retry loops and the
deterministic ``visible_in`` preflight, orders the reset, and writes the result
back through the host.

A backend implements :class:`RandomizationHost` — a capability surface of pose
reads and writes, recorded baselines, element names, support geometry, camera
models, and the diagnostics sink — and inherits every semantic above unchanged.
Nothing on that surface knows what a randomization is, so a new backend supports
the whole layer by providing pose access alone.

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
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

from auto_atom.config.randomization import (
    PoseRandomRange,
    RandomizationConstraintConfig,
    RandomizationDistributionConfig,
    RandomizationFailureConfig,
    RandomizationGeneratorConfig,
    RandomizationGeneratorKind,
    RandomizationGroupConfig,
    RandomizationInput,
    RandomizationPoissonDiskConfig,
    RandomizationSelectorKind,
    RandomizationStrategy,
    RandomizationVisibilityConfig,
    ResolvedRandomizationConfig,
    ResolvedRandomizationScope,
    canonical_randomization_spec,
)
from auto_atom.config.reference import RandomizationReference
from auto_atom.contracts import RandomizationConstraintReport
from auto_atom.randomization import (
    CollisionParticipant,
    PoissonDiskCandidateStream,
    RandomizationAction,
    RandomizationAncestors,
    RandomizationFailureError,
    RandomizationPlan,
    compile_randomization_plan,
    copy_randomization_ancestors,
    distribution_uses_space_filling_history,
    find_collision_participant,
    find_visibility_infeasibility,
    history_clearance,
    maximin_select,
    parse_entity_reference,
    reference_ancestors,
    resolve_collision_ancestors,
    resolve_collision_radius,
    sample_pose_batch,
    sample_pose_for_env,
    select_randomization_region,
    validate_randomization_configuration,
)
from auto_atom.utils.pose import PoseState, compose_pose, inverse_pose

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
    """The simulator surface the randomization executor needs.

    Every member is a **capability**: read or write the pose of a scene
    element, read a recorded baseline, read support/camera geometry, report
    diagnostics. Nothing here knows what a randomization is, which is what lets
    a new backend support the whole randomization layer by implementing pose
    access alone.
    """

    @property
    def batch_size(self) -> int: ...

    @property
    def object_names(self) -> Container[str]: ...

    @property
    def operator_names(self) -> Container[str]: ...

    @property
    def rng(self) -> np.random.Generator:
        """The generator the executor draws candidate streams from."""
        ...

    @property
    def seed(self) -> Optional[int]:
        """The seed the generator was built from, or ``None`` when unseeded."""
        ...

    @property
    def episode_index(self) -> int:
        """How many resets have happened; used to derive per-episode sample indices."""
        ...

    def live_pose(self, label: str) -> PoseState:
        """Current world pose of a target label.

        ``label`` is an object name, or ``<operator>.base`` /
        ``<operator>.eef`` for an operator's base body / home end-effector.
        """
        ...

    def baseline_pose(self, label: str) -> Optional[PoseState]:
        """Recorded reset baseline for a target label, or ``None`` if unrecorded."""
        ...

    def get_camera_pose(self, camera_name: str) -> PoseState: ...

    def set_camera_pose(
        self,
        camera_name: str,
        pose: PoseState,
        env_mask: np.ndarray,
    ) -> None: ...

    def camera_names(self) -> List[str]:
        """Names of the cameras this scene actually has."""
        ...

    def get_camera_model(self, camera_name: str, env_index: int) -> Any: ...

    def get_support_geometry(self, entity_name: str, env_index: int) -> Any: ...

    def get_operator_support_geometry(
        self,
        operator_name: str,
        part: str,
        env_index: int,
    ) -> Any: ...

    def set_target_pose(
        self,
        kind: str,
        owner: str,
        pose: PoseState,
        env_mask: np.ndarray,
    ) -> None:
        """Write one element part's pose into the scene for the masked envs.

        ``kind`` is ``object``, ``operator_base`` or ``operator_eef``: the same
        element addressing ``live_pose`` reads with.
        """
        ...

    def evaluate_constraints(
        self,
        candidate_poses: Mapping[str, PoseState],
        *,
        env_index: int,
        constraints: Optional[RandomizationConstraintConfig],
        ancestors: Optional[Mapping[str, RandomizationAncestors]],
        target_names: Optional[Any],
    ) -> RandomizationConstraintReport: ...

    def record_reset_diagnostics(
        self,
        env_index: int,
        diagnostics: Mapping[str, Any],
    ) -> None:
        """Store one reset diagnostic entry (why a placement was hard)."""
        ...


class RandomizationExecutor:
    """Feasibility loop, coverage history, and candidate selection."""

    def __init__(
        self,
        host: RandomizationHost,
        config: Optional[ResolvedRandomizationConfig] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._host = host
        self._config = config or ResolvedRandomizationConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._history: Dict[Tuple[str, ...], List[np.ndarray]] = {}
        self._poisson_streams: Dict[Tuple[object, ...], PoissonDiskCandidateStream] = {}
        # Auto-resolved collision radii are cached per (kind, owner, env). Object
        # entries stay valid for the backend lifetime (static geometry); operator
        # entries are dropped every episode (their geometry follows the episode
        # configuration) — see ``begin_episode``.
        self._auto_radius_cache: Dict[Tuple[Any, ...], float] = {}

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

    # ------------------------------------------------------------------
    #  Plan and configuration
    # ------------------------------------------------------------------

    @property
    def config(self) -> ResolvedRandomizationConfig:
        """The task-level randomization config this executor applies."""
        return self._config

    @property
    def scope(self) -> ResolvedRandomizationScope:
        """The resolved per-entity / per-camera entries plus the strategy."""
        return self._config.scope

    @property
    def plan(self) -> RandomizationPlan:
        """The compiled action graph.

        Compilation is a pure function of the config and the host's element
        names, so it is re-derived on demand: a host may register elements after
        construction, and the plan is cheap to rebuild.
        """
        scope = self._config.scope
        return compile_randomization_plan(
            scope.entities,
            object_names=set(self._host.object_names),
            operator_names=set(self._host.operator_names),
            randomization_groups=(
                self._config.groups
                if scope.strategy == RandomizationStrategy.RSA
                else {}
            ),
            strategy=scope.strategy,
        )

    def action_dependencies(self) -> Dict[str, Set[str]]:
        """Reference dependency edges between the plan's actions."""
        return {
            label: set(dependencies)
            for label, dependencies in self.plan.dependencies.items()
        }

    @property
    def camera_randomization(self) -> Mapping[str, RandomizationInput]:
        """Configured per-camera pose randomization entries."""
        return self.scope.cameras

    # ------------------------------------------------------------------
    #  Episode preparation
    # ------------------------------------------------------------------

    def begin_episode(self) -> None:
        """Prepare one reset before any pose is sampled.

        Operator auto radii depend on the episode's configuration (base/EEF
        geometry follows the home state just applied), so operator entries are
        dropped and re-resolved each reset while object auto radii (static
        geometry) stay cached. Configured regions are validated here so a bad
        reference fails before the scene is mutated.
        """
        self._auto_radius_cache = {
            key: value
            for key, value in self._auto_radius_cache.items()
            if key[0] == "object"
        }
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Validate target-specific rules for every configured region.

        Public because a bad reference should surface when a task is
        configured or inspected, not only when a reset first samples the
        entry that uses it.
        """
        unknown = validate_randomization_configuration(
            self.scope.entities,
            object_names=set(self._host.object_names),
            operator_names=set(self._host.operator_names),
        )
        for name in unknown:
            self._logger.warning(
                "Randomization key '%s' does not match any object or "
                "operator handler — skipping.",
                name,
            )

    # ------------------------------------------------------------------
    #  Pose sampling: regions, references, and generators
    # ------------------------------------------------------------------

    def _select_region(
        self,
        spec: RandomizationInput,
        *,
        candidate_index: int = 0,
    ) -> PoseRandomRange:
        """Select one region for one sampling attempt."""
        return select_randomization_region(
            self._host.rng,
            spec,
            candidate_index=candidate_index,
        )

    def _baseline_or_live(self, label: str) -> PoseState:
        """A target's recorded reset baseline, or its live pose when unrecorded.

        The recorded baseline is the frame a reset starts from, so it is the
        right default for delta-carry references. Elements registered after
        initialization have no recorded baseline; their live pose is the only
        sensible fallback.
        """
        baseline = self._host.baseline_pose(label)
        if baseline is not None:
            return baseline
        return self._host.live_pose(label)

    def _template_pose(self, label: str) -> PoseState:
        """Batch-shaped buffer template for one action's per-env samples."""
        if label not in self.plan.actions:
            return PoseState().broadcast_to(self._host.batch_size)
        return self._host.live_pose(label)

    def _poisson_stream_for_action(
        self,
        action_spec: RandomizationAction,
        env_index: int,
        rand_range: PoseRandomRange,
        distribution: Optional[RandomizationDistributionConfig],
        working_poses: Dict[str, PoseState],
    ) -> Optional[PoissonDiskCandidateStream]:
        """Return the persistent physical-position stream for one proposal range."""
        generator = getattr(
            distribution,
            "generator",
            RandomizationGeneratorKind.IID,
        )
        if isinstance(generator, RandomizationGeneratorConfig):
            poisson_config = generator.poisson_disk
        elif generator == RandomizationGeneratorKind.POISSON_DISK:
            poisson_config = RandomizationPoissonDiskConfig()
        else:
            return None

        position_axes = tuple(
            axis for axis in ("x", "y", "z") if rand_range.axis_range(axis) is not None
        )
        if not position_axes:
            return None
        spacing = float(getattr(distribution, "spacing", 0.0))
        if spacing <= 0.0:
            raise ValueError(
                f"Poisson-disk randomization '{action_spec.label}' requires "
                "distribution.spacing > 0"
            )
        lower_bounds = tuple(
            float(rand_range.axis_range(axis)[0]) for axis in position_axes
        )
        upper_bounds = tuple(
            float(rand_range.axis_range(axis)[1]) for axis in position_axes
        )
        # The stream is keyed by the *physical* frame the range resolves to, so a
        # relative range keeps one coherent lattice while its reference moves.
        reference_context: list[float] = []
        for reference in rand_range.references():
            if isinstance(reference, RandomizationReference):
                if reference == RandomizationReference.ABSOLUTE_WORLD:
                    continue
                if reference == RandomizationReference.ABSOLUTE_BASE:
                    label = f"{action_spec.owner}.base"
                    pose = working_poses.get(label)
                    if pose is None:
                        pose = self._baseline_or_live(label)
                else:
                    pose = working_poses.get(action_spec.label)
                    if pose is None:
                        pose = self._host.live_pose(action_spec.label)
            else:
                bare, attr = parse_entity_reference(reference)
                if attr is None and bare in self._host.operator_names:
                    attr = "base"
                key = f"{bare}.{attr}" if attr is not None else bare
                pose = working_poses.get(key)
                if pose is None and attr is None:
                    pose = working_poses.get(bare)
                if pose is None:
                    pose = self._baseline_or_live(key)
            selected_pose = pose.select(env_index)
            reference_context.extend(
                np.asarray(selected_pose.position[0], dtype=np.float64).tolist()
            )
            reference_context.extend(
                np.asarray(selected_pose.orientation[0], dtype=np.float64).tolist()
            )
        key = (
            env_index,
            action_spec.label,
            position_axes,
            lower_bounds,
            upper_bounds,
            spacing,
            tuple(reference_context),
            poisson_config.hypersphere,
            int(poisson_config.ncandidates),
            poisson_config.optimization,
        )
        stream = self._poisson_streams.get(key)
        if stream is None:
            label_seed = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(action_spec.label)
            )
            stream = PoissonDiskCandidateStream(
                poisson_config,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                radius=spacing,
                seed=int((self._host.seed or 0) + env_index * 10_007 + label_seed),
            )
            self._poisson_streams[key] = stream
        return stream

    def _sample_pose_for_env(
        self,
        base_pose: PoseState,
        rand_range: PoseRandomRange,
        env_index: int,
        *,
        reference_poses: Optional[
            Mapping[Union[RandomizationReference, str], PoseState]
        ] = None,
        distribution: Any = None,
        sample_index: int = 0,
        poisson_stream: Optional[PoissonDiskCandidateStream] = None,
    ) -> PoseState:
        """Sample one environment's pose from one region."""
        return sample_pose_for_env(
            self._host.rng,
            base_pose=base_pose,
            rand_range=rand_range,
            env_index=env_index,
            batch_size=self._host.batch_size,
            reference_poses=reference_poses,
            distribution=distribution,
            sample_index=sample_index,
            reset_index=self._host.episode_index,
            poisson_stream=poisson_stream,
        )

    def _resolve_reference_base_pose_for_env(
        self,
        reference: Union[RandomizationReference, str],
        sampled_poses: Dict[str, PoseState],
        default_pose: PoseState,
        env_index: int,
    ) -> PoseState:
        """Resolve one reference to the baseline pose the sampler adds its delta to.

        For enum modes the target's own ``default_pose`` is returned. For an
        entity-name reference the delta-carry algorithm is applied:
        ``delta = ref_sampled * ref_default⁻¹``, then ``delta * default_pose``,
        so the target moves with the referenced entity while preserving their
        original spatial relationship.
        """
        if isinstance(reference, RandomizationReference):
            return default_pose
        bare, attr = parse_entity_reference(reference)
        if attr is None and bare in self._host.operator_names:
            attr = "base"  # plain operator name defaults to its base
        if attr is not None:
            if bare not in self._host.operator_names:
                raise ValueError(
                    f"Randomization reference '{reference}' — '.{attr}' is only "
                    f"valid for operator names, but '{bare}' is not a known operator."
                )
            ref_default = self._baseline_or_live(f"{bare}.{attr}").select(env_index)
            ref_sampled = sampled_poses.get(f"{bare}.{attr}")
        else:
            if bare not in self._host.object_names:
                raise ValueError(
                    f"Randomization reference '{reference}' is not a known mode "
                    "('relative', 'absolute_world', 'absolute_base') nor an existing "
                    "object/operator name."
                )
            ref_default = self._baseline_or_live(bare).select(env_index)
            ref_sampled = sampled_poses.get(bare)
        if ref_sampled is None:
            return default_pose  # entity not randomized → no delta
        delta = compose_pose(ref_sampled, inverse_pose(ref_default))
        return compose_pose(delta, default_pose)

    def _resolve_reference_poses_for_env(
        self,
        rand_range: PoseRandomRange,
        sampled_poses: Dict[str, PoseState],
        default_pose: PoseState,
        env_index: int,
    ) -> Dict[Union[RandomizationReference, str], PoseState]:
        """Resolve every reference used by one range to its baseline pose."""
        return {
            reference: self._resolve_reference_base_pose_for_env(
                reference,
                sampled_poses,
                default_pose,
                env_index,
            )
            for reference in rand_range.references()
        }

    def _operator_default_eef_following_base(
        self,
        label: str,
        owner: str,
        env_index: int,
        sampled_poses: Optional[Dict[str, PoseState]] = None,
    ) -> tuple[PoseState, PoseState]:
        """Return the operator's default EEF pose rigidly tracking its current
        base, plus the resolved current base pose.

        The recorded EEF baseline is a **world** frame pose. If the operator's
        base is later randomized, naïvely reusing that world-frame default leaves
        the EEF target at its old absolute position, and the IK chain has to
        bridge a base-induced offset that grows with the base randomization
        range — quickly becoming unreachable and surfacing as ``ik_unreachable``
        failures even when the EEF offset itself is small.

        Re-anchoring to the current base preserves the original eef-in-base
        relative pose, so randomizing the base does not implicitly enlarge the
        EEF reach budget.

        ``sampled_poses`` (if given) is consulted for an in-flight base sample so
        eef sampling sees the base that was just decided in the same iteration;
        otherwise the host's live base pose is used.
        """
        default_eef_world = self._baseline_or_live(label).select(env_index)
        default_base_world = self._baseline_or_live(f"{owner}.base").select(env_index)
        current_base_world: Optional[PoseState] = None
        if sampled_poses is not None:
            current_base_world = sampled_poses.get(owner)
        if current_base_world is None:
            current_base_world = self._host.live_pose(f"{owner}.base").select(env_index)
        eef_in_default_base = compose_pose(
            inverse_pose(default_base_world), default_eef_world
        )
        return (
            compose_pose(current_base_world, eef_in_default_base),
            current_base_world,
        )

    def _sample_target_for_env(
        self,
        action_spec: RandomizationAction,
        env_index: int,
        working_poses: Dict[str, PoseState],
        *,
        candidate_index: int = 0,
    ) -> tuple[Dict[str, PoseState], List[PendingRandomizationAction]]:
        """Sample one action's pose for one environment."""
        label = action_spec.label
        owner = action_spec.owner
        if action_spec.kind == "unknown" or (
            action_spec.kind != "object" and owner not in self._host.operator_names
        ):
            self._logger.warning(
                "Randomization key '%s' does not match any object or operator "
                "handler — skipping.",
                owner,
            )
            return {}, []

        selected_range = self._select_region(
            action_spec.randomization,
            candidate_index=candidate_index,
        )
        distribution = action_spec.randomization.distribution
        poisson_stream = self._poisson_stream_for_action(
            action_spec,
            env_index,
            selected_range,
            distribution,
            working_poses,
        )
        if action_spec.kind == "object":
            if RandomizationReference.ABSOLUTE_BASE in selected_range.references():
                raise ValueError(
                    f"Object '{owner}' randomization cannot use 'absolute_base' — "
                    "only operator end-effector randomization is defined in a "
                    "base frame."
                )
            default_pose = self._baseline_or_live(label).select(env_index)
            sampled = self._sample_pose_for_env(
                default_pose,
                selected_range,
                0,
                reference_poses=self._resolve_reference_poses_for_env(
                    selected_range,
                    working_poses,
                    default_pose,
                    env_index,
                ),
                distribution=distribution,
                sample_index=candidate_index,
                poisson_stream=poisson_stream,
            )
            return {label: sampled}, [
                self._pending_action(
                    action_spec,
                    owner,
                    sampled,
                    selected_range,
                    env_index,
                )
            ]

        if action_spec.kind == "operator_base":
            if RandomizationReference.ABSOLUTE_BASE in selected_range.references():
                raise ValueError(
                    f"Operator '{owner}' base randomization cannot use "
                    "'absolute_base' — the base IS the frame."
                )
            default_pose = self._baseline_or_live(label).select(env_index)
            sampled = self._sample_pose_for_env(
                default_pose,
                selected_range,
                0,
                reference_poses=self._resolve_reference_poses_for_env(
                    selected_range,
                    working_poses,
                    default_pose,
                    env_index,
                ),
                distribution=distribution,
                sample_index=candidate_index,
                poisson_stream=poisson_stream,
            )
        elif action_spec.kind == "operator_eef":
            sampled = self._sample_operator_eef_pose_for_env(
                label,
                owner,
                selected_range,
                env_index,
                working_poses,
                distribution=distribution,
                sample_index=candidate_index,
                poisson_stream=poisson_stream,
            )
        else:
            raise ValueError(f"Unknown randomization action kind: {action_spec.kind}")
        return {
            owner: sampled,
            label: sampled,
        }, [
            self._pending_action(
                action_spec,
                owner,
                sampled,
                selected_range,
                env_index,
            )
        ]

    def _pending_action(
        self,
        action_spec: RandomizationAction,
        owner: str,
        pose: PoseState,
        selected_range: PoseRandomRange,
        env_index: int,
    ) -> PendingRandomizationAction:
        """Wrap one sampled pose with its resolved radius and constraints."""
        return PendingRandomizationAction(
            kind=action_spec.kind,
            owner=owner,
            label=action_spec.label,
            pose=pose,
            radius=self._collision_radius(
                kind=action_spec.kind,
                owner=owner,
                env_index=env_index,
                spec_radius=float(selected_range.collision_radius),
                margin=float(selected_range.collision_margin),
            ),
            references=selected_range.references(),
            constraints=action_spec.randomization.constraints,
        )

    def _sample_operator_eef_pose_for_env(
        self,
        label: str,
        owner: str,
        rand_range: PoseRandomRange,
        env_index: int,
        sampled_poses: Dict[str, PoseState],
        *,
        distribution: Any = None,
        sample_index: int = 0,
        poisson_stream: Optional[PoissonDiskCandidateStream] = None,
    ) -> PoseState:
        """Sample one operator's EEF pose for one environment."""
        following_base_default, base_world = self._operator_default_eef_following_base(
            label,
            owner,
            env_index,
            sampled_poses,
        )
        references = rand_range.references()
        if references == (RandomizationReference.ABSOLUTE_BASE,):
            # The range is expressed in the base frame, so sample there and lift
            # the result back into the world frame.
            default_in_base = compose_pose(
                inverse_pose(base_world),
                following_base_default,
            )
            sampled_in_base = self._sample_pose_for_env(
                default_in_base,
                rand_range,
                0,
                distribution=distribution,
                sample_index=sample_index,
                poisson_stream=poisson_stream,
            )
            return compose_pose(base_world, sampled_in_base)

        snapshot_default = self._baseline_or_live(label).select(env_index)
        reference_poses: Dict[Union[RandomizationReference, str], PoseState] = {}
        for reference in references:
            if isinstance(reference, RandomizationReference):
                reference_poses[reference] = following_base_default
            else:
                reference_poses[reference] = self._resolve_reference_base_pose_for_env(
                    reference,
                    sampled_poses,
                    snapshot_default,
                    env_index,
                )
        return self._sample_pose_for_env(
            following_base_default,
            rand_range,
            0,
            reference_poses=reference_poses,
            distribution=distribution,
            sample_index=sample_index,
            poisson_stream=poisson_stream,
        )

    # ------------------------------------------------------------------
    #  Collision radii
    # ------------------------------------------------------------------

    def _collision_radius(
        self,
        *,
        kind: str,
        owner: str,
        env_index: int,
        spec_radius: float,
        margin: float = 0.0,
    ) -> float:
        """Resolve a region's ``collision_radius`` for one environment.

        Positive values are used verbatim; ``0`` stays exempt; a negative value
        requests ``auto`` and resolves to the entity's conservative
        support-geometry radius plus ``collision_margin`` (cached per
        entity/environment).
        """
        radius = float(spec_radius)
        if radius >= 0.0:
            return radius
        auto = self._auto_collision_radius(kind=kind, owner=owner, env_index=env_index)
        return auto + float(margin)

    def _auto_collision_radius(self, *, kind: str, owner: str, env_index: int) -> float:
        """Return the host-derived conservative radius for one entity."""
        key = (kind, owner, env_index)
        cached = self._auto_radius_cache.get(key)
        if cached is not None:
            return cached
        if kind == "object":
            geometry = self._host.get_support_geometry(owner, env_index)
        elif kind in ("operator_base", "operator_eef"):
            part = "base" if kind == "operator_base" else "eef"
            geometry = self._host.get_operator_support_geometry(owner, part, env_index)
        else:
            raise KeyError(
                f"Unknown randomization kind {kind!r} for auto collision_radius "
                f"of '{owner}'."
            )
        cached = float(geometry.radius)
        self._auto_radius_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    #  Deterministic visibility preflight
    # ------------------------------------------------------------------

    def run_visibility_preflight(self, env_mask: np.ndarray) -> None:
        """Fail fast when a ``visible_in`` region is provably empty.

        A ``visible_in`` object region whose whole position box lies outside a
        required camera frustum can never be satisfied, so it is reported with
        ``attempts=0`` instead of exhausting the attempt loop.
        """
        action_specs = self.plan.actions
        infeasibility = find_visibility_infeasibility(
            {
                label: spec.randomization
                for label, spec in action_specs.items()
                if spec.kind == "object"
            },
            env_mask=env_mask,
            default_pose_of=self._host.baseline_pose,
            camera_names_of=self._visibility_camera_names,
            camera_model_of=self._host.get_camera_model,
        )
        if infeasibility is None:
            return
        constraints = action_specs[infeasibility.target].randomization.constraints
        self._host.record_reset_diagnostics(
            infeasibility.env_index,
            {
                "group": "component",
                "generator": "deterministic_infeasible",
                "member": infeasibility.target,
                "attempts": 0,
                "violations": list(infeasibility.violations),
                "minimum_clearance": float("-inf"),
                "mode": constraints.failure.mode.value,
            },
        )
        raise RandomizationFailureError(
            target=infeasibility.target,
            attempts=0,
            violations=list(infeasibility.violations),
            minimum_clearance=float("-inf"),
        )

    def _visibility_camera_names(
        self,
        visible: RandomizationVisibilityConfig,
        env_index: int,
    ) -> List[str]:
        """Resolve a visibility config's camera list for one environment."""
        if visible.cameras != "all":
            return list(visible.cameras)
        return self._host.camera_names()

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
        action_specs = self.plan.actions
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
                sampled_poses, actions = self._sample_target_for_env(
                    action_specs[action_label],
                    env_index,
                    working_poses,
                    candidate_index=(
                        self._host.episode_index * 1009
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
            self._host.record_reset_diagnostics(env_index, diagnostics)
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
        action_specs = self.plan.actions
        accepted_env_poses = {
            name: pose.select(env_index)
            for name, pose in accepted_sampled_poses.items()
        }
        working_poses = dict(accepted_env_poses)
        sampled_poses: Dict[str, PoseState] = {}
        actions: List[PendingRandomizationAction] = []
        local_participants: List[CollisionParticipant] = []
        dependencies = self.action_dependencies()
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
            rng = self._host.rng
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
                candidate_poses, candidate_actions = self._sample_target_for_env(
                    action_spec,
                    env_index,
                    working_poses,
                    candidate_index=(
                        self._host.episode_index * 1009
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
            self._host.record_reset_diagnostics(env_index, diagnostics)
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

    def apply_camera_randomization(self, env_mask: np.ndarray) -> None:
        """Sample and apply pose randomization for the configured cameras.

        Cameras sample a pose stream like any other target, but they own no
        collision, separation, or dependency semantics: ``absolute_base`` and
        entity-name references are rejected, and no constraint is evaluated.
        The host only has to be able to read and write a named camera's
        world pose.
        """
        for camera_name, randomization in self.camera_randomization.items():
            canonical = canonical_randomization_spec(randomization)
            rand_range = select_randomization_region(
                self._host.rng,
                canonical,
            )
            for reference in rand_range.references():
                if reference == RandomizationReference.ABSOLUTE_BASE:
                    raise ValueError(
                        f"Camera '{camera_name}' randomization cannot use "
                        "'absolute_base' — cameras have no operator base frame."
                    )
                if isinstance(reference, str) and not isinstance(
                    reference,
                    RandomizationReference,
                ):
                    raise ValueError(
                        f"Camera '{camera_name}' randomization cannot use entity "
                        f"reference '{reference}' — cameras do not participate in "
                        "entity dependency ordering."
                    )
            default_pose = self._host.baseline_pose(camera_name)
            if default_pose is None:
                self._logger.warning(
                    "Camera '%s' has no recorded reset baseline — skipping "
                    "randomization.",
                    camera_name,
                )
                continue
            sampled = sample_pose_batch(
                self._host.rng,
                base_pose=default_pose,
                rand_range=rand_range,
                env_mask=env_mask,
                batch_size=self._host.batch_size,
                distribution=canonical.distribution,
                reset_index=self._host.episode_index,
            )
            self._host.set_camera_pose(camera_name, sampled, env_mask)

    def apply_randomization(self, env_mask: np.ndarray) -> None:
        self.begin_episode()
        plan = self.plan
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
                self._host.set_target_pose(
                    action.kind,
                    action.owner,
                    action.pose,
                    env_mask,
                )
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

        self.apply_camera_randomization(env_mask)

        # A ``visible_in`` object region whose whole position box is outside a
        # required camera frustum is deterministically infeasible — fail fast
        # with a diagnostic instead of exhausting the attempt loop.
        self.run_visibility_preflight(env_mask)

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
        key_buffers = {name: self._template_pose(name) for name in component}
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
                    template = self._template_pose(action.label)
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
