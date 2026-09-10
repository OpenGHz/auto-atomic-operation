"""randomization configuration models (split from the former auto_atom.framework monolith)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

from auto_atom.config.reference import RandomizationReference


class RandomizationAxisConfig(BaseModel, frozen=True):
    """Randomization bounds and an optional reference for one pose axis."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    range: Tuple[float, float]
    """Inclusive ``[min, max]`` sampling range for this axis."""

    reference: Optional[Union[RandomizationReference, str]] = None
    """Axis-specific reference. ``None`` inherits the pose-level reference."""

    @field_validator("reference", mode="before")
    @classmethod
    def _coerce_reference(cls, v: object) -> object:
        if isinstance(v, str) and not isinstance(v, RandomizationReference):
            try:
                return RandomizationReference(v)
            except ValueError:
                return v
        return v


RandomizationAxisSpec = Union[Tuple[float, float], RandomizationAxisConfig]


class PoseRandomRange(BaseModel, frozen=True):
    """Per-entity pose randomization bounds.

    Each axis accepts either a compact ``[min, max]`` range or an expanded
    ``{range: [min, max], reference: ...}`` object. An expanded axis reference
    takes precedence over the pose-level ``reference``; an omitted axis
    reference inherits the pose-level value. The pose-level ``reference``
    selects one of three modes, **or** names another entity to track:

    - ``"relative"`` (default): each per-axis ``[min, max]`` range is an
      additive offset applied to the entity's default/initial pose.
    - ``"absolute_world"``: ranges are absolute world-frame values —
      metres for position, radians for Euler orientation. The default
      pose is ignored for any axis that has an explicit range.
    - ``"absolute_base"``: ranges are absolute values expressed in the
      operator's base frame. The sampled pose is then transformed into
      world frame before being applied. Only valid for operator
      end-effector randomization.
    - **Entity name** (e.g. ``"vase1"``): the referenced entity is
      randomized first; its displacement from its default pose is
      computed (``delta = sampled * default⁻¹``) and applied to this
      entity's default pose so they move together. Then the per-axis
      ranges are applied as additive offsets on top, just like
      ``relative`` mode. For an **operator** name, the plain form
      tracks the operator's **base** pose (equivalent to the
      ``"<operator>.base"`` form below).
    - **Operator attribute** (e.g. ``"arm.base"`` or ``"arm.eef"``):
      same delta-carry semantics as the entity-name form, but
      explicitly anchored to the operator's **base** or
      **end-effector** pose. Only ``.base`` / ``.eef`` suffixes are
      recognized, and only for operator names.

    A ``None`` value on an axis (the default) means "do not randomize
    this axis" — it keeps its value from the default pose (in the
    relevant frame) in all modes. Axes are independent, so absolute-mode
    ``x``/``y`` with ``z``/``roll``/``pitch``/``yaw`` left as ``None``
    produces the natural "place anywhere on this rectangle, keep default
    height and orientation" behavior.

    Example YAML entries::

        # Relative (default): sampled as default_pose + offset
        randomization:
          source_block:
            x: [-0.03, 0.03]
            y:
              range: [-0.20, 0.20]
              reference: absolute_world
            collision_radius: 0.04

        # Absolute world-frame: sampled as world-frame coordinates
        randomization:
          vase1:
            reference: absolute_world
            x: [0.10, 0.45]
            y: [-0.15, 0.15]

        # Entity reference: carry flower with vase1, then jitter ±5mm
        randomization:
          vase1:
            reference: absolute_world
            x: [0.22, 0.58]
            y: [-0.32, 0.27]
          flower:
            reference: vase1
            x: [-0.005, 0.005]
            y: [-0.005, 0.005]

        # Operator-base reference: carry vase with the arm's base
        randomization:
          arm:
            base:
              x: [-0.05, 0.05]
              y: [-0.05, 0.05]
          vase:
            reference: arm.base
            x: [-0.005, 0.005]
            y: [-0.005, 0.005]
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    x: Optional[RandomizationAxisSpec] = None
    """X range in metres, optionally with its own reference."""
    y: Optional[RandomizationAxisSpec] = None
    """Y range in metres, optionally with its own reference."""
    z: Optional[RandomizationAxisSpec] = None
    """Z range in metres, optionally with its own reference."""
    roll: Optional[RandomizationAxisSpec] = None
    """Roll range in radians, optionally with its own reference."""
    pitch: Optional[RandomizationAxisSpec] = None
    """Pitch range in radians, optionally with its own reference."""
    yaw: Optional[RandomizationAxisSpec] = None
    """Yaw range in radians, optionally with its own reference."""
    reference: Union[RandomizationReference, str] = RandomizationReference.RELATIVE
    """One of the :class:`RandomizationReference` modes (``"relative"``,
    ``"absolute_world"``, ``"absolute_base"``), the **name of another
    entity**, or an **operator attribute** (``"<operator>.base"`` /
    ``"<operator>.eef"``). An entity/attribute reference causes this
    entry to track the referenced pose's displacement (delta-carry) and
    then apply the per-axis ranges as relative offsets on top. A plain
    operator name is equivalent to ``"<operator>.base"``."""
    collision_radius: float = 0.05
    """Bounding radius used for pairwise collision rejection (metres).

    * ``> 0``: explicit radius.
    * ``0``: exempt — the entity does not participate in collision rejection.
    * ``< 0`` (convention ``-1``): ``auto`` — the backend derives a
      conservative radius from the entity's support geometry, plus
      ``collision_margin``. The magnitude is ignored; any negative value
      means ``auto``.

    ``auto`` semantics by participant:

    * **object**: radius covering the object body's geoms around its own
      geometric center.
    * **operator ``base``**: radius of the operator's **root/base body
      footprint** — geoms attached to the mount body only (not the arm
      subtree). A mount body with no geoms resolves to radius ``0.0``
      (effectively exempt).
    * **operator ``eef``**: radius of the **end-effector assembly** — geoms
      under the body owning the EEF site (gripper/fingers), measured around
      the EEF site. This varies with gripper open/close, so it is resolved
      per episode against the home configuration.
    """

    collision_margin: NonNegativeFloat = 0.0
    """Extra clearance (metres) added to the auto-derived radius when
    ``collision_radius < 0``."""

    @field_validator("reference", mode="before")
    @classmethod
    def _coerce_reference(cls, v: object) -> object:
        if isinstance(v, str) and not isinstance(v, RandomizationReference):
            try:
                return RandomizationReference(v)
            except ValueError:
                return v  # entity name — validated at sample time
        return v

    def axis_range(self, axis: str) -> Optional[Tuple[float, float]]:
        """Return one axis's concrete sampling range."""
        value = getattr(self, axis)
        if isinstance(value, RandomizationAxisConfig):
            return value.range
        return value

    def axis_reference(
        self,
        axis: str,
    ) -> Union[RandomizationReference, str]:
        """Return one axis's effective reference after fallback resolution."""
        value = getattr(self, axis)
        if isinstance(value, RandomizationAxisConfig) and value.reference is not None:
            return value.reference
        return self.reference

    def references(self) -> Tuple[Union[RandomizationReference, str], ...]:
        """Return every effective reference declared by this pose range."""
        references = [
            self.axis_reference(axis)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        ]
        return tuple(dict.fromkeys(references))


class PoseRandomizationConfig(BaseModel, frozen=True):
    """A choice among one or more independently configured pose regions.

    Each region is a complete :class:`PoseRandomRange`, so it owns its axis
    ranges, reference mode, and collision radius.  The legacy direct
    ``PoseRandomRange`` form remains valid wherever this wrapper is accepted.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    regions: List[PoseRandomRange] = Field(min_length=1)
    """Non-empty collection of candidate regions sampled at reset."""


PoseRandomizationSpec = Union[PoseRandomRange, PoseRandomizationConfig]


class RandomizationGeneratorKind(str, Enum):
    """Concrete candidate generator used by a randomization distribution."""

    IID = "iid"
    """Independent pseudo-random candidates from NumPy's generator."""

    LATIN_HYPERCUBE = "latin_hypercube"
    """Latin-hypercube candidates from SciPy's QMC implementation."""

    HALTON = "halton"
    """Halton low-discrepancy candidates from SciPy's QMC implementation."""

    SOBOL = "sobol"
    """Sobol low-discrepancy candidates from SciPy's QMC implementation."""

    POISSON_DISK = "poisson_disk"
    """Poisson-disk candidates from SciPy's QMC implementation."""


class RandomizationPoissonDiskConfig(BaseModel, frozen=True):
    """Algorithm parameters for the SciPy Poisson-disk candidate generator.

    The configuration is nested under ``distribution.generator.poisson_disk``
    only when the default Poisson-disk parameters need to be overridden.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    hypersphere: Literal["volume", "surface"] = "volume"
    """Whether potential points are sampled inside or on the candidate sphere."""

    ncandidates: PositiveInt = 30
    """Number of Poisson-disk proposals considered per active point."""

    optimization: Optional[Literal["random-cd", "lloyd"]] = None
    """Optional SciPy post-processing optimization."""


class RandomizationGeneratorConfig(BaseModel, frozen=True):
    """Parameterized candidate generators keyed by their algorithm name."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    poisson_disk: RandomizationPoissonDiskConfig
    """SciPy Poisson-disk parameters."""


RandomizationGeneratorInput = Union[
    RandomizationGeneratorKind,
    RandomizationGeneratorConfig,
]


class RandomizationSelectorKind(str, Enum):
    """Candidate acceptance or selection policy."""

    FIRST_FEASIBLE = "first_feasible"
    """Accept the first candidate satisfying all hard constraints."""

    MAXIMIN = "maximin"
    """Select the current candidate farthest from accepted samples."""


class RandomizationVisibilityGeometry(str, Enum):
    """Geometry approximation used by a camera visibility constraint."""

    CENTER = "center"
    """Check only the entity reference point."""

    BOUNDING_SPHERE = "bounding_sphere"
    """Check a conservative sphere enclosing the entity."""

    SUPPORT_HULL = "support_hull"
    """Check the backend-provided support points or convex hull."""


class RandomizationVisibilityMode(str, Enum):
    """Visibility test used by a camera constraint."""

    FRUSTUM = "frustum"
    """Require the support geometry to project inside every image."""

    SEGMENTATION = "segmentation"
    """Additionally require a minimum visible rendered fraction."""


class RandomizationFailureMode(str, Enum):
    """Behavior when no candidate satisfies a hard randomization constraint."""

    ERROR = "error"
    """Fail reset and report the violated constraint."""

    BEST_EFFORT = "best_effort"
    """Apply the least-violating candidate and report diagnostics."""


class RandomizationStrategy(str, Enum):
    """Placement strategy used for one reset's randomized components."""

    RSA = "rsa"
    """Place independent objects sequentially and keep accepted members fixed."""

    JOINT_REJECTION = "joint_rejection"
    """Sample every component member together and reject the whole proposal on failure."""


class RandomizationVisibilityConfig(BaseModel, frozen=True):
    """Keep an entity's support geometry inside a set of camera views."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    cameras: Union[Literal["all"], List[str]] = "all"
    """Camera names to satisfy, or ``all`` for the backend's observation set."""

    geometry: RandomizationVisibilityGeometry = (
        RandomizationVisibilityGeometry.BOUNDING_SPHERE
    )
    """Entity geometry approximation used for the image-bound check."""

    mode: RandomizationVisibilityMode = RandomizationVisibilityMode.FRUSTUM
    """Frustum-only or rendered-segmentation visibility."""

    margin_px: NonNegativeInt = 0
    """Required pixel margin from every image edge."""

    min_visible_fraction: float = 0.0
    """Minimum rendered fraction when ``mode=segmentation``; range ``[0, 1]``."""

    @field_validator("cameras", mode="after")
    @classmethod
    def _validate_cameras(
        cls, value: Union[Literal["all"], List[str]]
    ) -> Union[Literal["all"], List[str]]:
        if value == "all":
            return value
        if not value or any(not str(name).strip() for name in value):
            raise ValueError("visibility cameras must be 'all' or non-empty names")
        return [str(name) for name in value]

    @field_validator("min_visible_fraction", mode="after")
    @classmethod
    def _validate_visible_fraction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("min_visible_fraction must be in [0, 1]")
        return value


class RandomizationSeparationConfig(BaseModel, frozen=True):
    """Hard separation constraints for randomized entities."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    scope: Literal["randomized", "scene"] = "randomized"
    """Check only randomized participants or all backend-supported scene geometry."""

    clearance: NonNegativeFloat = 0.0
    """Required surface clearance in metres between the selected geometries.

    The actual enforced minimum center distance is
    ``collision_radius_i + collision_radius_j + clearance``. Object size is
    carried by the radius terms, so ``clearance`` is a pure surface gap and is
    size-adaptive by construction.
    """

    geometry: Literal["center", "support"] = "support"
    """Use center distances or backend support geometry."""


class RandomizationFailureConfig(BaseModel, frozen=True):
    """Failure policy and budget for constrained randomization.

    ``failure`` governs the feasibility/acceptance loop: what happens when no
    candidate satisfies the collision/separation or visibility requirements
    within ``max_attempts``. It therefore lives under ``constraints``.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    mode: RandomizationFailureMode = RandomizationFailureMode.ERROR
    """Fail closed or apply a best-effort candidate with a diagnostic."""

    max_attempts: PositiveInt = 100
    """Maximum candidate attempts before applying the failure policy."""


class RandomizationConstraintConfig(BaseModel, frozen=True):
    """Hard constraints applied to sampled pose candidates.

    A constraint set carries the optional visibility / separation requirements
    plus the ``failure`` policy for the feasibility loop that tries to satisfy
    them (including the always-on bounding-sphere separation between randomized
    participants).
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    visible_in: Optional[RandomizationVisibilityConfig] = None
    """Require the target entity to remain inside the selected camera views."""

    separated: Optional[RandomizationSeparationConfig] = None
    """Require clearance from other randomized entities or the scene.

    This block carries only the separation geometry and clearance. The
    placement strategy is a scope-level property (``randomization.strategy``)
    because it also governs the always-on collision rejection, which applies
    whether or not ``separated`` is configured.
    """

    failure: RandomizationFailureConfig = RandomizationFailureConfig()
    """Behavior when the constrained proposal is infeasible or exhausted."""


class RandomizationDistributionConfig(BaseModel, frozen=True):
    """Candidate generator and selector for one randomization target."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    generator: RandomizationGeneratorInput = RandomizationGeneratorKind.IID
    """Concrete candidate generator, optionally with Poisson-disk parameters."""

    selector: RandomizationSelectorKind = RandomizationSelectorKind.FIRST_FEASIBLE
    """Policy used to accept or choose generated feasible candidates."""

    region_weighting: Literal["equal", "volume"] = "volume"
    """How multiple proposal regions are selected for uniform sampling."""

    candidate_count: PositiveInt = 1
    """Maximum candidate pool size considered for one target or joint sample."""

    spacing: NonNegativeFloat = 0.0
    """Sampling spacing in metres for Poisson and cross-reset history coverage.

    This is a per-entity stream property (how spread out consecutive samples are
    over the proposal volume and across resets). It is not a clearance and does
    not participate in collision or separation checks.
    """


class RandomizationGroupGeneratorKind(str, Enum):
    """Joint candidate-placement algorithms for randomized-object groups."""

    HARD_SPHERE_RSA = "hard_sphere_rsa"
    """Random sequential adsorption using each member's bounding-sphere radius."""


class RandomizationGroupDistributionConfig(BaseModel, frozen=True):
    """Distribution settings shared by every member of one placement group."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    generator: RandomizationGroupGeneratorKind
    """Joint placement algorithm applied after member proposals generate candidates."""


class RandomizationGroupConfig(BaseModel, frozen=True):
    """A jointly randomized set of scene objects."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    members: List[str] = Field(min_length=2)
    """Distinct object randomization keys placed by this joint distribution."""

    distribution: RandomizationGroupDistributionConfig
    """Joint placement distribution and its physical clearance."""

    failure: RandomizationFailureConfig = RandomizationFailureConfig()
    """Behavior when sequential placement cannot find a feasible local candidate."""

    @field_validator("members", mode="after")
    @classmethod
    def _validate_members(cls, value: List[str]) -> List[str]:
        if any(not member.strip() for member in value):
            raise ValueError("randomization group members must be non-empty names")
        if len(set(value)) != len(value):
            raise ValueError("randomization group members must be distinct")
        return value


class RandomizationSpec(BaseModel, frozen=True):
    """Canonical proposal, distribution, and constraint specification."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    proposal: PoseRandomizationSpec
    """Single or multi-region pose proposal."""

    distribution: RandomizationDistributionConfig = RandomizationDistributionConfig()
    """Distribution objective for accepted candidates."""

    constraints: RandomizationConstraintConfig = RandomizationConstraintConfig()
    """Visibility, separation, and failure policy evaluated by the backend.

    ``failure`` lives under ``constraints`` because it governs the feasibility
    loop that tries to satisfy the collision/separation and visibility
    requirements within the retry budget.
    """


RandomizationInput = Union[PoseRandomizationSpec, RandomizationSpec]


def canonical_randomization_spec(spec: RandomizationInput) -> RandomizationSpec:
    """Normalize legacy ranges into the canonical randomization specification."""
    if isinstance(spec, RandomizationSpec):
        return spec
    # Preserve the historical equal-probability region mixture for legacy
    # ``PoseRandomizationConfig`` values. New canonical specs default to
    # proposal-volume weighting for uniform-feasible sampling. Failure policy
    # uses the new fail-closed semantics (``error``) via ``constraints``.
    return RandomizationSpec(
        proposal=spec,
        distribution=RandomizationDistributionConfig(region_weighting="equal"),
    )


def pose_randomization_regions(
    spec: RandomizationInput,
) -> Tuple[PoseRandomRange, ...]:
    """Return the concrete regions represented by a randomization spec."""
    if isinstance(spec, RandomizationSpec):
        spec = spec.proposal
    if isinstance(spec, PoseRandomizationConfig):
        return tuple(spec.regions)
    return (spec,)


class OperatorRandomizationConfig(BaseModel):
    """Randomization options for an operator.

    ``base`` controls the operator's logical base pose. ``eef`` controls its
    home end-effector pose. Concrete backends own the mapping from these
    logical poses to simulator or hardware state.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    base: Optional[RandomizationInput] = None
    """Optional single- or multi-region randomization for the operator base."""
    eef: Optional[RandomizationInput] = None
    """Optional single- or multi-region randomization for the end effector."""


class RandomizationScopeConfig(BaseModel):
    """Container for global randomization defaults and the target maps.

    This is the value of ``task.randomization``. It groups five orthogonal
    concerns:

    * ``strategy`` — the scope-wide placement strategy. It is deliberately
      *not* nested under ``constraints.separated``: it also governs the
      always-on collision rejection between randomized participants, which
      applies whether or not ``separated`` is configured.
    * ``distribution`` — the global default candidate-generation settings.
      Bare ``entities`` and ``cameras`` ranges inherit this default.
    * ``constraints`` — the global default feasibility settings (visibility,
      separation, failure policy). Only bare ``entities`` ranges inherit this
      default: cameras own no separation, visibility, or feasibility-loop
      semantics.
    * ``entities`` — the per-entity entries (objects, or operators with
      ``base`` / ``eef``). Advanced ``RandomizationSpec`` entries are fully
      explicit and override the scope defaults entirely.
    * ``cameras`` — the per-camera entries. They inherit ``distribution`` (how
      the pose stream is generated) but never ``constraints``.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    strategy: RandomizationStrategy = RandomizationStrategy.RSA
    """Placement strategy for one reset's reference-connected components.

    ``rsa`` places members sequentially and keeps accepted members fixed;
    ``joint_rejection`` samples the whole component together and rejects it on
    any failure. The strategy is scope-wide and cannot be overridden per
    entity, because one reset applies one placement strategy.
    """

    distribution: RandomizationDistributionConfig = RandomizationDistributionConfig()
    """Global default distribution inherited by bare entity and camera ranges."""

    constraints: RandomizationConstraintConfig = RandomizationConstraintConfig()
    """Global default constraints (visible_in / separated / failure) inherited
    by bare entity ranges."""

    entities: Dict[
        str,
        Union[RandomizationInput, OperatorRandomizationConfig],
    ] = Field(default_factory=dict)
    """Per-entity randomization entries."""

    cameras: Dict[
        str,
        Union[PoseRandomRange, RandomizationSpec],
    ] = Field(default_factory=dict)
    """Per-camera pose randomization entries.

    Keys are logical camera names exposed by the selected backend. Each entry
    is a ``PoseRandomRange`` controlling which axes are randomized and how.

    Cameras sample a pose stream but own no separation / visibility /
    feasibility semantics. They therefore inherit the scope-wide
    ``distribution`` (generator, selector, spacing, candidate pool) but never
    ``constraints``; declaring ``constraints`` on a camera entry is rejected
    rather than silently ignored. Only ``relative`` (default) and
    ``absolute_world`` reference modes are supported; ``absolute_base`` and
    entity-name references are rejected.

    Example YAML::

        randomization:
          cameras:
            env1_cam:
              x: [-0.05, 0.05]
              y: [-0.05, 0.05]
              pitch: [-0.1, 0.1]
            env0_cam:
              reference: absolute_world
              x: [0.8, 1.0]
              y: [-0.1, 0.1]
              z: [0.4, 0.6]
    """

    @field_validator("cameras", mode="after")
    @classmethod
    def _validate_camera_entries(
        cls,
        value: Dict[str, Union[PoseRandomRange, RandomizationSpec]],
    ) -> Dict[str, Union[PoseRandomRange, RandomizationSpec]]:
        for name, spec in value.items():
            if isinstance(spec, PoseRandomizationConfig) or (
                isinstance(spec, RandomizationSpec)
                and isinstance(spec.proposal, PoseRandomizationConfig)
            ):
                raise ValueError(
                    f"cameras[{name!r}] accepts one PoseRandomRange; "
                    "regions are not supported for cameras"
                )
            if isinstance(spec, RandomizationSpec) and (
                spec.constraints != RandomizationConstraintConfig()
            ):
                raise ValueError(
                    f"cameras[{name!r}] must not declare constraints; cameras "
                    "have no separation, visibility, or feasibility-loop "
                    "semantics (only the scope-wide distribution is inherited)"
                )
        return value


@dataclass(frozen=True)
class ResolvedRandomizationScope:
    """A randomization scope expanded against its own defaults.

    ``entities`` and ``cameras`` hold the same shape as the scope maps, but
    every bare range is already wrapped into a ``RandomizationSpec`` carrying
    the defaults it inherits. ``strategy`` is the effective separation strategy
    the backend must apply.
    """

    entities: Dict[str, Union[RandomizationInput, OperatorRandomizationConfig]] = field(
        default_factory=dict
    )
    """Resolved per-entity entries with scope defaults applied."""

    cameras: Dict[str, RandomizationInput] = field(default_factory=dict)
    """Resolved per-camera entries with the scope distribution applied."""

    strategy: RandomizationStrategy = RandomizationStrategy.RSA
    """Effective placement strategy for this scope."""


def _apply_scope_defaults(
    value: RandomizationInput,
    *,
    distribution: RandomizationDistributionConfig,
    constraints: Optional[RandomizationConstraintConfig],
) -> RandomizationInput:
    """Give a bare proposal the given defaults, or pass an advanced spec through.

    Legacy ``PoseRandomRange`` / ``PoseRandomizationConfig`` inputs carry no
    distribution/constraints of their own and therefore inherit the scope-wide
    defaults. An advanced ``RandomizationSpec`` is fully explicit and is
    returned unchanged.

    ``constraints`` is ``None`` for targets that own no constraint semantics
    (cameras), which keeps their built-in default instead.
    """
    if isinstance(value, RandomizationSpec):
        return value
    return RandomizationSpec(
        proposal=value,
        distribution=distribution,
        constraints=(
            constraints if constraints is not None else RandomizationConstraintConfig()
        ),
    )


def resolve_randomization_scope(
    scope: RandomizationScopeConfig,
) -> ResolvedRandomizationScope:
    """Expand a randomization scope into resolved targets plus the strategy.

    Bare object ranges and bare operator ``base`` / ``eef`` ranges inherit the
    scope-wide ``distribution`` and ``constraints`` defaults (three-level
    fallback: target explicit > scope default > built-in default). Bare camera
    ranges inherit only ``distribution``: cameras have no separation,
    visibility, or feasibility-loop semantics, so their ``constraints`` stays at
    the built-in default. The result is consumed by the backends.
    """
    strategy = scope.strategy
    resolved: Dict[str, Union[RandomizationInput, OperatorRandomizationConfig]] = {}
    for name, value in scope.entities.items():
        if isinstance(value, OperatorRandomizationConfig):
            resolved[name] = OperatorRandomizationConfig(
                base=(
                    _apply_scope_defaults(
                        value.base,
                        distribution=scope.distribution,
                        constraints=scope.constraints,
                    )
                    if value.base is not None
                    else None
                ),
                eef=(
                    _apply_scope_defaults(
                        value.eef,
                        distribution=scope.distribution,
                        constraints=scope.constraints,
                    )
                    if value.eef is not None
                    else None
                ),
            )
            continue
        resolved[name] = _apply_scope_defaults(
            value,
            distribution=scope.distribution,
            constraints=scope.constraints,
        )
    cameras = {
        name: _apply_scope_defaults(
            value,
            distribution=scope.distribution,
            constraints=None,
        )
        for name, value in scope.cameras.items()
    }
    return ResolvedRandomizationScope(
        entities=resolved,
        cameras=cameras,
        strategy=strategy,
    )


@dataclass(frozen=True)
class ResolvedRandomizationConfig:
    """The complete randomization configuration of one task.

    This is the randomization layer's *input*: the resolved scope (per-entity
    and per-camera entries plus the placement strategy) together with the
    generated-group definitions. It exists so that a backend never holds a
    randomization config at all — the task factory builds this value and hands
    it to the randomization layer, which compiles it into a
    :class:`~auto_atom.randomization.RandomizationPlan`.

    The two halves travel together because they are one task-level config and
    because the plan compiler needs both: the scope supplies the entries and the
    strategy, the groups supply the generated multi-member components that only
    the ``rsa`` strategy consumes.
    """

    scope: ResolvedRandomizationScope = field(
        default_factory=ResolvedRandomizationScope
    )
    """Resolved per-entity and per-camera entries plus the placement strategy."""

    groups: Mapping[str, RandomizationGroupConfig] = field(default_factory=dict)
    """Generated multi-member groups consumed by the ``rsa`` strategy."""

    @property
    def is_empty(self) -> bool:
        """True when nothing in the scope asks for pose randomization."""
        return not (self.scope.entities or self.scope.cameras)

    @classmethod
    def from_scope_config(
        cls,
        config: Optional[RandomizationScopeConfig],
        *,
        groups: Optional[Mapping[str, RandomizationGroupConfig]] = None,
    ) -> "ResolvedRandomizationConfig":
        """Resolve a task's ``randomization`` config into an executor input."""
        if config is None:
            return cls(groups=dict(groups or {}))
        return cls(
            scope=resolve_randomization_scope(config),
            groups=dict(groups or {}),
        )
