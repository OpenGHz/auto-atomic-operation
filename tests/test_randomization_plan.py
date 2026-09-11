from __future__ import annotations

import numpy as np
import pytest

from auto_atom.config.randomization import (
    OperatorRandomizationConfig,
    PoseRandomRange,
    RandomizationConstraintConfig,
    RandomizationDistributionConfig,
    RandomizationFailureConfig,
    RandomizationFailureMode,
    RandomizationGeneratorConfig,
    RandomizationGeneratorKind,
    RandomizationGroupConfig,
    RandomizationGroupDistributionConfig,
    RandomizationGroupGeneratorKind,
    RandomizationPoissonDiskConfig,
    RandomizationScopeConfig,
    RandomizationSelectorKind,
    RandomizationStrategy,
    RandomizationSpec,
    ResolvedRandomizationConfig,
    resolve_randomization_scope,
)
from auto_atom.config.task import AutoAtomConfig
from auto_atom.randomization import (
    PoissonDiskCandidateStream,
    RandomizationFailureError,
    compile_randomization_plan,
    maximin_select,
    unit_candidate,
)
from auto_atom.runner.data_replay import DataReplayTaskFileConfig


def test_compile_plan_groups_all_randomized_objects_for_separation() -> None:
    separation = RandomizationConstraintConfig(
        separated={"clearance": 0.02},
    )
    plan = compile_randomization_plan(
        {
            "left": RandomizationSpec(
                proposal=PoseRandomRange(x=(0.0, 1.0)),
                constraints=separation,
            ),
            "middle": PoseRandomRange(x=(0.0, 1.0)),
            "right": PoseRandomRange(x=(0.0, 1.0)),
        },
        object_names={"left", "middle", "right"},
        operator_names=set(),
    )

    assert plan.components == (("left", "middle", "right"),)


def test_compile_plan_automatically_builds_rsa_component_for_overlapping_regions() -> (
    None
):
    plan = compile_randomization_plan(
        {
            "left": PoseRandomRange(x=(0.0, 1.0)),
            "right": PoseRandomRange(x=(0.5, 1.5)),
        },
        object_names={"left", "right"},
        operator_names=set(),
        strategy=RandomizationStrategy.RSA,
    )

    assert plan.components == (("left", "right"),)
    assert len(plan.groups) == 1


def test_compile_plan_joint_rejection_keeps_automatic_component_without_groups() -> (
    None
):
    plan = compile_randomization_plan(
        {
            "left": PoseRandomRange(x=(0.0, 1.0)),
            "right": PoseRandomRange(x=(0.5, 1.5)),
        },
        object_names={"left", "right"},
        operator_names=set(),
        strategy=RandomizationStrategy.JOINT_REJECTION,
    )

    assert plan.components == (("left", "right"),)
    assert plan.groups == {}


def test_compile_plan_keeps_operator_dependency_order() -> None:
    plan = compile_randomization_plan(
        {
            "arm": OperatorRandomizationConfig(
                base=PoseRandomRange(x=(-0.1, 0.1)),
                eef=PoseRandomRange(reference="arm.base", x=(-0.01, 0.01)),
            ),
            "cup": PoseRandomRange(reference="arm.base", x=(-0.01, 0.01)),
        },
        object_names={"cup"},
        operator_names={"arm"},
    )

    assert plan.order.index("arm.base") < plan.order.index("arm.eef")
    assert plan.order.index("arm.base") < plan.order.index("cup")


def _hard_sphere_group(
    members: list[str],
    *,
    mode: RandomizationFailureMode = RandomizationFailureMode.ERROR,
    max_attempts: int = 100,
) -> RandomizationGroupConfig:
    return RandomizationGroupConfig(
        members=members,
        distribution=RandomizationGroupDistributionConfig(
            generator=RandomizationGroupGeneratorKind.HARD_SPHERE_RSA,
        ),
        failure=RandomizationFailureConfig(mode=mode, max_attempts=max_attempts),
    )


def test_compile_plan_connects_hard_sphere_rsa_group_members() -> None:
    group = _hard_sphere_group(["large", "small"])

    plan = compile_randomization_plan(
        {
            "large": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.10),
            "small": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.05),
        },
        object_names={"large", "small"},
        operator_names=set(),
        randomization_groups={"tabletop": group},
    )

    assert plan.components == (("large", "small"),)
    assert plan.groups == {"tabletop": group}


@pytest.mark.parametrize(
    ("randomization", "object_names", "operator_names", "group", "match"),
    [
        (
            {"block": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.05)},
            {"block"},
            set(),
            _hard_sphere_group(["missing", "block"]),
            "unknown",
        ),
        (
            {
                "arm": OperatorRandomizationConfig(
                    base=PoseRandomRange(x=(0.0, 1.0), collision_radius=0.05)
                ),
                "block": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.05),
            },
            {"block"},
            {"arm"},
            _hard_sphere_group(["arm", "block"]),
            "must be an object",
        ),
        (
            {
                "anchor": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.05),
                "block": PoseRandomRange(
                    reference="anchor",
                    x=(0.0, 1.0),
                    collision_radius=0.05,
                ),
            },
            {"anchor", "block"},
            set(),
            _hard_sphere_group(["anchor", "block"]),
            "named entity references",
        ),
        (
            {
                "large": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.10),
                "small": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.0),
            },
            {"large", "small"},
            set(),
            _hard_sphere_group(["large", "small"]),
            "collision_radius > 0",
        ),
        (
            {
                "large": RandomizationSpec(
                    proposal=PoseRandomRange(x=(0.0, 1.0), collision_radius=0.10),
                    constraints=RandomizationConstraintConfig(
                        separated={"clearance": 0.01}
                    ),
                ),
                "small": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.05),
            },
            {"large", "small"},
            set(),
            _hard_sphere_group(["large", "small"]),
            "constraints.separated",
        ),
        (
            {
                "large": RandomizationSpec(
                    proposal=PoseRandomRange(x=(0.0, 1.0), collision_radius=0.10),
                    distribution=RandomizationDistributionConfig(
                        selector=RandomizationSelectorKind.MAXIMIN
                    ),
                ),
                "small": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.05),
            },
            {"large", "small"},
            set(),
            _hard_sphere_group(["large", "small"]),
            "selector=first_feasible",
        ),
    ],
)
def test_compile_plan_rejects_invalid_hard_sphere_rsa_groups(
    randomization,
    object_names,
    operator_names,
    group,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        compile_randomization_plan(
            randomization,
            object_names=object_names,
            operator_names=operator_names,
            randomization_groups={"tabletop": group},
        )


def test_config_selects_automatic_randomization_strategy() -> None:
    config = AutoAtomConfig.model_validate(
        {
            "stages": [],
            "env_name": "randomization_test",
            "randomization": {
                "strategy": "joint_rejection",
                "entities": {
                    "large": {"x": [0.0, 1.0]},
                    "small": {"x": [0.0, 1.0]},
                },
            },
        }
    )
    assert (
        resolve_randomization_scope(config.randomization).strategy
        == RandomizationStrategy.JOINT_REJECTION
    )

    default_config = AutoAtomConfig.model_validate(
        {"stages": [], "env_name": "randomization_test"}
    )
    assert (
        resolve_randomization_scope(default_config.randomization).strategy
        == RandomizationStrategy.RSA
    )


def test_data_replay_disables_object_randomization() -> None:
    config = DataReplayTaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {
                "env_name": "randomization_test",
                "stages": [],
                "randomization": {
                    "entities": {
                        "large": {"x": [0.0, 1.0]},
                        "small": {"x": [0.0, 1.0]},
                    },
                },
            },
        }
    )

    assert config.task.randomization.entities == {}
    assert config.task.randomization.entities == {}
    # Replay drops the entries it must not sample; the master switch stays on so
    # camera randomization keeps working.
    assert config.task.randomization.enabled is True


def test_scope_master_switch_reaches_the_resolved_scope() -> None:
    scope = RandomizationScopeConfig(
        enabled=False,
        entities={"cup": PoseRandomRange(x=(0.0, 1.0))},
        cameras={"head_cam": PoseRandomRange(x=(0.0, 1.0))},
    )

    resolved = resolve_randomization_scope(scope)
    assert resolved.enabled is False
    # The entries stay configured and validated; only application is off.
    assert set(resolved.entities) == {"cup"}
    assert set(resolved.cameras) == {"head_cam"}


def test_scope_applies_combines_the_master_switch_and_emptiness() -> None:
    entry = {"cup": PoseRandomRange(x=(0.0, 1.0))}

    applied = ResolvedRandomizationConfig.from_scope_config(
        RandomizationScopeConfig(entities=dict(entry))
    )
    assert applied.applies is True
    assert applied.is_empty is False

    disabled = ResolvedRandomizationConfig.from_scope_config(
        RandomizationScopeConfig(enabled=False, entities=dict(entry))
    )
    # The entry is still configured, but a disabled scope applies nothing.
    assert disabled.applies is False
    assert disabled.is_empty is False

    assert (
        ResolvedRandomizationConfig.from_scope_config(
            RandomizationScopeConfig()
        ).applies
        is False
    )


def test_scope_defaults_inherited_and_strategy_resolved() -> None:
    scope = RandomizationScopeConfig(
        distribution=RandomizationDistributionConfig(
            generator=RandomizationGeneratorKind.SOBOL,
            spacing=0.05,
        ),
        strategy=RandomizationStrategy.JOINT_REJECTION,
        constraints=RandomizationConstraintConfig(
            separated={"clearance": 0.02},
            failure=RandomizationFailureConfig(max_attempts=7),
        ),
        entities={
            # Bare range inherits scope distribution/constraints/failure.
            "cup": {"x": [0.0, 1.0], "collision_radius": 0.05},
            # Advanced spec is fully explicit; its own constraints/failure win.
            "advanced": RandomizationSpec(
                proposal=PoseRandomRange(x=(0.0, 1.0)),
                distribution=RandomizationDistributionConfig(
                    generator=RandomizationGeneratorKind.IID
                ),
                constraints=RandomizationConstraintConfig(
                    visible_in={"cameras": "all"}
                ),
            ),
            # Operator with a bare base range inherits scope defaults too.
            "arm": OperatorRandomizationConfig(
                base=PoseRandomRange(x=(0.0, 1.0)),
                eef=None,
            ),
        },
    )

    resolved_scope = resolve_randomization_scope(scope)
    assert resolved_scope.strategy == RandomizationStrategy.JOINT_REJECTION
    resolved = resolved_scope.entities

    cup = resolved["cup"]
    assert isinstance(cup, RandomizationSpec)
    assert cup.distribution.generator == RandomizationGeneratorKind.SOBOL
    assert cup.distribution.spacing == 0.05
    assert cup.constraints.separated is not None
    assert cup.constraints.separated.clearance == 0.02
    assert cup.constraints.failure.max_attempts == 7

    advanced = resolved["advanced"]
    assert isinstance(advanced, RandomizationSpec)
    assert advanced.distribution.generator == RandomizationGeneratorKind.IID
    assert advanced.constraints.visible_in is not None
    assert advanced.constraints.separated is None
    assert advanced.constraints.failure.max_attempts == 100

    arm = resolved["arm"]
    assert isinstance(arm, OperatorRandomizationConfig)
    assert isinstance(arm.base, RandomizationSpec)
    assert arm.base.distribution.spacing == 0.05
    assert arm.base.constraints.separated is not None
    assert arm.base.constraints.failure.max_attempts == 7


def test_scope_strategy_is_scope_wide() -> None:
    """There is one placement strategy per reset, declared on the scope."""
    scope = RandomizationScopeConfig(
        strategy=RandomizationStrategy.JOINT_REJECTION,
        entities={"cup": PoseRandomRange(x=(0.0, 1.0))},
    )
    assert (
        resolve_randomization_scope(scope).strategy
        == RandomizationStrategy.JOINT_REJECTION
    )

    # A per-entity spec cannot change it: `RandomizationSpec.constraints` has
    # no strategy anywhere, so there is nothing to disagree with.
    spec = RandomizationSpec(
        proposal=PoseRandomRange(x=(0.0, 1.0)),
        constraints=RandomizationConstraintConfig(separated={"clearance": 0.02}),
    )
    resolved = resolve_randomization_scope(
        RandomizationScopeConfig(
            strategy=RandomizationStrategy.RSA, entities={"cup": spec}
        )
    )
    assert resolved.strategy == RandomizationStrategy.RSA
    assert resolved.entities["cup"].constraints.separated.clearance == 0.02


def test_cameras_inherit_scope_distribution_but_not_constraints() -> None:
    scope = RandomizationScopeConfig(
        distribution=RandomizationDistributionConfig(
            generator=RandomizationGeneratorKind.SOBOL,
            spacing=0.05,
        ),
        strategy=RandomizationStrategy.JOINT_REJECTION,
        constraints=RandomizationConstraintConfig(
            separated={"clearance": 0.02},
            failure=RandomizationFailureConfig(max_attempts=7),
        ),
        cameras={"head_cam": PoseRandomRange(x=(0.0, 1.0))},
    )

    resolved = resolve_randomization_scope(scope)
    assert resolved.strategy == RandomizationStrategy.JOINT_REJECTION

    camera = resolved.cameras["head_cam"]
    assert isinstance(camera, RandomizationSpec)
    # ``distribution`` is a pose-stream property, so cameras inherit it.
    assert camera.distribution.generator == RandomizationGeneratorKind.SOBOL
    assert camera.distribution.spacing == 0.05
    # ``constraints`` is entity-only: separation / visibility / failure have no
    # camera meaning, so the built-in defaults stay in place.
    assert camera.constraints.separated is None
    assert camera.constraints.visible_in is None
    assert camera.constraints.failure.max_attempts == 100


def test_camera_advanced_spec_keeps_its_own_distribution() -> None:
    scope = RandomizationScopeConfig(
        distribution=RandomizationDistributionConfig(
            generator=RandomizationGeneratorKind.SOBOL
        ),
        cameras={
            "head_cam": RandomizationSpec(
                proposal=PoseRandomRange(x=(0.0, 1.0)),
                distribution=RandomizationDistributionConfig(
                    generator=RandomizationGeneratorKind.IID
                ),
            )
        },
    )

    camera = resolve_randomization_scope(scope).cameras["head_cam"]
    assert isinstance(camera, RandomizationSpec)
    assert camera.distribution.generator == RandomizationGeneratorKind.IID


def test_cameras_reject_declared_constraints() -> None:
    with pytest.raises(ValueError, match="must not declare constraints"):
        RandomizationScopeConfig(
            cameras={
                "head_cam": RandomizationSpec(
                    proposal=PoseRandomRange(x=(0.0, 1.0)),
                    constraints=RandomizationConstraintConfig(
                        visible_in={"cameras": "all"}
                    ),
                )
            }
        )

    with pytest.raises(ValueError, match="must not declare constraints"):
        RandomizationScopeConfig(
            cameras={
                "head_cam": RandomizationSpec(
                    proposal=PoseRandomRange(x=(0.0, 1.0)),
                    constraints=RandomizationConstraintConfig(
                        failure=RandomizationFailureConfig(max_attempts=3)
                    ),
                )
            }
        )


def test_camera_entries_stay_out_of_the_separation_strategy() -> None:
    scope = RandomizationScopeConfig(
        strategy=RandomizationStrategy.JOINT_REJECTION,
        entities={"cup": PoseRandomRange(x=(0.0, 1.0))},
        cameras={"head_cam": PoseRandomRange(x=(0.0, 1.0))},
    )

    resolved = resolve_randomization_scope(scope)
    assert resolved.strategy == RandomizationStrategy.JOINT_REJECTION
    camera = resolved.cameras["head_cam"]
    assert isinstance(camera, RandomizationSpec)
    assert camera.constraints.separated is None


def test_collision_radius_negative_marks_auto_and_margin_parseable() -> None:
    rng = PoseRandomRange(
        x=(0.0, 1.0),
        collision_radius=-1,
        collision_margin=0.01,
    )
    assert rng.collision_radius == -1
    assert rng.collision_margin == 0.01
    # 0 stays the explicit "exempt" marker and >0 the explicit radius.
    assert PoseRandomRange(x=(0.0, 1.0), collision_radius=0.0).collision_radius == 0.0
    assert PoseRandomRange(x=(0.0, 1.0), collision_radius=0.04).collision_radius == 0.04


def test_hard_sphere_group_accepts_auto_radius_but_rejects_exempt_member() -> None:
    plan = compile_randomization_plan(
        {
            "large": PoseRandomRange(
                x=(0.0, 1.0),
                collision_radius=0.10,
            ),
            "small": PoseRandomRange(
                x=(0.0, 1.0),
                collision_radius=-1,
                collision_margin=0.005,
            ),
        },
        object_names={"large", "small"},
        operator_names=set(),
        randomization_groups={"tabletop": _hard_sphere_group(["large", "small"])},
    )
    assert set(plan.groups) == {"tabletop"}

    with pytest.raises(ValueError, match="exempt"):
        compile_randomization_plan(
            {
                "large": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.10),
                "small": PoseRandomRange(x=(0.0, 1.0), collision_radius=0.0),
            },
            object_names={"large", "small"},
            operator_names=set(),
            randomization_groups={"tabletop": _hard_sphere_group(["large", "small"])},
        )


def test_halton_candidates_are_deterministic_and_bounded() -> None:
    first = unit_candidate(
        rng=np.random.default_rng(7),
        dimension=4,
        generator=RandomizationGeneratorKind.HALTON,
        index=8,
        candidate_count=32,
    )
    second = unit_candidate(
        rng=np.random.default_rng(999),
        dimension=4,
        generator=RandomizationGeneratorKind.HALTON,
        index=8,
        candidate_count=32,
    )
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_sobol_candidates_use_scipy_qmc() -> None:
    candidate = unit_candidate(
        rng=np.random.default_rng(7),
        dimension=3,
        generator=RandomizationGeneratorKind.SOBOL,
        index=2,
        candidate_count=8,
    )
    assert candidate.shape == (3,)
    assert np.all((candidate >= 0.0) & (candidate <= 1.0))


def test_poisson_disk_generator_accepts_scipy_parameters() -> None:
    generator = RandomizationGeneratorConfig(
        poisson_disk=RandomizationPoissonDiskConfig(
            hypersphere="surface",
            ncandidates=11,
        )
    )
    stream = PoissonDiskCandidateStream(
        generator.poisson_disk,
        lower_bounds=(-0.2, -0.1),
        upper_bounds=(0.4, 0.3),
        radius=0.1,
        seed=7,
    )
    candidate = stream.next()
    assert candidate.shape == (2,)
    assert np.all(candidate >= (-0.2, -0.1))
    assert np.all(candidate <= (0.4, 0.3))


def test_poisson_disk_rejects_abstract_radius_configuration() -> None:
    with pytest.raises(ValueError, match="radius"):
        RandomizationPoissonDiskConfig.model_validate({"radius": 0.2})


def test_poisson_disk_stream_preserves_intra_pool_spacing() -> None:
    stream = PoissonDiskCandidateStream(
        RandomizationPoissonDiskConfig(ncandidates=11),
        lower_bounds=(0.0, 0.0, 0.0),
        upper_bounds=(1.0, 1.0, 1.0),
        radius=0.2,
        seed=7,
    )
    samples = np.asarray([stream.next() for _ in range(4)])
    distances = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2)
    off_diagonal = distances[np.triu_indices(len(samples), k=1)]
    assert np.all(off_diagonal >= 0.2 - 1e-12)


def test_maximin_select_respects_existing_history() -> None:
    candidates = np.asarray([[0.0], [0.1], [0.5], [0.9]], dtype=np.float64)
    selected = maximin_select(
        candidates,
        count=1,
        seed_points=np.asarray([[0.0]], dtype=np.float64),
    )
    assert np.allclose(selected, [[0.9]])


def test_canonical_generator_and_selector_are_validated() -> None:
    spec = RandomizationSpec(
        proposal=PoseRandomRange(x=(0.0, 1.0)),
        distribution=RandomizationDistributionConfig(
            generator=RandomizationGeneratorKind.LATIN_HYPERCUBE,
            selector=RandomizationSelectorKind.MAXIMIN,
            candidate_count=8,
        ),
        constraints=RandomizationConstraintConfig(
            failure=RandomizationFailureConfig(
                mode=RandomizationFailureMode.ERROR,
                max_attempts=3,
            )
        ),
    )
    assert spec.constraints.failure.max_attempts == 3
    assert spec.distribution.selector == RandomizationSelectorKind.MAXIMIN


def test_first_feasible_allows_a_candidate_pool_without_changing_selector_scope() -> (
    None
):
    distribution = RandomizationDistributionConfig(
        generator=RandomizationGeneratorKind.HALTON,
        selector=RandomizationSelectorKind.FIRST_FEASIBLE,
        candidate_count=8,
    )
    assert distribution.candidate_count == 8


def test_failure_error_contains_attempt_and_constraint_details() -> None:
    error = RandomizationFailureError(
        target="object",
        attempts=4,
        violations=["object:collides:other"],
        minimum_clearance=-0.03,
    )
    assert error.target == "object"
    assert error.attempts == 4
    assert error.violations == ("object:collides:other",)
    assert "after 4 attempts" in str(error)
