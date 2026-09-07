from __future__ import annotations

import numpy as np
import pytest

from auto_atom.framework import (
    OperatorRandomizationConfig,
    PoseRandomRange,
    RandomizationConstraintConfig,
    RandomizationDistributionConfig,
    RandomizationFailureConfig,
    RandomizationFailureMode,
    RandomizationGeneratorConfig,
    RandomizationGeneratorKind,
    RandomizationPoissonDiskConfig,
    RandomizationSelectorKind,
    RandomizationSpec,
)
from auto_atom.randomization import (
    PoissonDiskCandidateStream,
    RandomizationFailureError,
    compile_randomization_plan,
    maximin_select,
    unit_candidate,
)


def test_compile_plan_groups_all_randomized_objects_for_separation() -> None:
    separation = RandomizationConstraintConfig(
        separated={"min_distance": 0.02},
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
        failure=RandomizationFailureConfig(
            mode=RandomizationFailureMode.ERROR,
            max_attempts=3,
        ),
    )
    assert spec.failure.max_attempts == 3
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
