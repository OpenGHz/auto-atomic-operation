from __future__ import annotations

import numpy as np
import pytest

from auto_atom.framework import (
    OperatorRandomizationConfig,
    PoseRandomRange,
    RandomizationConstraintConfig,
    RandomizationDistributionConfig,
    RandomizationDistributionKind,
    RandomizationFailureConfig,
    RandomizationFailureMode,
    RandomizationSequenceKind,
    RandomizationSpec,
)
from auto_atom.randomization import (
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


def test_low_discrepancy_candidates_are_deterministic_and_bounded() -> None:
    first = unit_candidate(
        rng=np.random.default_rng(7),
        dimension=4,
        sequence=RandomizationSequenceKind.LOW_DISCREPANCY,
        index=8,
        candidate_count=32,
    )
    second = unit_candidate(
        rng=np.random.default_rng(999),
        dimension=4,
        sequence=RandomizationSequenceKind.LOW_DISCREPANCY,
        index=8,
        candidate_count=32,
    )
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_maximin_select_respects_existing_history() -> None:
    candidates = np.asarray([[0.0], [0.1], [0.5], [0.9]], dtype=np.float64)
    selected = maximin_select(
        candidates,
        count=1,
        seed_points=np.asarray([[0.0]], dtype=np.float64),
    )
    assert np.allclose(selected, [[0.9]])


def test_canonical_failure_policy_is_validated() -> None:
    spec = RandomizationSpec(
        proposal=PoseRandomRange(x=(0.0, 1.0)),
        distribution=RandomizationDistributionConfig(
            kind=RandomizationDistributionKind.SPACE_FILLING,
            sequence=RandomizationSequenceKind.STRATIFIED,
            candidate_count=8,
        ),
        failure=RandomizationFailureConfig(
            mode=RandomizationFailureMode.ERROR,
            max_attempts=3,
        ),
    )
    assert spec.failure.max_attempts == 3
    with pytest.raises(ValueError, match="space_filling requires"):
        RandomizationDistributionConfig(
            kind=RandomizationDistributionKind.SPACE_FILLING,
            sequence=RandomizationSequenceKind.IID,
        )


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
