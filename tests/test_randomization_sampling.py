"""Unit tests for the backend-neutral randomization sampling primitives.

These run the pure helpers directly, with no simulator: axis/reference
semantics, the fixed random-draw order that keeps resets reproducible, region
selection weighting, and collision/radius resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.config.randomization import (
    PoseRandomizationConfig,
    PoseRandomRange,
    RandomizationDistributionConfig,
    RandomizationGeneratorKind,
    RandomizationSpec,
)
from auto_atom.config.reference import RandomizationReference
from auto_atom.randomization import (
    CollisionParticipant,
    distribution_uses_space_filling_history,
    find_collision_participant,
    history_clearance,
    resolve_collision_ancestors,
    resolve_collision_radius,
    sample_pose_batch,
    sample_pose_for_env,
    select_randomization_region,
)
from auto_atom.utils.pose import PoseState


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> PoseState:
    return PoseState(position=(x, y, z))


# ---------------------------------------------------------------------------
# Axis and reference semantics
# ---------------------------------------------------------------------------


def test_unconfigured_axes_keep_the_baseline() -> None:
    rng = np.random.default_rng(0)
    sampled = sample_pose_for_env(
        rng,
        base_pose=_pose(1.0, 2.0, 3.0),
        rand_range=PoseRandomRange(y=(0.0, 0.0)),
        env_index=0,
        batch_size=1,
    )
    assert np.allclose(sampled.position[0], [1.0, 2.0, 3.0])


def test_relative_axis_offsets_from_its_own_reference_baseline() -> None:
    rng = np.random.default_rng(0)
    reference = PoseState(position=np.asarray([[10.0, 0.0, 0.0]]))
    sampled = sample_pose_for_env(
        rng,
        base_pose=_pose(1.0, 0.0, 0.0),
        rand_range=PoseRandomRange.model_validate(
            {"x": {"range": [0.5, 0.5], "reference": "anchor"}}
        ),
        env_index=0,
        batch_size=1,
        reference_poses={"anchor": reference},
    )
    # ``anchor`` is the baseline for the x axis: 10.0 + 0.5.
    assert sampled.position[0][0] == pytest.approx(10.5)


def test_absolute_world_axis_replaces_rather_than_offsets() -> None:
    rng = np.random.default_rng(0)
    sampled = sample_pose_for_env(
        rng,
        base_pose=_pose(1.0, 0.0, 0.0),
        rand_range=PoseRandomRange.model_validate(
            {
                "reference": "absolute_world",
                "x": [0.5, 0.5],
            }
        ),
        env_index=0,
        batch_size=1,
    )
    assert sampled.position[0][0] == pytest.approx(0.5)


def test_sampling_is_reproducible_for_a_given_seed() -> None:
    kwargs = dict(
        base_pose=_pose(),
        rand_range=PoseRandomRange(
            x=(0.0, 1.0), y=(0.0, 1.0), z=(0.0, 1.0), yaw=(0.0, 1.0)
        ),
        env_index=0,
        batch_size=1,
        sample_index=3,
        reset_index=1,
    )
    first = sample_pose_for_env(np.random.default_rng(7), **kwargs)
    second = sample_pose_for_env(np.random.default_rng(7), **kwargs)
    assert np.allclose(first.position, second.position)
    assert np.allclose(first.orientation, second.orientation)


def test_channels_are_batched_and_only_enabled_rows_are_sampled() -> None:
    rng = np.random.default_rng(0)
    sampled = sample_pose_batch(
        rng,
        base_pose=_pose(1.0, 0.0, 0.0),
        rand_range=PoseRandomRange(x=(0.5, 0.5)),
        env_mask=np.asarray([True, False, True], dtype=bool),
        batch_size=3,
    )
    assert sampled.batch_size == 3
    assert sampled.position[0][0] == pytest.approx(1.5)
    # The masked-out environment keeps the baseline untouched.
    assert sampled.position[1][0] == pytest.approx(1.0)
    assert sampled.position[2][0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------


def test_single_region_selection_consumes_no_randomness() -> None:
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    selected = select_randomization_region(rng, PoseRandomRange(x=(0.0, 1.0)))
    assert selected.axis_range("x") == (0.0, 1.0)
    # A legacy single-region range must not shift the stream.
    assert rng.bit_generator.state == before


def test_volume_weighting_favours_the_larger_region() -> None:
    spec = RandomizationSpec(
        proposal=PoseRandomizationConfig(
            regions=[
                PoseRandomRange(x=(0.0, 1.0)),
                PoseRandomRange(x=(0.0, 9.0)),
            ]
        ),
        distribution=RandomizationDistributionConfig(region_weighting="volume"),
    )
    rng = np.random.default_rng(0)
    picks = [
        select_randomization_region(rng, spec).axis_range("x") for _ in range(2000)
    ]
    large = sum(1 for pick in picks if pick == (0.0, 9.0))
    # Volumes are 1 and 9, so the large region wins about 90% of the draws.
    assert 0.85 < large / len(picks) < 0.95


def test_equal_weighting_splits_regions_evenly() -> None:
    spec = RandomizationSpec(
        proposal=PoseRandomizationConfig(
            regions=[
                PoseRandomRange(x=(0.0, 1.0)),
                PoseRandomRange(x=(0.0, 9.0)),
            ]
        ),
        distribution=RandomizationDistributionConfig(region_weighting="equal"),
    )
    rng = np.random.default_rng(0)
    picks = [
        select_randomization_region(rng, spec).axis_range("x") for _ in range(2000)
    ]
    large = sum(1 for pick in picks if pick == (0.0, 9.0))
    assert 0.45 < large / len(picks) < 0.55


# ---------------------------------------------------------------------------
# Collision participants
# ---------------------------------------------------------------------------


def test_collision_uses_the_radius_sum_only() -> None:
    participant = CollisionParticipant(
        owner="other",
        label="other",
        pose=_pose(1.0, 0.0, 0.0),
        radius=0.3,
    )
    # Centre distance is 1.0; the candidate is blocked once
    # collision_radius + 0.3 exceeds it.
    assert (
        find_collision_participant(
            owner_name="cup",
            env_index=0,
            candidate_pose=_pose(0.0, 0.0, 0.0),
            collision_radius=0.8,
            ancestors=set(),
            collision_participants=[participant],
        )
        is participant
    )
    # 0.6 + 0.3 = 0.9 < 1.0, so the candidate is clear.
    assert (
        find_collision_participant(
            owner_name="cup",
            env_index=0,
            candidate_pose=_pose(0.0, 0.0, 0.0),
            collision_radius=0.6,
            ancestors=set(),
            collision_participants=[participant],
        )
        is None
    )
    # ``extra_clearance`` adds to the radius sum, not to the geometry.
    assert (
        find_collision_participant(
            owner_name="cup",
            env_index=0,
            candidate_pose=_pose(0.0, 0.0, 0.0),
            collision_radius=0.6,
            ancestors=set(),
            collision_participants=[participant],
            extra_clearance=0.2,
        )
        is participant
    )


def test_exempt_and_self_and_ancestor_participants_are_skipped() -> None:
    participant = CollisionParticipant(
        owner="other",
        label="other",
        pose=_pose(0.0, 0.0, 0.0),
        radius=1.0,
    )
    common = dict(
        owner_name="cup",
        env_index=0,
        candidate_pose=_pose(0.0, 0.0, 0.0),
        collision_participants=[participant],
    )
    # Radius 0 means "exempt from collision rejection".
    assert (
        find_collision_participant(collision_radius=0.0, ancestors=set(), **common)
        is None
    )
    # A participant owned by the candidate itself is never self-colliding.
    assert (
        find_collision_participant(
            collision_radius=1.0,
            ancestors=set(),
            **{**common, "owner_name": "other"},
        )
        is None
    )
    # Articulated pairs are exempt in both directions.
    assert (
        find_collision_participant(
            collision_radius=1.0,
            ancestors={"other"},
            **common,
        )
        is None
    )
    assert (
        find_collision_participant(
            collision_radius=1.0,
            ancestors=set(),
            **{
                **common,
                "collision_participants": [
                    CollisionParticipant(
                        owner="other",
                        label="other",
                        pose=_pose(0.0, 0.0, 0.0),
                        radius=1.0,
                        ancestors={"cup"},
                    )
                ],
            },
        )
        is None
    )


def test_batched_participant_radius_selects_the_environment() -> None:
    participant = CollisionParticipant(
        owner="other",
        label="other",
        pose=_pose(1.0, 0.0, 0.0),
        radius=np.asarray([0.0, 0.9]),
    )
    assert (
        find_collision_participant(
            owner_name="cup",
            env_index=0,
            candidate_pose=_pose(),
            collision_radius=0.5,
            ancestors=set(),
            collision_participants=[participant],
        )
        is None
    )
    assert (
        find_collision_participant(
            owner_name="cup",
            env_index=1,
            candidate_pose=_pose(),
            collision_radius=0.5,
            ancestors=set(),
            collision_participants=[participant],
        )
        is participant
    )


@pytest.mark.parametrize(
    ("radius", "env_index", "expected"),
    [
        (0.4, 0, 0.4),
        (np.asarray([]), 0, 0.0),
        (np.asarray([0.4]), 3, 0.4),
        (np.asarray([0.4, 0.9]), 1, 0.9),
    ],
)
def test_collision_radius_resolution(radius, env_index, expected) -> None:
    assert resolve_collision_radius(radius, env_index) == pytest.approx(expected)


def test_collision_ancestor_resolution() -> None:
    assert resolve_collision_ancestors({"a"}, 0) == {"a"}
    assert resolve_collision_ancestors([], 0) == set()
    assert resolve_collision_ancestors([{"a"}], 4) == {"a"}
    assert resolve_collision_ancestors([{"a"}, {"b"}], 1) == {"b"}


# ---------------------------------------------------------------------------
# Space-filling history
# ---------------------------------------------------------------------------


def test_history_clearance_measures_the_nearest_accepted_sample() -> None:
    assert history_clearance(np.zeros(3), []) == float("inf")
    history = [np.asarray([1.0, 0.0, 0.0]), np.asarray([4.0, 0.0, 0.0])]
    assert history_clearance(np.zeros(3), history) == pytest.approx(1.0)


def test_only_non_iid_generators_use_cross_reset_history() -> None:
    assert not distribution_uses_space_filling_history(None)
    assert not distribution_uses_space_filling_history(
        RandomizationDistributionConfig(generator=RandomizationGeneratorKind.IID)
    )
    for generator in (
        RandomizationGeneratorKind.SOBOL,
        RandomizationGeneratorKind.POISSON_DISK,
    ):
        assert distribution_uses_space_filling_history(
            RandomizationDistributionConfig(generator=generator)
        )


def test_absolute_base_reference_is_supported_for_the_base_action() -> None:
    rng = np.random.default_rng(0)
    sampled = sample_pose_for_env(
        rng,
        base_pose=_pose(1.0, 0.0, 0.0),
        rand_range=PoseRandomRange.model_validate(
            {
                "reference": "absolute_base",
                "x": [0.25, 0.25],
            }
        ),
        env_index=0,
        batch_size=1,
    )
    assert sampled.position[0][0] == pytest.approx(0.25)
    # The relative default is what makes ``absolute_base`` distinguishable.
    assert RandomizationReference.RELATIVE.value == "relative"
