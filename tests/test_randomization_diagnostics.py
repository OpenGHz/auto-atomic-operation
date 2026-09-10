"""Unit tests for the backend-neutral randomization diagnostics.

Covers configuration validation (reference frames and target forms), the
world-AABB/depth reasoning used by the deterministic ``visible_in`` preflight,
and the preflight itself. All of it runs without a simulator.
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
    OperatorRandomizationConfig,
    PoseRandomRange,
    RandomizationConstraintConfig,
    RandomizationFailureConfig,
    RandomizationSpec,
)
from auto_atom.contracts import CameraModel
from auto_atom.randomization import (
    camera_frustum_disjoint_box,
    find_visibility_infeasibility,
    object_region_world_box,
    reference_ancestors,
    validate_pose_randomization_spec,
    validate_randomization_configuration,
)
from auto_atom.utils.pose import PoseState

# Identity orientation: the camera looks along -z with +y up and +x right.
CAMERA = CameraModel(
    name="head_cam",
    pose=PoseState(),
    width=640,
    height=480,
    fovy_radians=1.0,
    near=0.1,
    far=10.0,
)


def _spec(visible_in=None, failure_mode="error") -> RandomizationSpec:
    return RandomizationSpec(
        proposal=PoseRandomRange(x=(0.0, 1.0)),
        constraints=RandomizationConstraintConfig(
            visible_in=visible_in,
            failure=RandomizationFailureConfig(mode=failure_mode),
        ),
    )


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


def test_absolute_base_is_rejected_outside_an_end_effector() -> None:
    spec = PoseRandomRange.model_validate(
        {"reference": "absolute_base", "x": [0.0, 1.0]}
    )
    with pytest.raises(ValueError, match="only operator end-effector"):
        validate_pose_randomization_spec(
            "Object 'cup'",
            spec,
            allow_absolute_base=False,
            object_names={"cup"},
            operator_names={"arm"},
        )
    validate_pose_randomization_spec(
        "Operator 'arm' end effector",
        spec,
        allow_absolute_base=True,
        object_names={"cup"},
        operator_names={"arm"},
    )


def test_absolute_base_cannot_be_mixed_with_other_frames() -> None:
    spec = PoseRandomRange.model_validate(
        {
            "x": {"range": [0.0, 1.0], "reference": "absolute_base"},
            "y": {"range": [0.0, 1.0], "reference": "absolute_world"},
        }
    )
    with pytest.raises(ValueError, match="cannot mix 'absolute_base'"):
        validate_pose_randomization_spec(
            "Operator 'arm' end effector",
            spec,
            allow_absolute_base=True,
            object_names=set(),
            operator_names={"arm"},
        )


def test_unknown_or_non_operator_references_are_rejected() -> None:
    with pytest.raises(ValueError, match="is not a known object or operator"):
        validate_pose_randomization_spec(
            "Object 'cup'",
            PoseRandomRange.model_validate(
                {"x": {"range": [0.0, 1.0], "reference": "ghost"}}
            ),
            allow_absolute_base=False,
            object_names={"cup"},
            operator_names={"arm"},
        )

    with pytest.raises(ValueError, match="is not a known operator"):
        validate_pose_randomization_spec(
            "Object 'cup'",
            PoseRandomRange.model_validate(
                {"x": {"range": [0.0, 1.0], "reference": "cup.eef"}}
            ),
            allow_absolute_base=False,
            object_names={"cup"},
            operator_names={"arm"},
        )


def test_operator_attribute_references_are_accepted() -> None:
    validate_pose_randomization_spec(
        "Object 'cup'",
        PoseRandomRange.model_validate(
            {
                "x": {"range": [0.0, 1.0], "reference": "arm.base"},
                "y": {"range": [0.0, 1.0], "reference": "arm.eef"},
            }
        ),
        allow_absolute_base=False,
        object_names={"cup"},
        operator_names={"arm"},
    )


def test_target_form_must_match_the_target_kind() -> None:
    with pytest.raises(TypeError, match="not an operator randomization config"):
        validate_randomization_configuration(
            {"cup": OperatorRandomizationConfig(base=PoseRandomRange(x=(0.0, 1.0)))},
            object_names={"cup"},
            operator_names={"arm"},
        )

    with pytest.raises(TypeError, match="nested form with explicit"):
        validate_randomization_configuration(
            {"arm": PoseRandomRange(x=(0.0, 1.0))},
            object_names={"cup"},
            operator_names={"arm"},
        )


def test_unknown_configuration_keys_are_returned_not_swallowed() -> None:
    unknown = validate_randomization_configuration(
        {
            "cup": PoseRandomRange(x=(0.0, 1.0)),
            "arm": OperatorRandomizationConfig(
                base=PoseRandomRange(x=(0.0, 1.0)),
                eef=PoseRandomRange.model_validate(
                    {"reference": "absolute_base", "z": [0.0, 1.0]}
                ),
            ),
            "typo": PoseRandomRange(x=(0.0, 1.0)),
        },
        object_names={"cup"},
        operator_names={"arm"},
    )
    assert unknown == ("typo",)


def test_reference_ancestors_resolve_operator_shorthand_and_transitivity() -> None:
    ancestors = reference_ancestors(
        ("arm.base", "cup", "arm.eef"),
        {"arm.base": {"torso"}, "cup": {"tray"}},
        operator_names={"arm"},
    )
    assert ancestors == {"arm", "cup", "torso", "tray"}

    # A bare operator name is shorthand for its base, matching the plan.
    assert reference_ancestors(("arm",), {}, operator_names={"arm"}) == {"arm"}


# ---------------------------------------------------------------------------
# World box and frustum predicate
# ---------------------------------------------------------------------------


def test_world_box_offsets_relative_ranges_and_keeps_unconfigured_axes() -> None:
    lower, upper = object_region_world_box(
        PoseRandomRange(x=(0.0, 1.0)),
        np.asarray([5.0, 2.0, 3.0]),
    )
    assert np.allclose(lower, [5.0, 2.0, 3.0])
    assert np.allclose(upper, [6.0, 2.0, 3.0])

    lower, upper = object_region_world_box(
        PoseRandomRange.model_validate(
            {"reference": "absolute_world", "x": [1.0, 0.0]}
        ),
        np.asarray([5.0, 2.0, 3.0]),
    )
    assert np.allclose(lower, [0.0, 2.0, 3.0])
    assert np.allclose(upper, [1.0, 2.0, 3.0])


def test_world_box_is_undefined_for_entity_tracked_axes() -> None:
    assert (
        object_region_world_box(
            PoseRandomRange.model_validate(
                {"x": {"range": [0.0, 1.0], "reference": "tray"}}
            ),
            np.zeros(3),
        )
        is None
    )
    assert (
        object_region_world_box(
            PoseRandomRange.model_validate(
                {"reference": "absolute_base", "x": [0.0, 1.0]}
            ),
            np.zeros(3),
        )
        is None
    )


@pytest.mark.parametrize(
    ("center", "expected"),
    [
        ((0.0, 0.0, -2.0), False),  # in front of the camera
        ((0.0, 0.0, 2.0), True),  # behind it
        ((0.0, 0.0, -20.0), True),  # beyond far
        ((0.0, 0.0, -0.01), True),  # inside near
        ((10.0, 0.0, -2.0), True),  # far to the right
        ((0.0, 10.0, -2.0), True),  # far above
    ],
)
def test_frustum_disjoint_box(center, expected) -> None:
    box = np.asarray(center, dtype=np.float64)
    assert camera_frustum_disjoint_box(CAMERA, box, box) is expected


def test_degenerate_clip_range_is_disjoint() -> None:
    camera = CameraModel(
        name="head_cam",
        pose=PoseState(),
        width=640,
        height=480,
        fovy_radians=1.0,
        near=5.0,
        far=1.0,
    )
    box = np.zeros(3)
    assert camera_frustum_disjoint_box(camera, box, box)


# ---------------------------------------------------------------------------
# Deterministic visibility preflight
# ---------------------------------------------------------------------------


def _preflight(
    targets,
    *,
    default_pose=None,
    camera_names=("head_cam",),
    env_mask=None,
    camera=CAMERA,
):
    pose = (
        PoseState(position=(0.0, 0.0, -2.0)) if default_pose is None else default_pose
    )
    return find_visibility_infeasibility(
        targets,
        env_mask=(np.asarray([True], dtype=bool) if env_mask is None else env_mask),
        default_pose_of=lambda _name: pose,
        camera_names_of=lambda visible, _env_index: (
            list(visible.cameras) if visible.cameras != "all" else list(camera_names)
        ),
        camera_model_of=lambda _name, _env_index: camera,
    )


def test_preflight_reports_a_box_that_can_never_be_visible() -> None:
    infeasibility = _preflight(
        {"cup": _spec(visible_in={"cameras": ["head_cam"]})},
        default_pose=PoseState(position=(0.0, 0.0, 2.0)),
    )
    assert infeasibility is not None
    assert infeasibility.target == "cup"
    assert infeasibility.violations == ("cup:outside_view:head_cam",)


def test_preflight_is_silent_when_the_region_reaches_the_frustum() -> None:
    assert _preflight({"cup": _spec(visible_in={"cameras": ["head_cam"]})}) is None


def test_preflight_leaves_non_fail_closed_targets_to_the_retry_loop() -> None:
    assert (
        _preflight(
            {
                "cup": _spec(
                    visible_in={"cameras": ["head_cam"]},
                    failure_mode="best_effort",
                )
            },
            default_pose=PoseState(position=(0.0, 0.0, 2.0)),
        )
        is None
    )


def test_preflight_skips_constraint_free_and_masked_out_targets() -> None:
    behind = PoseState(position=(0.0, 0.0, 2.0))
    assert _preflight({"cup": _spec()}, default_pose=behind) is None
    assert (
        _preflight(
            {"cup": _spec(visible_in={"cameras": ["head_cam"]})},
            default_pose=behind,
            env_mask=np.asarray([False], dtype=bool),
        )
        is None
    )
    assert (
        find_visibility_infeasibility(
            {"cup": _spec(visible_in={"cameras": ["head_cam"]})},
            env_mask=np.asarray([True], dtype=bool),
            default_pose_of=lambda _name: None,
            camera_names_of=lambda _visible, _env_index: ["head_cam"],
            camera_model_of=lambda _name, _env_index: CAMERA,
        )
        is None
    )
