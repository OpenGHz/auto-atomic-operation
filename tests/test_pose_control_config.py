import pytest
from pydantic import ValidationError

from auto_atom.framework import (
    AxisAlignmentDirection,
    AxisAlignmentOrientationGoalConfig,
    AxisReference,
    ControlledFrameKind,
    FixedOrientationGoalConfig,
    PoseControlConfig,
)


def _axis_alignment_goal() -> dict[str, object]:
    return {
        "kind": "axis_alignment",
        "controlled_axis": [0.0, 0.0, 1.0],
        "target_axis": {
            "vector": [0.0, 1.0, 0.0],
            "reference": "object",
        },
        "direction": "either",
    }


def test_pose_control_defaults_to_legacy_eef_control() -> None:
    pose = PoseControlConfig(position=(0.1, 0.2, 0.3), orientation=(0, 0, 0, 1))

    assert pose.controlled_frame.kind == ControlledFrameKind.EEF
    assert pose.controlled_frame.frame is None
    assert pose.orientation_goal is None
    assert pose.orientation == (0.0, 0.0, 0.0, 1.0)


def test_held_object_control_accepts_an_optional_object_local_frame() -> None:
    pose = PoseControlConfig.model_validate(
        {
            "controlled_frame": {
                "kind": "held_object",
                "frame": "plate2_site",
            }
        }
    )

    assert pose.controlled_frame.kind == ControlledFrameKind.HELD_OBJECT
    assert pose.controlled_frame.frame == "plate2_site"
    with pytest.raises(ValidationError, match="frozen"):
        pose.controlled_frame.frame = "other_site"


@pytest.mark.parametrize("frame", ["plate2_site", ""])
def test_eef_control_rejects_an_object_local_frame(frame: str) -> None:
    with pytest.raises(ValidationError):
        PoseControlConfig.model_validate(
            {"controlled_frame": {"kind": "eef", "frame": frame}}
        )


def test_orientation_goal_uses_its_discriminator() -> None:
    fixed = PoseControlConfig.model_validate(
        {
            "orientation_goal": {
                "kind": "fixed",
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        }
    )
    aligned = PoseControlConfig.model_validate(
        {"orientation_goal": _axis_alignment_goal()}
    )

    assert isinstance(fixed.orientation_goal, FixedOrientationGoalConfig)
    assert fixed.orientation_goal.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert isinstance(
        aligned.orientation_goal,
        AxisAlignmentOrientationGoalConfig,
    )
    assert aligned.orientation_goal.target_axis.reference == AxisReference.OBJECT
    assert aligned.orientation_goal.direction == AxisAlignmentDirection.EITHER


def test_fixed_orientation_goal_normalizes_its_quaternion() -> None:
    pose = PoseControlConfig.model_validate(
        {
            "orientation_goal": {
                "kind": "fixed",
                "quaternion_xyzw": [0.0, 0.0, 0.0, 2.0],
            }
        }
    )

    assert isinstance(pose.orientation_goal, FixedOrientationGoalConfig)
    assert pose.orientation_goal.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "quaternion",
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, float("nan"), 1.0],
        [0.0, 0.0, float("inf"), 1.0],
    ],
)
def test_fixed_orientation_goal_rejects_invalid_quaternions(
    quaternion: list[float],
) -> None:
    with pytest.raises(ValidationError, match="finite and non-zero"):
        PoseControlConfig.model_validate(
            {
                "orientation_goal": {
                    "kind": "fixed",
                    "quaternion_xyzw": quaternion,
                }
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("controlled_axis", [0.0, 0.0, 0.0], "controlled_axis"),
        ("controlled_axis", [0.0, 2.0, 0.0], "controlled_axis"),
        ("target_axis", [1.0, 1.0, 0.0], "target_axis.vector"),
        ("target_axis", [float("inf"), 0.0, 0.0], "target_axis.vector"),
    ],
)
def test_axis_alignment_requires_finite_unit_vectors(
    field: str,
    value: list[float],
    message: str,
) -> None:
    goal = _axis_alignment_goal()
    if field == "controlled_axis":
        goal[field] = value
    else:
        goal["target_axis"] = {"vector": value, "reference": "world"}

    with pytest.raises(ValidationError, match=message):
        PoseControlConfig.model_validate({"orientation_goal": goal})


@pytest.mark.parametrize("reference", ["eef", "eef_world", "object_world", "auto"])
def test_target_axis_only_accepts_supported_references(reference: str) -> None:
    goal = _axis_alignment_goal()
    goal["target_axis"] = {"vector": [1.0, 0.0, 0.0], "reference": reference}

    with pytest.raises(ValidationError):
        PoseControlConfig.model_validate({"orientation_goal": goal})


@pytest.mark.parametrize(
    "legacy_orientation",
    [
        {"orientation": [0.0, 0.0, 0.0, 1.0]},
        {"rotation": [0.0, 0.0, 0.0]},
    ],
)
def test_orientation_goal_rejects_legacy_orientation_fields(
    legacy_orientation: dict[str, object],
) -> None:
    with pytest.raises(
        ValidationError,
        match="orientation_goal cannot be combined",
    ):
        PoseControlConfig.model_validate(
            {
                **legacy_orientation,
                "orientation_goal": _axis_alignment_goal(),
            }
        )


@pytest.mark.parametrize(
    "orientation_goal",
    [
        {
            "kind": "fixed",
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        _axis_alignment_goal(),
    ],
)
def test_orientation_goal_rejects_rotational_randomization(
    orientation_goal: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="rotational randomization"):
        PoseControlConfig.model_validate(
            {
                "orientation_goal": orientation_goal,
                "randomization": {"yaw": [-0.1, 0.1]},
            }
        )


def test_orientation_goal_accepts_position_randomization() -> None:
    pose = PoseControlConfig.model_validate(
        {
            "orientation_goal": _axis_alignment_goal(),
            "randomization": {"x": [-0.1, 0.1]},
        }
    )

    assert pose.randomization is not None
    assert pose.randomization.x == (-0.1, 0.1)


def test_axis_alignment_rejects_relative_waypoints() -> None:
    with pytest.raises(ValidationError, match="does not support relative=true"):
        PoseControlConfig.model_validate(
            {
                "relative": True,
                "orientation_goal": _axis_alignment_goal(),
            }
        )


def test_axis_alignment_rejects_arc_waypoints() -> None:
    with pytest.raises(ValidationError, match="does not support arc movement"):
        PoseControlConfig.model_validate(
            {
                "arc": {
                    "pivot": [0.0, 0.0, 0.0],
                    "axis": [0.0, 0.0, 1.0],
                    "angle": 0.5,
                },
                "orientation_goal": _axis_alignment_goal(),
            }
        )


def test_fixed_orientation_goal_rejects_arc_waypoints() -> None:
    with pytest.raises(ValidationError, match="orientation_goal does not support arc"):
        PoseControlConfig.model_validate(
            {
                "arc": {
                    "pivot": [0.0, 0.0, 0.0],
                    "axis": [0.0, 0.0, 1.0],
                    "angle": 0.5,
                },
                "orientation_goal": {
                    "kind": "fixed",
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        )


def test_held_object_control_rejects_arc_waypoints() -> None:
    with pytest.raises(
        ValidationError,
        match="held_object controlled_frame does not support arc movement",
    ):
        PoseControlConfig.model_validate(
            {
                "controlled_frame": {"kind": "held_object"},
                "arc": {
                    "pivot": [0.0, 0.0, 0.0],
                    "axis": [0.0, 0.0, 1.0],
                    "angle": 0.5,
                },
            }
        )


def test_new_nested_configs_forbid_unknown_fields() -> None:
    goal = _axis_alignment_goal()
    goal["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        PoseControlConfig.model_validate({"orientation_goal": goal})
