"""The ``nearest_feasible`` orientation goal: solver, grouping and runtime."""

from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from auto_atom.backend.mjc.clearance import (
    capture_template,
    check_paths,
    obstacle_geoms,
    probe_geoms,
    subtree_bodies,
)
from auto_atom.config.motion import PoseControlConfig, StageConfig
from auto_atom.contracts import ClearanceReport
from auto_atom.execution_model import PrimitiveAction
from auto_atom.execution_timeline import TaskFlowBuilder
from auto_atom.mock import MockObjectHandler, MockOperatorHandler, MockSceneBackend
from auto_atom.pose_goal import (
    nearest_feasible_candidates,
    nearest_feasible_orientations,
)
from auto_atom.runtime import GraspBinding, TaskRunner
from auto_atom.utils.pose import (
    PoseState,
    compose_pose,
    quaternion_angular_distance,
    quaternion_to_rotation_matrix,
)

_HALF_PI = float(np.pi / 2.0)
# Jaw axis level (side by side) or vertical (one above the other).
_JAW_BANDS = ((-0.524, 0.524), (1.047, _HALF_PI), (-_HALF_PI, -1.047))


def _quaternion(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.asarray([*(unit * np.sin(angle / 2.0)), np.cos(angle / 2.0)])


def _compose(*quaternions: np.ndarray) -> np.ndarray:
    pose = PoseState(orientation=quaternions[0])
    for quaternion in quaternions[1:]:
        pose = compose_pose(pose, PoseState(orientation=quaternion))
    return pose.orientation[0]


def _elevation(rotation: np.ndarray, axis: tuple[float, float, float]) -> float:
    return float(np.degrees(np.arcsin(np.clip((rotation @ axis)[2], -1.0, 1.0))))


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------


@pytest.mark.parametrize("heading", [-2.9, -0.4, 1.3, 3.1])
@pytest.mark.parametrize("spin", [-1.2, 0.0, 0.3])
def test_a_feasible_orientation_is_kept_unrotated(heading: float, spin: float) -> None:
    current = _compose(_quaternion((0, 0, 1), heading), _quaternion((1, 0, 0), spin))

    rotations, angles = nearest_feasible_orientations(
        current, (0, 0, 0, 1), (1, 0, 0), primary_elevation=((0.0, 0.0),)
    )

    assert angles[0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(
        rotations[0], quaternion_to_rotation_matrix(current), atol=1e-9
    )


def test_a_tilted_axis_is_levelled_by_the_tilt_alone() -> None:
    reference = _quaternion((0, 0, 1), 0.7)
    current = _compose(
        reference, _quaternion((0, 0, 1), 0.4), _quaternion((0, 1, 0), -0.35)
    )

    rotations, angles = nearest_feasible_orientations(
        current, reference, (1, 0, 0), primary_elevation=((0.0, 0.0),)
    )

    assert angles[0] == pytest.approx(0.35, abs=1e-9)
    in_reference = quaternion_to_rotation_matrix(reference).T @ rotations[0]
    assert _elevation(in_reference, (1, 0, 0)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("jaw_tilt_deg", "expected_jaw_deg", "expected_rotation_deg"),
    [(35.0, 30.0, 5.0), (50.0, 60.0, 10.0), (-80.0, -80.0, 0.0)],
)
def test_alternative_bands_pick_the_nearest_posture(
    jaw_tilt_deg: float, expected_jaw_deg: float, expected_rotation_deg: float
) -> None:
    current = _quaternion((1, 0, 0), np.radians(jaw_tilt_deg))

    rotations, angles = nearest_feasible_orientations(
        current,
        (0, 0, 0, 1),
        (1, 0, 0),
        primary_elevation=((0.0, 0.0),),
        constraints=[((0, 1, 0), _JAW_BANDS, None)],
    )

    assert np.degrees(angles[0]) == pytest.approx(expected_rotation_deg, abs=0.05)
    assert _elevation(rotations[0], (0, 1, 0)) == pytest.approx(
        expected_jaw_deg, abs=0.05
    )


def test_every_candidate_satisfies_every_constraint_in_ascending_order() -> None:
    current = _compose(_quaternion((0, 1, 1), 0.8), _quaternion((1, 0, 0), 0.4))

    rotations, angles = nearest_feasible_orientations(
        current,
        (0, 0, 0, 1),
        (1, 0, 0),
        primary_elevation=((-0.1, 0.1),),
        primary_azimuth=((-1.0, 0.5),),
        constraints=[((0, 1, 0), _JAW_BANDS, None)],
    )

    assert len(rotations) > 0
    assert np.all(np.diff(angles) >= 0.0)
    long_axes = rotations[:, :, 0]
    jaw_axes = rotations[:, :, 1]
    assert np.all(np.abs(np.arcsin(long_axes[:, 2])) <= 0.1 + 1e-9)
    azimuths = np.arctan2(long_axes[:, 1], long_axes[:, 0])
    assert np.all((azimuths >= -1.0 - 1e-9) & (azimuths <= 0.5 + 1e-9))
    jaw_elevations = np.abs(np.arcsin(np.clip(jaw_axes[:, 2], -1.0, 1.0)))
    assert np.all((jaw_elevations <= 0.524 + 1e-9) | (jaw_elevations >= 1.047 - 1e-9))


def test_postures_merge_nearest_first_and_report_which_posture_each_is() -> None:
    # Jaws 50 deg up: level is 20 deg away, upright 10 deg.
    current = _quaternion((1, 0, 0), np.radians(50.0))
    level = [((0, 1, 0), ((-0.524, 0.524),), None)]
    upright = [((0, 1, 0), ((1.047, _HALF_PI), (-_HALF_PI, -1.047)), None)]

    rotations, angles, postures = nearest_feasible_candidates(
        current,
        (0, 0, 0, 1),
        (1, 0, 0),
        primary_elevation=((0.0, 0.0),),
        postures=[level, upright],
    )

    assert postures[0] == 1
    assert np.degrees(angles[0]) == pytest.approx(10.0, abs=0.05)
    first_level = int(np.argmax(postures == 0))
    assert np.degrees(angles[first_level]) == pytest.approx(20.0, abs=0.05)
    assert np.all(np.diff(angles) >= 0.0)
    assert set(postures.tolist()) == {0, 1}

    # Without postures every candidate belongs to the single implicit one.
    _, _, single = nearest_feasible_candidates(
        current, (0, 0, 0, 1), (1, 0, 0), primary_elevation=((0.0, 0.0),)
    )
    assert set(single.tolist()) == {0}


def test_contradictory_constraints_leave_no_candidate() -> None:
    rotations, angles = nearest_feasible_orientations(
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0),
        primary_elevation=((0.0, 0.0),),
        constraints=[((1, 0, 0), ((0.5, 0.6),), None)],
    )

    assert rotations.shape == (0, 3, 3)
    assert angles.shape == (0,)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def _goal(**overrides: object) -> dict[str, object]:
    goal: dict[str, object] = {
        "kind": "nearest_feasible",
        "reference": "object",
        "primary_axis": {"axis": [1, 0, 0], "elevation": [[0.0, 0.0]]},
        "constraints": [
            {
                "frame": "eef",
                "axis": [0, 1, 0],
                "elevation": [[-0.5, 0.5], [1.0, 1.5708], [-1.5708, -1.0]],
            }
        ],
    }
    goal.update(overrides)
    return goal


def _held(goal: dict[str, object], **fields: object) -> dict[str, object]:
    return {
        "controlled_frame": {"kind": "held_object"},
        "position": fields.pop("position", [0.0, 0.0, 0.0]),
        "reference": fields.pop("reference", "object"),
        "orientation_goal": goal,
        **fields,
    }


def test_rounded_half_pi_is_clamped_and_goals_are_hashable() -> None:
    first = PoseControlConfig.model_validate(_held(_goal()))
    second = PoseControlConfig.model_validate(_held(_goal(), position=[1.0, 0, 0]))

    assert first.orientation_goal.constraints[0].elevation[1] == (1.0, _HALF_PI)
    assert hash(first.orientation_goal) == hash(second.orientation_goal)


@pytest.mark.parametrize(
    "goal",
    [
        _goal(reference="base"),
        _goal(constraints=[{"axis": [0, 1, 0]}]),
        _goal(primary_axis={"axis": [1, 0, 0], "elevation": [[0.2, 0.1]]}),
        _goal(primary_axis={"axis": [1, 0, 0], "elevation": [[0.0, 2.0]]}),
        _goal(primary_axis={"axis": [1, 1, 0]}),
        _goal(clearance={"bodies": []}),
        _goal(resolution=0.0),
        _goal(postures=[{"name": "a"}, {"name": "a"}]),
        _goal(postures=[{"name": " "}]),
    ],
)
def test_invalid_goals_are_rejected(goal: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PoseControlConfig.model_validate(_held(goal))


@pytest.mark.parametrize("fields", [{"relative": True}, {"reference": "eef_world"}])
def test_waypoint_level_combinations_are_rejected(fields: dict[str, object]) -> None:
    goal = _goal(clearance={"bodies": ["microwave"]})
    with pytest.raises(ValidationError):
        PoseControlConfig.model_validate(_held(goal, **fields))


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------


def _place_stage(goal: dict[str, object]) -> StageConfig:
    return StageConfig.model_validate(
        {
            "object": "target",
            "operation": "place",
            "param": {
                "pre_move": [
                    _held(goal, position=[-0.30, 0.0, 0.02]),
                    _held(goal, position=[0.0, 0.0, 0.02]),
                    _held(dict(goal, resolution=0.1), position=[0.0, 0.0, 0.01]),
                    _held(goal, position=[0.0, 0.0, 0.0]),
                ],
                "eef": {"close": False},
                "post_move": [
                    {"position": [0.0, -0.25, 0.0], "reference": "eef_world"}
                ],
            },
        }
    )


def test_equal_goals_share_one_group_that_survives_a_deep_copy() -> None:
    actions, _ = TaskFlowBuilder.build_actions(_place_stage(_goal()))
    poses = [action for action in actions if action.phase.value == "pre_move"]
    group = poses[0].orientation_group

    assert group is poses[1].orientation_group is poses[3].orientation_group
    assert poses[2].orientation_group is not group
    assert group.members == [poses[0], poses[1], poses[3]]
    # The group placing the object owns the release and what follows it.
    release = actions[4]
    assert release.kind == "eef" and group.release is release
    assert release.orientation_group is group
    assert group.retreat == [actions[-1]]
    assert poses[2].orientation_group.release is None

    copied = deepcopy(tuple(actions))
    assert copied[0].orientation_group is copied[3].orientation_group
    assert copied[0].orientation_group is not group
    assert copied[0].orientation_group.members[1] is copied[1]
    assert list(copied) == actions


# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------


class _ClearanceBackend(MockSceneBackend):
    """A mock backend whose clearance check rejects the first paths."""

    def __init__(self, *args: object, blocked: int = 0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.blocked = blocked
        self.calls: list[dict[str, object]] = []

    def held_object_clearance(
        self,
        operator_name,
        object_name,
        eef_paths,
        bodies,
        margin,
        *,
        open_paths=None,
        open_commands=None,
        env_index=0,
    ) -> ClearanceReport:
        first = sum(len(call["eef_paths"]) for call in self.calls)
        self.calls.append(
            {
                "object_name": object_name,
                "eef_paths": list(eef_paths),
                "open_paths": None if open_paths is None else list(open_paths),
                "open_commands": None if open_commands is None else list(open_commands),
                "bodies": tuple(bodies),
            }
        )
        for index in range(len(eef_paths)):
            if first + index >= self.blocked:
                return ClearanceReport(clear=True, distance=0.02, path_index=index)
        return ClearanceReport(
            clear=False, distance=0.0, probe_geom="finger", obstacle_geom="wall"
        )


def _runtime_setup(goal: dict[str, object], *, blocked: int = 0, jaw_roll: float = 0.0):
    # The EEF, pitched 10 deg and rolled ``jaw_roll`` about its approach axis,
    # holds the object 5 cm ahead, rolled 40 deg about the object's long
    # axis, so the object starts off level.
    eef_pose = PoseState(
        position=(0.1, -0.4, 0.3),
        orientation=_compose(
            _quaternion((0, 1, 0), 0.17), _quaternion((1, 0, 0), jaw_roll)
        ),
    )
    eef_from_object = PoseState(
        position=(0.05, 0.0, 0.0), orientation=_quaternion((1, 0, 0), 0.7)
    )
    operator = MockOperatorHandler(
        operator_name="arm", batch_size=1, end_effector_pose=eef_pose
    )
    backend = _ClearanceBackend(
        env_name="nearest_feasible",
        batch_size=1,
        operators={"arm": operator},
        objects={
            "tuber": MockObjectHandler(
                name="tuber", pose=compose_pose(eef_pose, eef_from_object)
            ),
            "target": MockObjectHandler(
                name="target",
                pose=PoseState(
                    position=(0.0, 0.2, 0.2), orientation=_quaternion((0, 0, 1), 0.3)
                ),
            ),
        },
        blocked=blocked,
    )
    backend.get_grasped_object_name = lambda operator_name, env_index: "tuber"
    binding = GraspBinding(
        env_index=0,
        operator_name="arm",
        object_name="tuber",
        eef_from_object=eef_from_object,
    )
    actions, _ = TaskFlowBuilder.build_actions(_place_stage(goal))
    return backend, operator, binding, actions


def _resolve(backend, operator, binding, action: PrimitiveAction):
    return TaskRunner._resolve_motion_goal(
        env_index=0,
        operator=operator,
        pose=action.pose,
        target=backend.get_object_handler("target"),
        backend=backend,
        action=action,
        grasp_binding=binding,
    )


def test_the_first_member_solves_and_later_members_reuse_it() -> None:
    backend, operator, binding, actions = _runtime_setup(_goal())

    first = _resolve(backend, operator, binding, actions[0])
    target = backend.get_object_handler("target").get_pose().select(0)
    in_target = quaternion_to_rotation_matrix(
        target.orientation[0]
    ).T @ quaternion_to_rotation_matrix(first.controlled_world_pose.orientation[0])
    assert _elevation(in_target, (1, 0, 0)) == pytest.approx(0.0, abs=1e-6)

    # The arm turned since; the second member still takes the solved goal.
    operator.end_effector_pose = PoseState(
        position=(0.3, 0.0, 0.3), orientation=_quaternion((0, 0, 1), 1.0)
    )
    second = _resolve(backend, operator, binding, actions[1])
    assert quaternion_angular_distance(
        first.controlled_world_pose.orientation[0],
        second.controlled_world_pose.orientation[0],
    ) == pytest.approx(0.0, abs=1e-9)
    assert actions[0].orientation_group.details["rotation_rad"] > 0.0


def test_clearance_walks_candidates_in_order_and_covers_the_retreat() -> None:
    goal = _goal(clearance={"bodies": ["microwave"], "step": 0.05})
    backend, operator, binding, actions = _runtime_setup(goal, blocked=20)

    _resolve(backend, operator, binding, actions[0])

    group = actions[0].orientation_group
    assert group.details["clearance_checked"] == 21
    call = backend.calls[0]
    assert call["object_name"] == "tuber" and call["bodies"] == ("microwave",)
    # Path: three members 30 cm apart at 5 cm steps -> 1 + 6 + 1 samples.
    assert call["eef_paths"][0].batch_size == 8
    # Opened gripper: the release pose, then 25 cm back along world -Y.
    opened = call["open_paths"][0]
    np.testing.assert_allclose(opened.position[0], call["eef_paths"][0].position[-1])
    np.testing.assert_allclose(
        opened.position[-1] - opened.position[0], [0.0, -0.25, 0.0], atol=1e-12
    )


def _two_posture_goal(**overrides: object) -> dict[str, object]:
    return _goal(
        constraints=[],
        postures=[
            {
                "name": "side_by_side",
                "constraints": [
                    {"frame": "eef", "axis": [0, 1, 0], "elevation": [[-0.524, 0.524]]}
                ],
            },
            {
                "name": "one_above_other",
                "constraints": [
                    {
                        "frame": "eef",
                        "axis": [0, 1, 0],
                        "elevation": [[1.047, 1.5708], [-1.5708, -1.047]],
                    }
                ],
                "offset": [0.0, 0.0, 0.03],
                "release_joint_positions": [0.008],
            },
        ],
        **overrides,
    )


def test_a_chosen_posture_lifts_every_member_and_sets_the_release_opening() -> None:
    # The jaws are rolled 70 deg from level: upright is the nearer posture.
    backend, operator, binding, actions = _runtime_setup(
        _two_posture_goal(), jaw_roll=1.22
    )
    target = backend.get_object_handler("target").get_pose().select(0)
    lift = quaternion_to_rotation_matrix(target.orientation[0]) @ np.array([0, 0, 0.03])

    first = _resolve(backend, operator, binding, actions[0])
    later = _resolve(backend, operator, binding, actions[3])

    group = actions[0].orientation_group
    assert group.details["posture"] == "one_above_other"
    assert group.release_joint_positions == (0.008,)
    for goal, local in ((first, (-0.30, 0.0, 0.02)), (later, (0.0, 0.0, 0.0))):
        nominal = compose_pose(target, PoseState(position=local)).position[0]
        np.testing.assert_allclose(
            goal.controlled_world_pose.position[0], nominal + lift, atol=1e-9
        )

    # The release command opens only as far as the posture says.
    commanded = []
    operator.control_eef = lambda eef, target, env_mask=None: (
        commanded.append(eef)
        or MockOperatorHandler.control_eef(operator, eef, target, env_mask)
    )
    TaskRunner._run_action(
        env_index=0,
        operator=operator,
        action=actions[4],
        target=backend.get_object_handler("target"),
        backend=backend,
        env_mask=np.asarray([True]),
    )
    assert commanded[0].close is False
    assert commanded[0].joint_positions == [0.008]


def test_clearance_measures_each_candidate_at_its_posture() -> None:
    goal = _two_posture_goal(clearance={"bodies": ["microwave"]})
    backend, operator, binding, actions = _runtime_setup(goal, blocked=10**9)

    with pytest.raises(ValueError, match="none of"):
        _resolve(backend, operator, binding, actions[0])

    commands = [c for call in backend.calls for c in call["open_commands"]]
    paths = [p for call in backend.calls for p in call["eef_paths"]]
    assert {tuple(command.joint_positions) for command in commands} == {
        (),
        (0.008,),
    }
    # The object rides 5 cm ahead of the EEF; its last sample is the release
    # target, 3 cm higher (along the target's z) for upright candidates.
    target = backend.get_object_handler("target").get_pose().select(0)
    release = compose_pose(target, PoseState(position=(0.0, 0.0, 0.0))).position[0]
    lift = quaternion_to_rotation_matrix(target.orientation[0]) @ np.array(
        [0.0, 0.0, 0.03]
    )
    for command, path in zip(commands, paths, strict=True):
        held = compose_pose(
            PoseState(position=path.position[-1], orientation=path.orientation[-1]),
            PoseState(position=(0.05, 0.0, 0.0)),
        ).position[0]
        expected = release + (lift if command.joint_positions == [0.008] else 0.0)
        np.testing.assert_allclose(held, expected, atol=1e-9)


def test_no_clear_candidate_fails_with_the_blocking_geoms() -> None:
    goal = _goal(clearance={"bodies": ["microwave"]})
    backend, operator, binding, actions = _runtime_setup(goal, blocked=10**9)

    with pytest.raises(ValueError, match="'finger' against 'wall'"):
        _resolve(backend, operator, binding, actions[0])
    assert actions[0].orientation_group.solution is None


def test_a_backend_without_clearance_support_refuses() -> None:
    goal = _goal(clearance={"bodies": ["microwave"]})
    backend, operator, binding, actions = _runtime_setup(goal)
    backend.held_object_clearance = MockSceneBackend.held_object_clearance.__get__(
        backend
    )

    with pytest.raises(NotImplementedError, match="held-object clearance"):
        _resolve(backend, operator, binding, actions[0])


# ----------------------------------------------------------------------
# MuJoCo clearance
# ----------------------------------------------------------------------

_SCENE = """
<mujoco>
  <worldbody>
    <body name="wall" pos="0.5 0 0">
      <geom name="wall_geom" type="box" size="0.01 0.5 0.5"/>
      <geom name="wall_marker" type="sphere" size="0.01" pos="0 0 0.6"
            contype="0" conaffinity="0"/>
    </body>
    <body name="tool" pos="0 0 0">
      <freejoint/>
      <geom name="tool_geom" type="box" size="0.05 0.02 0.02"/>
      <body name="finger" pos="0.05 0 0">
        <joint name="finger_slide" type="slide" axis="1 0 0"/>
        <geom name="finger_geom" type="box" size="0.02 0.01 0.01" pos="0.02 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_mujoco_clearance_finds_the_first_clear_path_without_touching_state() -> None:
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wall")
    probes = probe_geoms(model, subtree_bodies(model, tool))
    obstacles = obstacle_geoms(model, subtree_bodies(model, wall))
    assert len(probes) == 2 and len(obstacles) == 1
    held = capture_template(data, probes, data.xpos[tool], np.eye(3))
    # Extended finger: 4 cm further out, as an opened gripper would be.
    data.qpos[model.jnt_qposadr[1]] = 0.04
    mujoco.mj_kinematics(model, data)
    opened = capture_template(data, probes, data.xpos[tool], np.eye(3))
    qpos_before = data.qpos.copy()
    xpos_before = data.geom_xpos.copy()

    def path(x: float) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]), np.stack([np.eye(3)] * 2)

    # The closed tool's tip reaches x + 0.09; the wall face is at 0.49.
    report = check_paths(
        model, data, [held], [], [path(0.45), path(0.39)], None, obstacles, 0.005
    )
    assert report.clear and report.path_index == 1 and report.checked_paths == 2
    assert report.distance == pytest.approx(0.01, abs=1e-6)

    # Opened, the same path comes 4 cm closer and is blocked.
    report = check_paths(
        model, data, [held], [[opened]], [path(0.39)], [path(0.39)], obstacles, 0.005
    )
    assert not report.clear and report.released
    assert (report.probe_geom, report.obstacle_geom) == ("finger_geom", "wall_geom")

    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_array_equal(data.geom_xpos, xpos_before)
