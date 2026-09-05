"""Headless end-to-end regression for the P7 rack-plate task."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir

from auto_atom.framework import PoseReference
from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner


_ROOT = Path(__file__).resolve().parents[1]
_MAX_UPDATES = 600
_SETTLE_SECONDS = 0.5
_RACK_RIB_PREFIX = "rack_rib_"
_STAND_GUIDES = {
    "plate_stand_post_front_left",
    "plate_stand_post_front_right",
    "plate_stand_post_back_left",
    "plate_stand_post_back_right",
}


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type.name}: {name}"
    return int(object_id)


def _descendant_bodies(model: mujoco.MjModel, root_body: int) -> set[int]:
    bodies = {root_body}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if int(model.body_parentid[body_id]) in bodies and body_id not in bodies:
                bodies.add(body_id)
                changed = True
    return bodies


def test_rack_plate_p7_v4_umi_v3_completes_headless() -> None:
    ComponentRegistry.clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_ROOT / "aao_configs"),
    ):
        config = compose(
            config_name="rack_plate_p7_v4_umi_v3",
            overrides=["env.viewer=null"],
        )

    runner = TaskRunner().from_config(prepare_task_file(config))
    try:
        backend = runner._context.backend
        single_env = backend.get_env().envs[0]  # type: ignore[union-attr]
        model = single_env.model
        data = single_env.data
        geom_names = {
            geom_id: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(model.ngeom)
        }
        object_collision = _id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "object_collision",
        )
        rack_ribs = {
            geom_id
            for geom_id, name in geom_names.items()
            if name.startswith(_RACK_RIB_PREFIX)
        }
        stand_guides = {
            geom_id for geom_id, name in geom_names.items() if name in _STAND_GUIDES
        }
        operator_root = _id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "p7_mount",
        )
        operator_bodies = _descendant_bodies(model, operator_root)
        robot_geoms = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in operator_bodies
            and int(model.geom_contype[geom_id]) != 0
        }
        left_finger = _id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "eef_left_finger_collision",
        )
        right_finger = _id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "eef_right_finger_collision",
        )

        left_contact_seen = False
        right_contact_seen = False
        carrying_started = False
        placement_released = False
        initial_support_contacts: list[tuple[str, float]] = []
        forbidden_object_contacts: list[tuple[str, float]] = []
        forbidden_robot_contacts: list[tuple[str, str, float]] = []
        minimum_distances: defaultdict[str, float] = defaultdict(lambda: float("inf"))

        def audit_contacts(
            _model: mujoco.MjModel,
            current_data: mujoco.MjData,
        ) -> None:
            nonlocal left_contact_seen, right_contact_seen, carrying_started
            nonlocal placement_released
            if not carrying_started:
                carrying_started = bool(backend.is_object_grasped("arm", "object")[0])
            elif not placement_released:
                placement_released = not bool(backend.is_operator_grasping("arm")[0])
            for contact_index in range(current_data.ncon):
                contact = current_data.contact[contact_index]
                geom_a = int(contact.geom1)
                geom_b = int(contact.geom2)
                distance = float(contact.dist)
                pair = {geom_a, geom_b}
                if object_collision in pair:
                    other = geom_b if geom_a == object_collision else geom_a
                    other_name = geom_names[other]
                    if other == left_finger:
                        left_contact_seen = True
                    if other == right_finger:
                        right_contact_seen = True
                    if other in rack_ribs or other in stand_guides:
                        if other in stand_guides:
                            if not carrying_started:
                                initial_support_contacts.append((other_name, distance))
                            # Contact with the four pickup posts is an
                            # intentional support/grasp interaction.  The
                            # forbidden-object audit starts at the rack ribs.
                            continue
                        if (
                            other in rack_ribs
                            and carrying_started
                            and not placement_released
                        ):
                            forbidden_object_contacts.append((other_name, distance))
                            minimum_distances[other_name] = min(
                                minimum_distances[other_name], distance
                            )
                if geom_a in robot_geoms and geom_b in rack_ribs | stand_guides:
                    forbidden_robot_contacts.append(
                        (geom_names[geom_a], geom_names[geom_b], distance)
                    )
                elif geom_b in robot_geoms and geom_a in rack_ribs | stand_guides:
                    forbidden_robot_contacts.append(
                        (geom_names[geom_b], geom_names[geom_a], distance)
                    )

        single_env._pre_step_callbacks.append(audit_contacts)

        update = runner.reset()
        updates_used = 0
        while not bool(np.all(update.done)) and updates_used < _MAX_UPDATES:
            update = runner.update()
            updates_used += 1

        assert bool(np.all(update.done)), (
            f"rack_plate task did not finish in {_MAX_UPDATES} updates: "
            f"stage={update.stage_name}, phase={update.phase}, "
            f"details={update.details}"
        )
        assert update.success.tolist() == [True], (
            f"rack_plate task reached a terminal failure: details={update.details}, "
            f"records={runner.records}"
        )
        assert initial_support_contacts
        assert [record.stage_name for record in runner.records] == [
            "pick_plate",
            "place_plate",
        ]
        assert [record.status.value for record in runner.records] == [
            "succeeded",
            "succeeded",
        ]
        assert left_contact_seen and right_contact_seen

        plate_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        target_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "rack_target_site")
        plate_geom = object_collision
        completion_position = data.xpos[plate_body].copy()
        completion_error = float(
            np.linalg.norm(completion_position - data.site_xpos[target_site])
        )
        assert completion_error <= 0.025

        settle_steps = round(_SETTLE_SECONDS / model.opt.timestep)
        for _ in range(settle_steps):
            audit_contacts(model, data)
            mujoco.mj_step(model, data)
        audit_contacts(model, data)

        assert forbidden_object_contacts == [], minimum_distances
        assert forbidden_robot_contacts == []
        settled_position = data.xpos[plate_body].copy()
        settled_error = float(
            np.linalg.norm(settled_position - data.site_xpos[target_site])
        )
        assert settled_error <= 0.025
        # With the source mesh collision, the released plate can seat against
        # a rack rib by a few millimetres while remaining inside the 25 mm
        # placement tolerance. This is expected physical settling, not a
        # transport collision.
        assert float(np.linalg.norm(settled_position - completion_position)) <= 0.01

        # MuJoCo's compiled mesh frame for plate.obj puts its physical normal
        # on the first column of geom_xmat (the source mesh carries its own
        # authored axis transform).
        plate_normal = data.geom_xmat[plate_geom].reshape(3, 3)[:, 0]
        assert float(np.dot(plate_normal, np.array([1.0, 0.0, 0.0]))) >= np.cos(0.15)

        pick_waypoints = config.task.stages[0].param.pre_move
        assert len(pick_waypoints) == 2
        assert list(pick_waypoints[0].position) == [0.0, -0.025, 0.17]
        assert list(pick_waypoints[1].position) == [0.0, -0.025, 0.0]
        for waypoint in pick_waypoints:
            assert waypoint.reference == PoseReference.OBJECT
            orientation_goal = waypoint.orientation_goal
            assert orientation_goal.kind == "axis_alignment"
            assert orientation_goal.target_axis.reference == "object"
            assert orientation_goal.direction == "opposite"
    finally:
        runner.close()
        ComponentRegistry.clear()
