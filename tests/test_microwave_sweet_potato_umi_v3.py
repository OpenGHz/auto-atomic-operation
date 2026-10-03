"""Headless end-to-end regression for the UMI v3 sweet-potato-in-microwave task."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from auto_atom.config_loader import compose_task_config
from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner


_ROOT = Path(__file__).resolve().parents[1]
_MAX_UPDATES = 600
_SETTLE_SECONDS = 1.0
# Render-only gripper meshes are checked every N physics steps; contacts are
# checked every step.
_CLEARANCE_STRIDE = 12


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


def test_microwave_sweet_potato_umi_v3_completes_headless() -> None:
    ComponentRegistry.clear()
    config = compose_task_config(
        "microwave_sweet_potato",
        ["env.viewer=null"],
        config_dir=_ROOT / "aao_configs",
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
        microwave_bodies = {
            _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave"),
            _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave_door"),
        }
        microwave_geoms = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in microwave_bodies
            and int(model.geom_contype[geom_id]) != 0
        }
        potato = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "sweet_potato_collision")
        fingers = {
            _id(model, mujoco.mjtObj.mjOBJ_GEOM, "eef_left_finger_collision"),
            _id(model, mujoco.mjtObj.mjOBJ_GEOM, "eef_right_finger_collision"),
        }
        gripper_bodies = _descendant_bodies(
            model, _id(model, mujoco.mjtObj.mjOBJ_BODY, "umi_interface")
        )
        # Includes the non-colliding render meshes: the gripper must not clip
        # visibly through the cavity even where physics would not notice.
        gripper_meshes = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in gripper_bodies
            and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        ]

        finger_contacts: set[str] = set()
        carrying_started = False
        placement_released = False
        carried_contacts: list[tuple[str, float]] = []
        microwave_finger_contacts: list[tuple[str, str]] = []
        min_clearance = float("inf")
        step_count = 0

        def audit(_model: mujoco.MjModel, current_data: mujoco.MjData) -> None:
            nonlocal carrying_started, placement_released, min_clearance, step_count
            step_count += 1
            if not carrying_started:
                carrying_started = bool(
                    backend.is_object_grasped("arm", "sweet_potato")[0]
                )
            elif not placement_released:
                placement_released = not bool(backend.is_operator_grasping("arm")[0])
            for contact_index in range(current_data.ncon):
                contact = current_data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if potato in pair:
                    other = (pair - {potato}).pop()
                    if other in fingers:
                        finger_contacts.add(geom_names[other])
                    if (
                        other in microwave_geoms
                        and carrying_started
                        and not placement_released
                    ):
                        carried_contacts.append(
                            (geom_names[other], float(contact.dist))
                        )
                if pair & fingers and pair & microwave_geoms:
                    microwave_finger_contacts.append(
                        (
                            geom_names[(pair & fingers).pop()],
                            geom_names[(pair & microwave_geoms).pop()],
                        )
                    )
            if step_count % _CLEARANCE_STRIDE == 0:
                for gripper_geom in gripper_meshes:
                    for microwave_geom in microwave_geoms:
                        min_clearance = min(
                            min_clearance,
                            mujoco.mj_geomDistance(
                                _model,
                                current_data,
                                gripper_geom,
                                microwave_geom,
                                0.02,
                                None,
                            ),
                        )

        single_env._pre_step_callbacks.append(audit)

        update = runner.reset()
        updates_used = 0
        while not bool(np.all(update.done)) and updates_used < _MAX_UPDATES:
            update = runner.update()
            updates_used += 1

        assert bool(np.all(update.done)), (
            f"microwave task did not finish in {_MAX_UPDATES} updates: "
            f"stage={update.stage_name}, phase={update.phase}, "
            f"details={update.details}"
        )
        assert update.success.tolist() == [True], (
            f"microwave task reached a terminal failure: details={update.details}, "
            f"records={runner.records}"
        )
        assert [record.stage_name for record in runner.records] == [
            "pick_sweet_potato",
            "place_sweet_potato_in_microwave",
        ]
        assert [record.status.value for record in runner.records] == [
            "succeeded",
            "succeeded",
        ]
        assert finger_contacts == {
            "eef_left_finger_collision",
            "eef_right_finger_collision",
        }
        assert carrying_started and placement_released

        potato_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "sweet_potato")
        target_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "microwave_target_site")
        completion_error = float(
            np.linalg.norm(data.xpos[potato_body] - data.site_xpos[target_site])
        )
        assert completion_error <= 0.025

        for _ in range(round(_SETTLE_SECONDS / model.opt.timestep)):
            audit(model, data)
            mujoco.mj_step(model, data)

        assert carried_contacts == []
        assert microwave_finger_contacts == []
        assert min_clearance > 0.0
        settled_error = float(
            np.linalg.norm(data.xpos[potato_body] - data.site_xpos[target_site])
        )
        assert settled_error <= 0.025
        # The tuber still points into the cavity after settling.
        long_axis = data.xmat[potato_body].reshape(3, 3)[:, 0]
        target_axis = data.site_xmat[target_site].reshape(3, 3)[:, 0]
        assert float(long_axis @ target_axis) > np.cos(0.2)
    finally:
        runner.close()


@pytest.mark.parametrize("preset", ["gravity", "zero_gravity"])
def test_randomization_presets_complete_headless(preset: str) -> None:
    """Each preset samples a wide scene and still finishes the task."""
    ComponentRegistry.clear()
    config = compose_task_config(
        "microwave_sweet_potato",
        [
            "env.viewer=null",
            "task.seed=3",
            f"randomization=microwave_sweet_potato/{preset}",
        ],
        config_dir=_ROOT / "aao_configs",
    )
    zero_gravity = preset == "zero_gravity"

    runner = TaskRunner().from_config(prepare_task_file(config))
    try:
        backend = runner._context.backend
        single_env = backend.get_env().envs[0]  # type: ignore[union-attr]
        model, data = single_env.model, single_env.data
        # One scene camera in front plus the wrist camera, which rides the arm.
        assert set(backend.camera_names()) == {"env1_cam", "eef_wrist_cam"}
        assert backend.operator_camera_names() == {"eef_wrist_cam"}
        np.testing.assert_allclose(
            model.opt.gravity, [0.0, 0.0, 0.0 if zero_gravity else -9.81]
        )
        microwave = _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave")
        potato = _id(model, mujoco.mjtObj.mjOBJ_BODY, "sweet_potato")
        target = _id(model, mujoco.mjtObj.mjOBJ_SITE, "microwave_target_site")
        eef = _id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_pose")
        front = _id(model, mujoco.mjtObj.mjOBJ_CAMERA, "env1_cam")
        front_spec = single_env._camera_specs["env1_cam"]
        half_fovy = np.tan(np.radians(float(model.cam_fovy[front])) / 2.0)
        door = int(
            model.jnt_qposadr[
                _id(model, mujoco.mjtObj.mjOBJ_JOINT, "microwave_door_hinge")
            ]
        )
        microwave_starts = []
        door_angles = []
        for _ in range(2):
            update = runner.reset()
            microwave_starts.append(data.xpos[microwave].copy())
            # The door opens between square to the front and its stop.
            door_angles.append(float(data.qpos[door]))
            assert -2.0943 <= door_angles[-1] <= -1.0123
            # Without gravity the tuber floats well clear of the counter top
            # (z = 0.06); with it, the tuber rests there in its settled pose.
            potato_height = float(data.xpos[potato][2])
            if zero_gravity:
                assert potato_height > 0.15
                # Placed by the front camera's view rather than the counter:
                # the tuber's centre projects inside the image.
                camera_point = data.cam_xmat[front].reshape(3, 3).T @ (
                    data.xpos[potato] - data.cam_xpos[front]
                )
                depth = -float(camera_point[2])
                assert depth > 0.0
                assert abs(camera_point[1]) / depth < half_fovy
                assert abs(camera_point[0]) / depth < (
                    half_fovy * front_spec.width / front_spec.height
                )
            else:
                assert potato_height == pytest.approx(0.0821, abs=1e-3)
            # The home pose follows the tuber, so the gripper starts above it
            # however far the tuber moved, and upright.
            assert float(data.site_xpos[eef][2]) > potato_height + 0.05
            assert float(data.site_xmat[eef].reshape(3, 3)[2, 2]) > 0.5
            updates_used = 0
            while not bool(np.all(update.done)) and updates_used < _MAX_UPDATES:
                update = runner.update()
                updates_used += 1
            assert update.success.tolist() == [True], update.details
            assert (
                float(np.linalg.norm(data.xpos[potato] - data.site_xpos[target]))
                < 0.025
            )
        # The microwave itself is randomized on the counter, door included.
        assert float(np.linalg.norm(microwave_starts[0] - microwave_starts[1])) > 1e-3
        assert abs(door_angles[0] - door_angles[1]) > 1e-3
    finally:
        runner.close()
