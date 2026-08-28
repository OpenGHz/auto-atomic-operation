"""Headless end-to-end regression for the dishwasher plate task."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir

from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner


_ROOT = Path(__file__).resolve().parents[1]
_MAX_UPDATES = 1_000
_MAX_PAD_PENETRATION_M = 1.5e-3
_POST_RELEASE_SETTLE_SECONDS = 1.0


def test_dishwasher_plate_completes_headless() -> None:
    """Run the real MuJoCo task and require both pick and place to succeed."""

    ComponentRegistry.clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_ROOT / "aao_configs"),
    ):
        config = compose(
            config_name="dishwasher_plate",
            overrides=["env.batch_size=1", "env.viewer=null"],
        )

    runner = TaskRunner().from_config(prepare_task_file(config))
    try:
        single_env = runner._context.backend.get_env().envs[0]  # type: ignore[union-attr]
        model = single_env.model
        data = single_env.data
        geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(model.ngeom)
        ]
        finger_pads = {
            geom_id
            for geom_id, name in enumerate(geom_names)
            if name.startswith("eef_") and "finger_pad_" in name
        }
        plate_collision = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "plate2_collision",
        )
        dishwasher_collision = {
            geom_id
            for geom_id, name in enumerate(geom_names)
            if name.startswith("dishwasher_") and int(model.geom_contype[geom_id]) != 0
        }
        allowed_plate_supports = {
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in (
                "dishwasher_rack1_wire_c038_s000_001_contact",
                "dishwasher_rack1_wire_c050_s000_001_contact",
            )
        }
        assert plate_collision >= 0
        assert len(finger_pads) == 4
        assert len(dishwasher_collision) == 282
        assert len(allowed_plate_supports) == 2
        assert allowed_plate_supports <= dishwasher_collision

        physics_audit_steps = 0
        maximum_pad_penetration = 0.0
        forbidden_plate_contacts: list[tuple[str, float]] = []
        forbidden_pad_contacts: list[tuple[str, str, float]] = []

        def audit_physical_contacts(
            _model: mujoco.MjModel,
            current_data: mujoco.MjData,
        ) -> None:
            nonlocal physics_audit_steps, maximum_pad_penetration
            physics_audit_steps += 1
            for contact in current_data.contact:
                geom_a = int(contact.geom1)
                geom_b = int(contact.geom2)
                distance = float(contact.dist)
                pair = {geom_a, geom_b}

                if plate_collision in pair:
                    other = geom_b if geom_a == plate_collision else geom_a
                    if other in finger_pads:
                        maximum_pad_penetration = max(
                            maximum_pad_penetration,
                            max(0.0, -distance),
                        )
                    elif (
                        other in dishwasher_collision
                        and other not in allowed_plate_supports
                    ):
                        forbidden_plate_contacts.append((geom_names[other], distance))

                pad = next((geom for geom in pair if geom in finger_pads), None)
                obstacle = next(
                    (geom for geom in pair if geom in dishwasher_collision),
                    None,
                )
                if pad is not None and obstacle is not None:
                    forbidden_pad_contacts.append(
                        (geom_names[pad], geom_names[obstacle], distance)
                    )

        # This hook is invoked immediately before every internal mj_step, so
        # the 1200 Hz collision audit does not miss contacts hidden inside a
        # 30 Hz TaskRunner update.
        single_env._pre_step_callbacks.append(audit_physical_contacts)

        update = runner.reset()
        updates_used = 0
        while not bool(np.all(update.done)) and updates_used < _MAX_UPDATES:
            update = runner.update()
            updates_used += 1

        assert bool(np.all(update.done)), (
            f"dishwasher_plate did not finish in {_MAX_UPDATES} updates: "
            f"stage={update.stage_name}, phase={update.phase}, "
            f"details={update.details}"
        )
        assert update.success.tolist() == [True], (
            "dishwasher_plate reached a terminal failure: "
            f"details={update.details}, records={runner.records}"
        )

        assert [record.stage_name for record in runner.records] == [
            "pick_plate",
            "place_plate_in_upper_rack",
        ]
        assert [record.status.value for record in runner.records] == [
            "succeeded",
            "succeeded",
        ]
        assert runner.records[0].details["event"] == "eef_grasped"
        assert runner.records[0].details["grasped_object"] == "plate2"
        assert runner.records[0].details["target_ctrl"] == 0.019
        assert runner.records[1].details["event"] == "eef_reached"
        assert runner.records[1].details["grasped_object"] == ""
        assert runner.records[1].details["steps"] == 30
        assert physics_audit_steps >= updates_used * single_env._n_substeps

        plate_body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "plate2",
        )
        target_site = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "dishwasher_rack1_target_site",
        )
        completion_position = data.xpos[plate_body].copy()
        completion_error = float(
            np.linalg.norm(completion_position - data.site_xpos[target_site])
        )
        assert completion_error <= 0.025

        extra_steps = round(_POST_RELEASE_SETTLE_SECONDS / model.opt.timestep)
        for _ in range(extra_steps):
            audit_physical_contacts(model, data)
            mujoco.mj_step(model, data)
        # Audit the state produced by the final mj_step as well; the callback
        # and loop above otherwise observe only each step's input state.
        audit_physical_contacts(model, data)

        assert maximum_pad_penetration <= _MAX_PAD_PENETRATION_M
        assert forbidden_plate_contacts == []
        assert forbidden_pad_contacts == []

        settled_position = data.xpos[plate_body].copy()
        settled_error = float(
            np.linalg.norm(settled_position - data.site_xpos[target_site])
        )
        assert settled_error <= 0.025
        assert float(np.linalg.norm(settled_position - completion_position)) <= 0.005

        plate_axis_world = data.xmat[plate_body].reshape(3, 3)[:, 2]
        target_axis_world = data.site_xmat[target_site].reshape(3, 3)[:, 1]
        axis_error = float(
            np.arccos(
                np.clip(
                    np.dot(plate_axis_world, target_axis_world),
                    -1.0,
                    1.0,
                )
            )
        )
        assert axis_error <= 0.15
    finally:
        runner.close()
        ComponentRegistry.clear()
