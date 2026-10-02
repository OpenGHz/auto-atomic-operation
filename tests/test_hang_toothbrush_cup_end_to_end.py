"""Headless end-to-end regression for hang_toothbrush_cup on the Robotiq weld."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from auto_atom.config_loader import compose_task_config
from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner

_ROOT = Path(__file__).resolve().parents[1]
_MAX_UPDATES = 600


def test_hang_toothbrush_cup_completes_on_gs_scene_layout() -> None:
    """Carry the cup by its handle within the default reach tolerance.

    The grip on the handle tilts the welded Robotiq body; with a soft mocap
    weld the eef_pose site, 0.149 m away, stays outside the 1 cm / 0.08 rad
    tolerance and the lift times out. ``model_name=demo_gs`` selects the GS
    scene layout (the variant the task waypoints are tuned for) while keeping
    native rendering, so the test needs neither CUDA nor the GS assets.
    """

    ComponentRegistry.clear()
    config = compose_task_config(
        "hang_toothbrush_cup",
        ["model_name=demo_gs", "env.batch_size=4", "++env.viewer=null", "~env.cameras"],
        config_dir=_ROOT / "aao_configs",
    )
    runner = TaskRunner().from_config(prepare_task_file(config))
    try:
        update = runner.reset()
        for _ in range(_MAX_UPDATES):
            if bool(np.all(update.done)):
                break
            update = runner.update()

        failures = [
            (record.env_index, record.stage_name, record.details.get("event"))
            for record in runner.records
            if record.status.value != "succeeded"
        ]
        assert not failures
        assert bool(np.all(update.success))
    finally:
        runner.close()
        ComponentRegistry.clear()
