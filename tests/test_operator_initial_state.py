from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.runtime import ComponentRegistry, load_task_file_hydra
from auto_atom.utils.pose import quaternion_angular_distance


def _build_press_button_backend(overrides: list[str] | None = None):
    ComponentRegistry.clear()
    task_file = load_task_file_hydra(
        "press_blue_button",
        config_dir=ROOT / "aao_configs",
        overrides=["env.batch_size=1", "env.viewer=null", *(overrides or [])],
    )
    backend = task_file.backend(task_file.task, task_file.task_operators)
    backend.setup(task_file.task)
    return backend


def _eef_pose_and_finger_distance(backend):
    handler = backend.get_operator_handler("arm")
    pose = handler.get_end_effector_pose().select(0)
    env = backend.env.envs[0]
    left_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    right_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad"
    )
    finger_distance = float(
        np.linalg.norm(env.data.geom_xpos[left_id] - env.data.geom_xpos[right_id])
    )
    return pose, finger_distance


def _normalized(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def test_closed_initial_eef_keeps_press_button_home_pose() -> None:
    open_backend = _build_press_button_backend(
        overrides=["task_operators.0.initial_state=null"]
    )
    try:
        open_pose, open_distance = _eef_pose_and_finger_distance(open_backend)
    finally:
        open_backend.teardown()
        ComponentRegistry.clear()

    closed_backend = _build_press_button_backend()
    try:
        closed_pose, closed_distance = _eef_pose_and_finger_distance(closed_backend)
    finally:
        closed_backend.teardown()
        ComponentRegistry.clear()

    pos_error = float(np.linalg.norm(closed_pose.position[0] - open_pose.position[0]))
    ori_error = float(
        quaternion_angular_distance(
            _normalized(closed_pose.orientation[0]),
            _normalized(open_pose.orientation[0]),
        )
    )

    assert pos_error < 5e-5, f"home EEF position drifted by {pos_error:.6e} m"
    assert ori_error < 5e-5, f"home EEF orientation drifted by {ori_error:.6e} rad"
    assert open_distance > 0.08, (
        f"expected open reference gripper to stay open, got {open_distance:.6f} m"
    )
    assert closed_distance < 0.03, (
        f"expected closed initial gripper to settle closed, got {closed_distance:.6f} m"
    )
