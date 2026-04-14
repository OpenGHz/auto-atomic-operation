"""Verify that gripper observation and control values are physically consistent.

For each robot/gripper combination, the test checks:

1. **Monotonic tracking**: as the commanded control value increases, the
   observed joint position (qpos) also increases monotonically — proving that
   observation and control are in the same physical space.

2. **Monotonic finger-distance relationship**: the raw joint value maps
   monotonically to the actual Euclidean distance between the left and right
   finger pads, so the observation/control space faithfully tracks real finger
   opening.

3. **Framework round-trip**: values written through ``apply_joint_action``
   appear consistently in ``capture_observation`` — both the measurement
   (qpos) and the action (ctrl) channels.

Notes
-----
*  Grippers with parallel-linkage mechanisms (xf9600, robotiq) have equality
   constraints that create internal forces, so qpos will NOT converge exactly
   to ctrl. The tests therefore check *monotonic tracking* and *bounded error*
   rather than exact equality.
*  The xf9600 actuator (kp=5000) requires timestep ≤ 0.001 s for stability;
   the test overrides ``model.opt.timestep`` to 0.001 before stepping.
*  Gravity is disabled so the robot doesn't fall; only the gripper actuator
   is exercised.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "xmls"

# ── Per-robot test parameters ────────────────────────────────────────────────
_ROBOT_CONFIGS: list[dict] = [
    {
        "id": "xf9600_airbot_play",
        "scene_xml": _ASSETS / "scenes" / "open_door" / "demo_airbot_play.xml",
        "actuator": "xfg_claw_joint",
        "left_pad": "xfg_left_finger_pad_upper",
        "right_pad": "xfg_right_finger_pad_upper",
        "ctrl_range": (0.0, 0.02),
        # ctrl 0 → open, ctrl 0.02 → closed
        "open_is_min": True,
    },
    {
        "id": "robotiq_pick_and_place",
        "scene_xml": _ASSETS / "scenes" / "pick_and_place" / "demo.xml",
        "actuator": "fingers_actuator",
        "left_pad": "left_finger_pad",
        "right_pad": "right_finger_pad",
        "ctrl_range": (0.0, 0.82),
        # ctrl 0 → open, ctrl 0.82 → closed
        "open_is_min": True,
    },
]

_IDS = [cfg["id"] for cfg in _ROBOT_CONFIGS]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_model(cfg: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load model, disable gravity, set a safe timestep, and reset."""
    scene = cfg["scene_xml"]
    if not scene.exists():
        pytest.skip(f"Scene XML not found: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.001  # xf9600 kp=5000 needs ≤0.001 for stability
    model.opt.gravity[:] = 0  # prevent robot from falling
    data = mujoco.MjData(model)
    return model, data


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


def _hold_all_joints(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set every actuator ctrl to its current joint qpos so joints stay put."""
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        if jid >= 0:
            qadr = model.jnt_qposadr[jid]
            data.ctrl[i] = data.qpos[qadr]


def _set_gripper_and_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_name: str,
    value: float,
    n_steps: int = 3000,
) -> None:
    """Hold all joints, override the gripper ctrl, and step until settled."""
    _hold_all_joints(model, data)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    data.ctrl[aid] = value
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def _get_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_name: str,
) -> float:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    jid = model.actuator_trnid[aid, 0]
    qadr = model.jnt_qposadr[jid]
    return float(data.qpos[qadr])


def _finger_pad_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_geom: str,
    right_geom: str,
) -> float:
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, left_geom)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, right_geom)
    assert lid >= 0 and rid >= 0, (
        f"Finger pad geom not found: {left_geom}({lid}), {right_geom}({rid})"
    )
    return float(np.linalg.norm(data.geom_xpos[lid] - data.geom_xpos[rid]))


# ── Test 1: qpos tracks ctrl monotonically ──────────────────────────────────


@pytest.mark.parametrize("cfg", _ROBOT_CONFIGS, ids=_IDS)
def test_qpos_tracks_ctrl_monotonically(cfg: dict) -> None:
    """As ctrl sweeps from min to max, the observed qpos must increase
    monotonically.  This proves observation and control share the same
    physical space, even if linkage forces prevent exact convergence."""
    model, data = _load_model(cfg)

    lo, hi = cfg["ctrl_range"]
    test_values = np.linspace(lo, hi, 8)
    qpos_values: list[float] = []

    for target in test_values:
        _reset(model, data)
        _set_gripper_and_step(model, data, cfg["actuator"], target)
        qpos_values.append(_get_joint_qpos(model, data, cfg["actuator"]))

    # qpos must be monotonically non-decreasing (ctrl increases → qpos increases)
    for i in range(len(qpos_values) - 1):
        assert qpos_values[i] <= qpos_values[i + 1] + 1e-4, (
            f"[{cfg['id']}] qpos not monotonically tracking ctrl: "
            f"ctrl={test_values[i]:.4f}→qpos={qpos_values[i]:.5f}, "
            f"ctrl={test_values[i + 1]:.4f}→qpos={qpos_values[i + 1]:.5f}"
        )

    # qpos range must be non-degenerate
    qpos_range = qpos_values[-1] - qpos_values[0]
    assert qpos_range > 0.001, (
        f"[{cfg['id']}] qpos range too small ({qpos_range:.5f}). "
        f"Gripper joint may not be responding to control."
    )

    print(f"\n[{cfg['id']}] qpos vs ctrl:")
    for v, q in zip(test_values, qpos_values):
        print(f"  ctrl={v:.4f} → qpos={q:.6f}")


# ── Test 2: finger distance changes monotonically with control ───────────────


@pytest.mark.parametrize("cfg", _ROBOT_CONFIGS, ids=_IDS)
def test_finger_distance_monotonic(cfg: dict) -> None:
    """Finger-pad distance must change monotonically as the control sweeps
    from fully open to fully closed."""
    model, data = _load_model(cfg)

    lo, hi = cfg["ctrl_range"]
    test_values = np.linspace(lo, hi, 8)
    distances: list[float] = []

    for target in test_values:
        _reset(model, data)
        _set_gripper_and_step(model, data, cfg["actuator"], target)
        distances.append(
            _finger_pad_distance(model, data, cfg["left_pad"], cfg["right_pad"])
        )

    # open_is_min=True: ctrl increases → finger distance decreases
    if cfg["open_is_min"]:
        for i in range(len(distances) - 1):
            assert distances[i] >= distances[i + 1] - 1e-4, (
                f"[{cfg['id']}] Finger distance not monotonically decreasing: "
                f"ctrl={test_values[i]:.4f}→dist={distances[i]:.5f}, "
                f"ctrl={test_values[i + 1]:.4f}→dist={distances[i + 1]:.5f}"
            )

    total_range = abs(distances[0] - distances[-1])
    assert total_range > 0.001, (
        f"[{cfg['id']}] Finger distance range too small ({total_range:.5f}m). "
        f"Fingers may not be moving."
    )

    print(f"\n[{cfg['id']}] Finger distance vs ctrl:")
    for v, d in zip(test_values, distances):
        print(f"  ctrl={v:.4f} → finger_dist={d:.5f}m")


def _framework_operator_config(cfg: dict) -> tuple[dict, str]:
    """Return (operators_dict, eef_key_fragment) for the given robot config."""
    if "xf9600" in cfg["id"]:
        return {
            "arm": {
                "arm_actuators": [
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                ],
                "eef_actuators": ["xfg_claw_joint"],
                "eef_output_name": "gripper",
            }
        }, "gripper"
    else:
        # Robotiq floating-base: arm is mocap-controlled, no arm actuators.
        return {
            "arm": {
                "arm_actuators": [],
                "eef_actuators": ["fingers_actuator"],
                "eef_output_name": "gripper",
            }
        }, "gripper"


# ── Test 3: capture_observation obs/action consistency via framework ─────────


@pytest.mark.parametrize("cfg", _ROBOT_CONFIGS, ids=_IDS)
def test_capture_observation_obs_action_consistent(cfg: dict) -> None:
    """The framework's capture_observation must report gripper measurement
    (qpos) and action (ctrl) values that are in the same space and track
    each other after convergence."""
    if not cfg["scene_xml"].exists():
        pytest.skip(f"Scene XML not found: {cfg['scene_xml']}")

    from auto_atom.basis.mjc.mujoco_basis import EnvConfig, DataType
    from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv

    operators, eef_key_fragment = _framework_operator_config(cfg)

    env_cfg = EnvConfig(
        model_path=cfg["scene_xml"],
        operators=operators,
        enabled_sensors={DataType.JOINT_POSITION},
        structured=True,
        sim_freq=1000,
        update_freq=1000,
    )

    env = UnifiedMujocoEnv(env_cfg)
    try:
        env.reset()
        env.model.opt.gravity[:] = 0

        lo, hi = cfg["ctrl_range"]
        test_values = [lo, (lo + hi) * 0.5, hi]

        for target in test_values:
            _hold_all_joints(env.model, env.data)
            aid = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg["actuator"]
            )
            ctrl = np.array(env.data.ctrl, dtype=np.float64)
            ctrl[aid] = target
            for _ in range(3000):
                env.step(ctrl)

            obs = env.capture_observation()

            # Find gripper measurement and action keys
            meas_key = None
            action_key = None
            for k in obs:
                if eef_key_fragment in k and "action" not in k and "joint_state" in k:
                    meas_key = k
                if eef_key_fragment in k and "action" in k and "joint_state" in k:
                    action_key = k

            assert meas_key is not None, (
                f"[{cfg['id']}] No gripper measurement key. Keys: {sorted(obs.keys())}"
            )
            assert action_key is not None, (
                f"[{cfg['id']}] No gripper action key. Keys: {sorted(obs.keys())}"
            )

            meas_pos = obs[meas_key]["data"]["position"]
            action_pos = obs[action_key]["data"]["position"]

            assert len(meas_pos) == len(action_pos), (
                f"[{cfg['id']}] Dimension mismatch: "
                f"meas={len(meas_pos)}, action={len(action_pos)}"
            )

            # The action (ctrl) channel must exactly equal the target we set.
            for i, a in enumerate(action_pos):
                assert abs(a - target) < 1e-6, (
                    f"[{cfg['id']}] Action channel does not match commanded ctrl: "
                    f"target={target:.4f}, action[{i}]={a:.4f}"
                )

            # The measurement (qpos) should be reasonably close to ctrl.
            # Linkage grippers have residual error, so allow up to 30% of range.
            max_err = (hi - lo) * 0.3
            for i, (m, a) in enumerate(zip(meas_pos, action_pos)):
                err = abs(m - a)
                assert err < max_err, (
                    f"[{cfg['id']}] Measurement too far from action at index {i}: "
                    f"meas={m:.4f}, action={a:.4f}, err={err:.4f}, max={max_err:.4f}"
                )

            print(
                f"  [{cfg['id']}] target={target:.4f}, "
                f"meas={meas_pos}, action={action_pos}"
            )
    finally:
        env.close()


# ── Test 4: apply_joint_action round-trip ────────────────────────────────────


@pytest.mark.parametrize("cfg", _ROBOT_CONFIGS, ids=_IDS)
def test_apply_joint_action_round_trip(cfg: dict) -> None:
    """Values written through apply_joint_action are faithfully reported
    in capture_observation — both measurement and action channels."""
    if not cfg["scene_xml"].exists():
        pytest.skip(f"Scene XML not found: {cfg['scene_xml']}")

    from auto_atom.basis.mjc.mujoco_basis import EnvConfig, DataType
    from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv

    operators, eef_key_fragment = _framework_operator_config(cfg)

    env_cfg = EnvConfig(
        model_path=cfg["scene_xml"],
        operators=operators,
        enabled_sensors={DataType.JOINT_POSITION},
        structured=False,
        sim_freq=1000,
        update_freq=1000,
    )

    env = UnifiedMujocoEnv(env_cfg)
    try:
        env.reset()
        env.model.opt.gravity[:] = 0

        lo, hi = cfg["ctrl_range"]
        test_values = [lo, (lo + hi) * 0.5, hi]

        for target in test_values:
            # Build action: current arm qpos + gripper target
            arm_qidx = env._op_arm_qidx["arm"]
            arm_pos = env.data.qpos[arm_qidx].copy()
            action = np.concatenate([arm_pos, [target]])

            for _ in range(3000):
                env.apply_joint_action("arm", action)

            obs = env.capture_observation()

            # Find keys (non-structured: …/position suffix)
            meas_key = None
            action_key = None
            for k in obs:
                if eef_key_fragment in k and "action" not in k and "position" in k:
                    meas_key = k
                if eef_key_fragment in k and "action" in k and "position" in k:
                    action_key = k

            assert meas_key is not None
            assert action_key is not None

            meas_val = float(np.asarray(obs[meas_key]["data"]).flat[0])
            action_val = float(np.asarray(obs[action_key]["data"]).flat[0])

            # Action channel must exactly match the ctrl we wrote.
            assert abs(action_val - target) < 1e-6, (
                f"[{cfg['id']}] Action obs != applied ctrl: "
                f"target={target:.4f}, action_obs={action_val:.4f}"
            )

            # Measurement should track action (within linkage tolerance).
            max_err = (hi - lo) * 0.3
            assert abs(meas_val - action_val) < max_err, (
                f"[{cfg['id']}] Measurement and action disagree: "
                f"meas={meas_val:.4f}, action={action_val:.4f}"
            )

        print(f"\n[{cfg['id']}] apply_joint_action round-trip: PASS")
    finally:
        env.close()


# ── Test 5: FingerDistanceMapper obs_map produces actual finger distances ────

_MAPPER_CONFIGS: list[dict] = [
    {
        "id": "xf9600_airbot_play",
        "scene_xml": _ASSETS / "scenes" / "open_door" / "demo_airbot_play.xml",
        "actuator": "xfg_claw_joint",
        "left_pad": "xfg_left_finger_pad_upper",
        "right_pad": "xfg_right_finger_pad_upper",
        "ctrl_range": (0.0, 0.02),
    },
    {
        "id": "robotiq_pick_and_place",
        "scene_xml": _ASSETS / "scenes" / "pick_and_place" / "demo.xml",
        "actuator": "fingers_actuator",
        "left_pad": "left_finger_pad",
        "right_pad": "right_finger_pad",
        "ctrl_range": (0.0, 0.82),
    },
]
_MAPPER_IDS = [c["id"] for c in _MAPPER_CONFIGS]


@pytest.mark.parametrize("cfg", _MAPPER_CONFIGS, ids=_MAPPER_IDS)
def test_finger_distance_mapper_obs_matches_geom_distance(cfg: dict) -> None:
    """obs_map output must match the actual geom-to-geom distance."""
    if not cfg["scene_xml"].exists():
        pytest.skip(f"Scene XML not found: {cfg['scene_xml']}")

    from auto_atom.mappers.finger_distance import FingerDistanceMapper

    model, data = _load_model(cfg)
    mapper = FingerDistanceMapper(
        left_pad_geom=cfg["left_pad"],
        right_pad_geom=cfg["right_pad"],
        actuator_name=cfg["actuator"],
    )
    mapper.bind(model, data)

    lo, hi = cfg["ctrl_range"]
    for target in np.linspace(lo, hi, 6):
        _reset(model, data)
        _set_gripper_and_step(model, data, cfg["actuator"], target)
        raw_qpos = np.array([_get_joint_qpos(model, data, cfg["actuator"])])
        mapped = mapper.obs_map(model, data, raw_qpos)
        actual_dist = _finger_pad_distance(
            model, data, cfg["left_pad"], cfg["right_pad"]
        )
        err = abs(float(mapped[0]) - actual_dist)
        assert err < 0.005, (
            f"[{cfg['id']}] obs_map({raw_qpos[0]:.5f})={mapped[0]:.5f} "
            f"but actual dist={actual_dist:.5f}, err={err:.5f}"
        )

    print(f"[{cfg['id']}] obs_map matches geom distance: PASS")


# ── Test 6: FingerDistanceMapper round-trip (obs_map ∘ ctrl_map ≈ identity) ──


@pytest.mark.parametrize("cfg", _MAPPER_CONFIGS, ids=_MAPPER_IDS)
def test_finger_distance_mapper_round_trip(cfg: dict) -> None:
    """ctrl_map(obs_map(qpos)) should recover a qpos close to the original."""
    if not cfg["scene_xml"].exists():
        pytest.skip(f"Scene XML not found: {cfg['scene_xml']}")

    from auto_atom.mappers.finger_distance import FingerDistanceMapper

    model, data = _load_model(cfg)
    mapper = FingerDistanceMapper(
        left_pad_geom=cfg["left_pad"],
        right_pad_geom=cfg["right_pad"],
        actuator_name=cfg["actuator"],
    )
    mapper.bind(model, data)

    lo, hi = cfg["ctrl_range"]
    for target in np.linspace(lo, hi, 8):
        _reset(model, data)
        _set_gripper_and_step(model, data, cfg["actuator"], target)
        raw_qpos = np.array([_get_joint_qpos(model, data, cfg["actuator"])])
        dist = mapper.obs_map(model, data, raw_qpos)
        recovered = mapper.ctrl_map(model, data, dist)
        err = abs(float(recovered[0]) - float(raw_qpos[0]))
        assert err < 0.002, (
            f"[{cfg['id']}] Round-trip error: qpos={raw_qpos[0]:.5f} "
            f"→ dist={dist[0]:.5f} → recovered={recovered[0]:.5f}, err={err:.5f}"
        )

    print(f"[{cfg['id']}] mapper round-trip: PASS")


# ── Test 7: framework with eef_mapper produces finger-distance observations ──


def test_framework_with_finger_distance_mapper() -> None:
    """When eef_mapper is configured, capture_observation should report
    finger distances instead of raw qpos, and apply_joint_action should
    accept finger distances as input."""
    scene = _ASSETS / "scenes" / "open_door" / "demo_airbot_play.xml"
    if not scene.exists():
        pytest.skip(f"Scene XML not found: {scene}")

    from auto_atom.basis.mjc.mujoco_basis import EnvConfig, DataType
    from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv
    from auto_atom.mappers.finger_distance import FingerDistanceMapper

    mapper = FingerDistanceMapper(
        left_pad_geom="xfg_left_finger_pad_upper",
        right_pad_geom="xfg_right_finger_pad_upper",
        actuator_name="xfg_claw_joint",
    )
    operators = {
        "arm": {
            "arm_actuators": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
            "eef_actuators": ["xfg_claw_joint"],
            "eef_output_name": "gripper",
            "eef_mapper": mapper,
        }
    }
    env_cfg = EnvConfig(
        model_path=scene,
        operators=operators,
        enabled_sensors={DataType.JOINT_POSITION},
        structured=True,
        sim_freq=1000,
        update_freq=1000,
    )
    env = UnifiedMujocoEnv(env_cfg)
    try:
        env.reset()
        env.model.opt.gravity[:] = 0

        # Command gripper ctrl=0.01 (raw) and step to converge.
        _hold_all_joints(env.model, env.data)
        aid = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "xfg_claw_joint"
        )
        ctrl = np.array(env.data.ctrl, dtype=np.float64)
        ctrl[aid] = 0.01
        for _ in range(3000):
            env.step(ctrl)

        obs = env.capture_observation()
        meas_key = None
        action_key = None
        for k in obs:
            if "gripper" in k and "action" not in k and "joint_state" in k:
                meas_key = k
            if "gripper" in k and "action" in k and "joint_state" in k:
                action_key = k

        assert meas_key is not None
        assert action_key is not None

        meas_pos = obs[meas_key]["data"]["position"][0]
        action_pos = obs[action_key]["data"]["position"][0]

        # Observation should be in finger-distance space (>>0.02, roughly 0.01-0.11m).
        assert meas_pos > 0.02, (
            f"Mapped observation {meas_pos:.5f} looks like raw qpos, not finger distance"
        )
        assert action_pos > 0.02, (
            f"Mapped action {action_pos:.5f} looks like raw qpos, not finger distance"
        )

        # The actual finger distance for reference:
        lid = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "xfg_left_finger_pad_upper"
        )
        rid = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "xfg_right_finger_pad_upper"
        )
        actual_dist = float(
            np.linalg.norm(env.data.geom_xpos[lid] - env.data.geom_xpos[rid])
        )
        assert abs(meas_pos - actual_dist) < 0.005, (
            f"Mapped meas {meas_pos:.5f} != actual dist {actual_dist:.5f}"
        )

        print(f"  meas (finger dist)  = {meas_pos:.5f}m")
        print(f"  action (finger dist)= {action_pos:.5f}m")
        print(f"  actual geom dist    = {actual_dist:.5f}m")

        # Now test apply_joint_action with a finger-distance value.
        # Command 0.05m finger distance via apply_joint_action.
        target_dist = 0.05
        arm_qidx = env._op_arm_qidx["arm"]
        arm_pos = env.data.qpos[arm_qidx].copy()
        action = np.concatenate([arm_pos, [target_dist]])
        for _ in range(3000):
            env.apply_joint_action("arm", action)

        obs2 = env.capture_observation()
        meas_pos2 = obs2[meas_key]["data"]["position"][0]
        actual_dist2 = float(
            np.linalg.norm(env.data.geom_xpos[lid] - env.data.geom_xpos[rid])
        )

        assert abs(meas_pos2 - actual_dist2) < 0.005
        # The convergence tolerance is larger here because the raw joint
        # convergence error (~0.004m) is amplified ~5x by the linkage ratio.
        assert abs(meas_pos2 - target_dist) < 0.035, (
            f"After commanding {target_dist:.3f}m, got {meas_pos2:.5f}m "
            f"(actual={actual_dist2:.5f}m)"
        )

        print(f"  commanded dist      = {target_dist:.5f}m")
        print(f"  observed dist       = {meas_pos2:.5f}m")
        print(f"  actual geom dist    = {actual_dist2:.5f}m")
        print("  framework with mapper: PASS")
    finally:
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
