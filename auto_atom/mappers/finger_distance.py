"""EEF mapper that converts between raw joint qpos/ctrl and finger-pad distance.

Works for any parallel-linkage gripper where the finger opening can be
measured as the Euclidean distance between two geom centers.  At ``bind()``
time the mapper sweeps the actuator range, records the (qpos, finger_dist)
curve, and builds a pair of monotonic interpolation functions for fast
runtime conversion.

Configure in YAML via Hydra ``_target_``::

    env:
      operators:
        arm:
          eef_mapper:
            _target_: auto_atom.mappers.finger_distance.FingerDistanceMapper
            left_pad_geom: xfg_left_finger_pad_upper
            right_pad_geom: xfg_right_finger_pad_upper
            actuator_name: xfg_claw_joint
"""

from __future__ import annotations

import mujoco
import numpy as np


class FingerDistanceMapper:
    """Bidirectional mapper between raw joint position and finger-pad distance.

    Methods
    -------
    bind(model, data)
        Build lookup tables by sweeping the actuator range.
    obs_map(model, data, raw) -> np.ndarray
        Forward: raw qpos/ctrl → finger distance (meters).
    ctrl_map(model, data, user) -> np.ndarray
        Inverse: finger distance (meters) → actuator ctrl value.
    """

    def __init__(
        self,
        left_pad_geom: str,
        right_pad_geom: str,
        actuator_name: str,
        n_samples: int = 16,
    ) -> None:
        self._left_pad_geom = left_pad_geom
        self._right_pad_geom = right_pad_geom
        self._actuator_name = actuator_name
        self._n_samples = n_samples
        # Filled by bind():
        self._qpos_lut: np.ndarray | None = None  # sorted ascending qpos
        self._dist_lut: np.ndarray | None = None  # corresponding finger distances

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Build a lookup table (LUT) mapping raw joint qpos to finger-pad distance.

        For parallel-linkage grippers (e.g. xf9600), the actuated slide joint
        ``clawj`` drives finger opening through equality constraints connecting
        passive linkage joints.  The relationship between ``clawj`` qpos and
        the actual finger-pad Euclidean distance is nonlinear and cannot be
        derived analytically from the XML alone.

        This method determines the mapping empirically:

        1. Sweep the actuator's ctrl range over ``n_samples`` evenly-spaced
           values from ``ctrl_lo`` to ``ctrl_hi``.
        2. For each ctrl value, reset the scene, hold all other actuators at
           their keyframe positions, set the gripper ctrl, and run ``mj_step``
           to let the equality constraints settle the passive joints.
        3. Record the resulting ``(qpos, finger_pad_distance)`` pair, where
           ``finger_pad_distance`` is the Euclidean distance between the
           world-frame centers of the left and right finger-pad geoms.

        The collected pairs are stored as two sorted arrays (``_qpos_lut``,
        ``_dist_lut``) used by ``obs_map`` / ``ctrl_map`` for piecewise-linear
        interpolation at runtime.

        Gravity is temporarily disabled and the model/data state is fully
        saved and restored, so calling ``bind`` has no side effects on the
        simulation state.
        """
        print(
            f"FingerDistanceMapper: building LUT for '{self._actuator_name}' "
            f"using geoms '{self._left_pad_geom}' and '{self._right_pad_geom}'..."
        )
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._left_pad_geom)
        rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._right_pad_geom)
        if lid < 0:
            raise ValueError(
                f"FingerDistanceMapper: left geom '{self._left_pad_geom}' not found"
            )
        if rid < 0:
            raise ValueError(
                f"FingerDistanceMapper: right geom '{self._right_pad_geom}' not found"
            )
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._actuator_name
        )
        if aid < 0:
            raise ValueError(
                f"FingerDistanceMapper: actuator '{self._actuator_name}' not found"
            )
        jid = model.actuator_trnid[aid, 0]
        qadr = model.jnt_qposadr[jid]
        ctrl_lo = float(model.actuator_ctrlrange[aid, 0])
        ctrl_hi = float(model.actuator_ctrlrange[aid, 1])

        # Save and restore model/data state around the sweep.
        saved_gravity = model.opt.gravity.copy()
        saved_timestep = model.opt.timestep
        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()
        saved_ctrl = data.ctrl.copy()

        model.opt.gravity[:] = 0
        model.opt.timestep = 0.001

        ctrl_values = np.linspace(ctrl_lo, ctrl_hi, self._n_samples)
        qpos_samples = np.empty(self._n_samples)
        dist_samples = np.empty(self._n_samples)

        for i, cv in enumerate(ctrl_values):
            # Reset to keyframe / zero state before each sample.
            mujoco.mj_resetData(model, data)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            # Hold all actuators at initial positions.
            for a in range(model.nu):
                j = model.actuator_trnid[a, 0]
                if j >= 0:
                    data.ctrl[a] = data.qpos[model.jnt_qposadr[j]]
            data.ctrl[aid] = cv
            for _ in range(1500):
                mujoco.mj_step(model, data)
            qpos_samples[i] = float(data.qpos[qadr])
            dist_samples[i] = float(
                np.linalg.norm(data.geom_xpos[lid] - data.geom_xpos[rid])
            )

        # Restore model/data state.
        model.opt.gravity[:] = saved_gravity
        model.opt.timestep = saved_timestep
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(model, data)

        # Ensure qpos is sorted ascending for np.interp (it should be since
        # ctrl sweeps monotonically and qpos tracks monotonically).
        order = np.argsort(qpos_samples)
        self._qpos_lut = qpos_samples[order]
        self._dist_lut = dist_samples[order]

    def obs_map(
        self, model: mujoco.MjModel, data: mujoco.MjData, raw: np.ndarray
    ) -> np.ndarray:
        """Forward map: raw joint qpos/ctrl → finger distance (meters)."""
        if self._qpos_lut is None:
            raise RuntimeError("FingerDistanceMapper.bind() has not been called.")
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        return np.interp(raw, self._qpos_lut, self._dist_lut)

    def ctrl_map(
        self, model: mujoco.MjModel, data: mujoco.MjData, user: np.ndarray
    ) -> np.ndarray:
        """Inverse map: finger distance (meters) → actuator ctrl value.

        Since finger distance decreases as qpos increases, the inverse
        interpolation uses the reversed (ascending distance) lookup.
        """
        if self._dist_lut is None:
            raise RuntimeError("FingerDistanceMapper.bind() has not been called.")
        user = np.asarray(user, dtype=np.float64).reshape(-1)
        # np.interp requires xp to be increasing.  _dist_lut is typically
        # decreasing (larger qpos → smaller distance), so reverse both.
        if self._dist_lut[-1] < self._dist_lut[0]:
            return np.interp(user, self._dist_lut[::-1], self._qpos_lut[::-1])
        return np.interp(user, self._dist_lut, self._qpos_lut)
