"""End-effector control for the MJWarp backend.

This is the ``control_eef`` half of :class:`~auto_atom.contracts.OperatorHandler`.
``move_to_pose`` needs IK and lands in a later round; the two are separated
because the gripper path depends only on pieces that already exist (actuator
writes, both halves of the grasp verdict, the eef-pose accessors) while the arm
path does not.

Each call advances a complete control update for the selected worlds before
evaluating completion. The scene-state adapter preserves inactive worlds when
the runtime dispatches one environment at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from auto_atom.backend.mjwarp.grasp import lateral_grasp_ok
from auto_atom.backend.mjwarp.operator_state import (
    MjWarpOperatorState,
    get_eef_pose_in_world,
)
from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.execution_model import ControlResult, ControlSignal


@dataclass
class MjWarpEefControl:
    """Per-world gripper control, reporting the same signals as the native path.

    ``eef_open_value`` / ``eef_close_value`` are the actuator commands for the
    two ends of travel, and ``grasp_axis`` is the axis the fingers close along
    (so the lateral check measures the other two).
    """

    state: MjWarpSceneState
    operator: MjWarpOperatorState

    eef_open_value: float = 0.0
    eef_close_value: float = 0.82
    eef_tolerance: float = 0.03
    timeout_steps: int = 100
    settle_steps: int = 5
    release_settle_steps: int = 0
    lateral_threshold: float = 0.0
    grasp_axis: int = 2
    n_substeps: int = 1

    _steps: np.ndarray = field(init=False, repr=False)
    _last_command_key: List[Optional[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nworld = self.state.nworld
        self._steps = np.zeros(nworld, dtype=np.int64)
        self._last_command_key = [None] * nworld

    @classmethod
    def for_operator(
        cls,
        state: MjWarpSceneState,
        operator: MjWarpOperatorState,
        **overrides: Any,
    ) -> "MjWarpEefControl":
        """Build with open/close/tolerance derived from the gripper's ctrlrange.

        The class defaults are robotiq-shaped (0 → 0.82) and are wrong by two
        orders of magnitude for other grippers. The UMI claw is
        ``ctrlrange="0 0.0165"``, where the default 0.03 tolerance exceeds the
        entire travel: ``actual >= command - tolerance`` then holds with the
        gripper fully open, so every close reports REACHED on the first tick with
        the fingers unmoved. Nothing errors -- the grasp check simply always
        succeeds. Native derives these per robot for the same reason
        (``mujoco_backend.py:2390-2408``), and this is the MJWarp equivalent.

        Tolerance is clamped to a fifth of the travel, bounded to
        ``[1e-4, 0.03]``, matching native: a fifth is loose enough to accept a
        gripper that has effectively closed, tight enough not to accept one that
        has barely moved.

        Explicit ``overrides`` win, so a config that states a value keeps it.
        """
        derived: dict[str, Any] = {}
        eef_ids = operator.eef_actuator_ids
        if eef_ids.size:
            actuator = int(eef_ids[0])
            low, high = (
                float(v) for v in state.host_model.actuator_ctrlrange[actuator]
            )
            span = high - low
            if span > 0:
                derived["eef_open_value"] = low
                derived["eef_close_value"] = high
                derived["eef_tolerance"] = min(0.03, max(1e-4, span * 0.2))

        derived.update(overrides)
        return cls(state=state, operator=operator, **derived)

    # ------------------------------------------------------------------
    # Command resolution
    # ------------------------------------------------------------------

    def target_value(self, close: bool, joint_positions: Any = None) -> float:
        """Actuator command for a requested gripper state.

        An explicit ``joint_positions`` wins over the open/close pair, matching
        native: a config that names a finger angle means that angle, not "as
        closed as this gripper goes".
        """
        if joint_positions:
            return float(joint_positions[0])
        return self.eef_close_value if close else self.eef_open_value

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def control(
        self,
        *,
        close: bool,
        require_grasp: bool = False,
        joint_positions: Any = None,
        target_body_name: Optional[str] = None,
        world_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance the gripper one control tick for the selected worlds.

        Writes each selected world's gripper command and advances its configured
        number of physical substeps, leaving other worlds unchanged.

        ``target_body_name`` is the grasp target. It is required for
        ``require_grasp``: without a target there is nothing to confirm a grasp
        against, which native reports as a config error rather than a failed
        grasp, and so does this.
        """
        nworld = self.state.nworld
        mask = self._normalize_mask(world_mask)
        command = self.target_value(close, joint_positions)
        signals = np.asarray([ControlSignal.RUNNING] * nworld, dtype=object)
        details: List[Dict[str, Any]] = [{} for _ in range(nworld)]

        key = f"{close}:{joint_positions}:{require_grasp}"
        target_ids = (
            self.state.descendant_body_ids(target_body_name)
            if target_body_name
            else None
        )
        finger_sides = self.state.finger_geom_sides(
            self.state.operator_body_ids(self.operator.root_body_name)
        )

        worlds = np.flatnonzero(mask)
        if worlds.size == 0:
            return ControlResult(signals=signals, details=details)

        if close and require_grasp and target_ids is None:
            for world in worlds:
                signals[world] = ControlSignal.FAILED
                details[world] = {
                    "event": "grasp_target_required",
                    "failure_category": "missing_grasp_target",
                    "failure_reason": (
                        "require_grasp=true needs a non-empty Stage target object"
                    ),
                }
                self._steps[world] = 0
            return ControlResult(signals=signals, details=details)

        for world in worlds:
            if self._last_command_key[world] != key:
                self._last_command_key[world] = key
                self._steps[world] = 0

        eef_ids = self.operator.eef_actuator_ids
        if eef_ids.size == 0:
            raise ValueError(
                f"Operator '{self.operator.name}' has no eef_actuators, so its "
                "gripper cannot be commanded."
            )
        self.state.set_ctrl(
            eef_ids, np.full(eef_ids.size, command, dtype=np.float64), world_mask=mask
        )
        self.state.step(self.n_substeps, world_mask=mask)
        self._steps[worlds] += 1

        self._evaluate(
            worlds=worlds,
            close=close,
            require_grasp=require_grasp,
            command=command,
            target_ids=target_ids,
            target_body_name=target_body_name,
            finger_sides=finger_sides,
            signals=signals,
            details=details,
        )
        return ControlResult(signals=signals, details=details)

    def _evaluate(
        self,
        *,
        worlds: np.ndarray,
        close: bool,
        require_grasp: bool,
        command: float,
        target_ids: Optional[frozenset],
        target_body_name: Optional[str],
        finger_sides: Dict[int, str],
        signals: np.ndarray,
        details: List[Dict[str, Any]],
    ) -> None:
        """Decide REACHED / RUNNING / TIMED_OUT per world after the step."""
        qpos_indices = self.operator.eef_qpos_indices
        actual_all = (
            self.state.get_joint_positions(qpos_indices)
            if qpos_indices.size
            else np.zeros((self.state.nworld, 1))
        )
        eef_positions, eef_orientations = get_eef_pose_in_world(
            self.state, self.operator
        )
        object_positions = (
            self.state.get_body_pose_batch(target_body_name)[0]
            if target_body_name
            else None
        )

        for world in worlds:
            actual = float(actual_all[world][0]) if qpos_indices.size else 0.0
            steps = int(self._steps[world])
            settle_ready = steps >= self.settle_steps

            grasp: Optional[Dict[str, Any]] = None
            if close and target_ids is not None:
                grasp = self._grasp_verdict(
                    world,
                    target_ids,
                    finger_sides,
                    object_positions[world] if object_positions is not None else None,
                    eef_positions[world],
                    eef_orientations[world],
                )

            reached, event = self._completion(
                close=close,
                require_grasp=require_grasp,
                command=command,
                actual=actual,
                steps=steps,
                settle_ready=settle_ready,
                grasp=grasp,
            )

            details[world] = {
                "event": event,
                "operator": self.operator.name,
                "eef_target": command,
                "eef_actual": actual,
                "eef_error": abs(actual - command),
                "settle_ready": settle_ready,
                "steps": steps,
                "grasped_object": (
                    target_body_name
                    if event == "eef_grasped" and target_body_name
                    else ""
                ),
            }
            if grasp is not None:
                details[world]["grasp_check"] = grasp

            if reached:
                signals[world] = ControlSignal.REACHED
                self._steps[world] = 0
            elif steps >= self.timeout_steps:
                details[world]["event"] = "eef_timeout"
                signals[world] = ControlSignal.TIMED_OUT
            else:
                signals[world] = ControlSignal.RUNNING

    def _grasp_verdict(
        self,
        world: int,
        target_ids: frozenset,
        finger_sides: Dict[int, str],
        object_position: Optional[np.ndarray],
        eef_position: np.ndarray,
        eef_orientation: np.ndarray,
    ) -> Dict[str, Any]:
        """Compose the contact and lateral halves into one verdict."""
        left, right = self.state.finger_contacts_with_target(
            world, target_ids, finger_sides
        )
        if object_position is None:
            lateral_ok, lateral_error = True, 0.0
        else:
            lateral_ok, lateral_error = lateral_grasp_ok(
                object_position,
                eef_position,
                eef_orientation,
                self.grasp_axis,
                self.lateral_threshold,
            )
        return {
            "left_contact": left,
            "right_contact": right,
            "lateral_ok": lateral_ok,
            "lateral_error": lateral_error,
            "lateral_threshold": self.lateral_threshold,
        }

    def _completion(
        self,
        *,
        close: bool,
        require_grasp: bool,
        command: float,
        actual: float,
        steps: int,
        settle_ready: bool,
        grasp: Optional[Dict[str, Any]],
    ) -> tuple[bool, str]:
        """``(reached, event)`` following the native acceptance ladder.

        The order matters and each rung has a distinct reason:

        1. A confirmed grasp (both fingers plus centred) is the strongest
           evidence and wins outright.
        2. Otherwise a close that does not require a grasp accepts on the
           commanded angle being reached.
        3. A gripper physically blocked by the object may never reach that
           angle, so after enough settling time a finger that has moved
           noticeably off the open position also counts. Without this rung a
           successful grasp on a stiff object reads as a timeout.
        4. Opening accepts on returning to the open position, after the
           configured release settling.

        Rungs 2 and 3 inherit a **sign assumption** from the native ladder that
        is worth stating because nothing enforces it: closing must *increase*
        the gripper joint value (Robotiq travels 0 open -> 0.82 closed). The
        comparisons are ``actual >= command - tolerance`` and ``actual > open +
        tolerance * 0.1``, so a gripper whose close direction *decreases* qpos
        would satisfy both at its open position and every close would be
        accepted on the first tick without the fingers having moved.
        ``eef_close_value`` must therefore be greater than ``eef_open_value``.
        """
        if grasp is not None and settle_ready:
            confirmed = (
                grasp["left_contact"] and grasp["right_contact"] and grasp["lateral_ok"]
            )
            if confirmed:
                return True, "eef_grasped"

        if close and not require_grasp:
            if actual >= command - self.eef_tolerance:
                return True, "eef_reached"
            if (
                steps >= max(self.settle_steps, 30)
                and actual > self.eef_open_value + self.eef_tolerance * 0.1
            ):
                return True, "eef_reached"
        elif not close:
            if (
                steps >= self.release_settle_steps
                and actual <= self.eef_open_value + self.eef_tolerance
            ):
                return True, "eef_reached"

        return False, "eef_moving"

    def _normalize_mask(self, world_mask: Optional[np.ndarray]) -> np.ndarray:
        nworld = self.state.nworld
        if world_mask is None:
            return np.ones(nworld, dtype=bool)
        mask = np.asarray(world_mask, dtype=bool).reshape(-1)
        if mask.shape != (nworld,):
            raise ValueError(
                f"world_mask must have shape ({nworld},), got {mask.shape}"
            )
        return mask
