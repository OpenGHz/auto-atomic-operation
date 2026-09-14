"""Backend-level grasp and contact queries for the MJWarp backend.

These are the questions stage post-conditions ask -- "is the plate grasped?",
"what is the gripper holding?", "is the tool touching the rack?" -- as opposed
to the per-tick verdict the gripper controller computes. They compose the
primitives from earlier rounds (the world-filtered contact scan, subtree body
ids, finger-side classification, lateral geometry) into the batched answers the
``SceneBackend`` contract returns.

The batched shape is the point: every query answers for **all** worlds at once,
because a stage post-condition is evaluated per environment and the runtime
indexes the result by env. Native loops over its per-replica ``MjData`` to build
the same array.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from auto_atom.backend.mjwarp.grasp import lateral_grasp_ok
from auto_atom.backend.mjwarp.operator_state import (
    MjWarpOperatorState,
    get_eef_pose_in_world,
)
from auto_atom.basis.mjwarp.state import MjWarpSceneState


class MjWarpGraspQueries:
    """Grasp/contact questions answered across every world.

    Topology that cannot change during a run -- the operator's body subtree and
    its finger-geom classification -- is resolved once here rather than per
    query, since a post-condition check runs on every control tick.
    """

    def __init__(
        self,
        state: MjWarpSceneState,
        operator: MjWarpOperatorState,
        *,
        lateral_threshold: float = 0.0,
        grasp_axis: int = 2,
    ) -> None:
        self.state = state
        self.operator = operator
        self.lateral_threshold = lateral_threshold
        self.grasp_axis = grasp_axis
        self._operator_bodies = state.operator_body_ids(operator.root_body_name)
        self._finger_sides = state.finger_geom_sides(self._operator_bodies)
        self._target_bodies: Dict[str, frozenset] = {}

    def _bodies_for(self, body_name: str) -> frozenset:
        """Subtree body ids for a target, cached: model topology is static."""
        cached = self._target_bodies.get(body_name)
        if cached is None:
            cached = self.state.descendant_body_ids(body_name)
            self._target_bodies[body_name] = cached
        return cached

    def is_object_grasped(self, body_name: str) -> np.ndarray:
        """``(nworld,)`` bool: both fingers on the target and it sits centred.

        Same two-part verdict the gripper controller uses, so a stage
        post-condition and the controller that satisfied it cannot disagree.
        """
        result = np.zeros(self.state.nworld, dtype=bool)
        target_bodies = self._bodies_for(body_name)
        if not target_bodies:
            return result

        object_positions, _ = self.state.get_body_pose_batch(body_name)
        eef_positions, eef_orientations = get_eef_pose_in_world(
            self.state, self.operator
        )
        for world in range(self.state.nworld):
            left, right = self.state.finger_contacts_with_target(
                world, target_bodies, self._finger_sides
            )
            if not (left and right):
                continue
            lateral_ok, _ = lateral_grasp_ok(
                object_positions[world],
                eef_positions[world],
                eef_orientations[world],
                self.grasp_axis,
                self.lateral_threshold,
            )
            result[world] = lateral_ok
        return result

    def is_operator_grasping(self, body_names: Mapping[str, str]) -> np.ndarray:
        """``(nworld,)`` bool: holding *any* of the known objects.

        ``body_names`` maps logical object name to MJCF body name, because the
        two differ and contacts are keyed on the body.
        """
        result = np.zeros(self.state.nworld, dtype=bool)
        for body_name in body_names.values():
            result |= self.is_object_grasped(body_name)
        return result

    def grasped_object_name(
        self,
        body_names: Mapping[str, str],
        world_index: int,
    ) -> Optional[str]:
        """Logical name of whatever one world is holding, or ``None``.

        Returns the first match in iteration order, as native does. A gripper
        holding two objects at once is not a state the contract describes, so
        there is no tie-break to define.
        """
        if not 0 <= world_index < self.state.nworld:
            raise IndexError(
                f"world_index must be in [0, {self.state.nworld}); got {world_index}."
            )
        for name, body_name in body_names.items():
            if bool(self.is_object_grasped(body_name)[world_index]):
                return name
        return None

    def is_operator_contacting(self, body_name: str) -> np.ndarray:
        """``(nworld,)`` bool: any operator geom touches the target at all.

        Weaker than a grasp on purpose -- this is what a ``press`` or ``push``
        post-condition needs, where one finger brushing the target counts and
        neither two-sided contact nor centring is required.
        """
        result = np.zeros(self.state.nworld, dtype=bool)
        target_bodies = self._bodies_for(body_name)
        if not target_bodies:
            return result

        for world in range(self.state.nworld):
            for body1, body2 in self.state.get_contact_body_pairs(world):
                first, second = int(body1), int(body2)
                touches_operator = (
                    first in self._operator_bodies or second in self._operator_bodies
                )
                touches_target = first in target_bodies or second in target_bodies
                # Both sides must be present *and* be different bodies, so an
                # operator-internal or target-internal contact does not count.
                if touches_operator and touches_target and first != second:
                    result[world] = True
                    break
        return result
