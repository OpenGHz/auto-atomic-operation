"""Per-world grasp-attachment welds for the MJWarp scene state.

The welds themselves are compiled into the scene, inactive (see
:mod:`auto_atom.scene_composition.attachments`). This module turns them on and
off per world: activating one writes the object's current pose in the gripper
frame into that world's row of ``eq_data`` and sets that world's
``eq_active``. ``eq_data`` is a model field, so the env batches it per world
whenever it carries attachment welds; otherwise every world would share one
grasp pose.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.scene_composition import grasp_weld_name


class MjWarpGraspAttachments:
    """Which object each operator holds by its weld, in every world."""

    def __init__(
        self,
        state: MjWarpSceneState,
        attachments: Iterable[Tuple[str, str]],
    ) -> None:
        """``attachments`` lists the ``(operator, object)`` welds the scene has."""
        import mujoco

        self.state = state
        self._welds: Dict[Tuple[str, str], int] = {}
        for operator, object_name in attachments:
            name = grasp_weld_name(operator, object_name)
            equality = mujoco.mj_name2id(
                state.host_model, mujoco.mjtObj.mjOBJ_EQUALITY, name
            )
            if equality < 0:
                raise ValueError(f"Grasp attachment weld '{name}' is missing.")
            self._welds[(operator, object_name)] = int(equality)
        if self._welds and int(state.model.eq_data.shape[0]) != state.nworld:
            raise ValueError(
                "Grasp attachments need eq_data batched per world; build the "
                "scene state with 'eq_data' in its batched fields."
            )
        # operator -> per-world held object (None when nothing is welded).
        self._held: Dict[str, list[Optional[str]]] = {}

    @property
    def welds(self) -> Dict[Tuple[str, str], int]:
        """``(operator, object) -> equality id`` of every weld."""
        return dict(self._welds)

    def weld_bodies(self, operator: str) -> Dict[str, Tuple[int, int]]:
        """``object -> (gripper body id, object body id)`` of ``operator``'s welds."""
        model = self.state.host_model
        return {
            object_name: (
                int(model.eq_obj1id[equality]),
                int(model.eq_obj2id[equality]),
            )
            for (owner, object_name), equality in self._welds.items()
            if owner == operator
        }

    def attached_object(self, operator: str, world: int) -> Optional[str]:
        """The object ``operator`` holds by its weld in ``world``, if any."""
        held = self._held.get(operator)
        return None if held is None else held[int(world)]

    def attached_bodies(self, operator: str) -> np.ndarray:
        """``(nworld,)`` body id of the welded object per world, ``-1`` for none."""
        bodies = np.full(self.state.nworld, -1, dtype=np.int64)
        for world, object_name in enumerate(self._held.get(operator, ())):
            if object_name is not None:
                equality = self._welds[(operator, object_name)]
                bodies[world] = int(self.state.host_model.eq_obj2id[equality])
        return bodies

    def attach(self, operator: str, object_name: str, world_mask: np.ndarray) -> None:
        """Weld ``object_name`` to ``operator``'s gripper in the masked worlds.

        Each world's weld holds the object at its current pose relative to the
        gripper, so nothing jumps. A world holding another object by its weld
        releases that one first.
        """
        equality = self._welds.get((operator, object_name))
        if equality is None:
            raise KeyError(
                f"No grasp attachment weld for operator '{operator}' and object "
                f"'{object_name}'."
            )
        held = self._held_for(operator)
        worlds = [
            int(world)
            for world in np.flatnonzero(np.asarray(world_mask, dtype=bool))
            if held[int(world)] != object_name
        ]
        if not worlds:
            return
        self.release(operator, _mask(self.state.nworld, worlds))

        import mujoco

        # Poses at the current qpos; the last step leaves xpos one substep old.
        self.state.forward()
        model = self.state.host_model
        gripper = int(model.eq_obj1id[equality])
        target = int(model.eq_obj2id[equality])
        xpos = self.state.data.xpos.numpy()
        xquat = self.state.data.xquat.numpy()
        eq_data = self.state.model.eq_data.numpy().copy()
        for world in worlds:
            gripper_quat = np.asarray(xquat[world][gripper], dtype=np.float64)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, gripper_quat)
            offset = np.asarray(xpos[world][target] - xpos[world][gripper], np.float64)
            inverse = np.zeros(4)
            mujoco.mju_negQuat(inverse, gripper_quat)
            relative_quat = np.zeros(4)
            mujoco.mju_mulQuat(
                relative_quat, inverse, np.asarray(xquat[world][target], np.float64)
            )
            # Weld data: anchor (3, in the object frame), the object's pose in
            # the gripper frame (position 3, quaternion 4 in wxyz), torquescale.
            row = np.asarray(eq_data[world][equality], dtype=np.float64)
            row[0:3] = 0.0
            row[3:6] = rotation.reshape(3, 3).T @ offset
            row[6:10] = relative_quat
            eq_data[world][equality] = row
        self.state.model.eq_data.assign(eq_data)
        self._set_active(equality, worlds, True)
        for world in worlds:
            held[world] = object_name

    def release(self, operator: str, world_mask: np.ndarray) -> list[Optional[str]]:
        """Deactivate ``operator``'s welds in the masked worlds.

        Returns, per world, the object that was released (``None`` where
        nothing was welded or the world is unmasked).
        """
        released: list[Optional[str]] = [None] * self.state.nworld
        held = self._held.get(operator)
        if held is None:
            return released
        by_equality: Dict[int, list[int]] = {}
        for world in np.flatnonzero(np.asarray(world_mask, dtype=bool)):
            object_name = held[int(world)]
            if object_name is None:
                continue
            released[int(world)] = object_name
            by_equality.setdefault(self._welds[(operator, object_name)], []).append(
                int(world)
            )
            held[int(world)] = None
        for equality, worlds in by_equality.items():
            self._set_active(equality, worlds, False)
        return released

    def reset(self, world_mask: np.ndarray) -> None:
        """Release every weld in the masked worlds, as a reset does."""
        worlds = [int(w) for w in np.flatnonzero(np.asarray(world_mask, dtype=bool))]
        if not worlds or not self._welds:
            return
        for held in self._held.values():
            for world in worlds:
                held[world] = None
        for equality in self._welds.values():
            self._set_active(equality, worlds, False)

    def _held_for(self, operator: str) -> list[Optional[str]]:
        return self._held.setdefault(operator, [None] * self.state.nworld)

    def _set_active(self, equality: int, worlds: list[int], active: bool) -> None:
        eq_active = self.state.data.eq_active.numpy().copy()
        eq_active[worlds, equality] = active
        self.state.data.eq_active.assign(eq_active)


def _mask(nworld: int, worlds: list[int]) -> np.ndarray:
    mask = np.zeros(nworld, dtype=bool)
    mask[worlds] = True
    return mask
