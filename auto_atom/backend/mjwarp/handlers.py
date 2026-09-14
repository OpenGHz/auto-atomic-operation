"""Object handlers for the MJWarp backend.

These satisfy :class:`~auto_atom.contracts.ObjectHandler`, which is what
``SceneBackend.apply_object_pose`` routes through -- and therefore the entire
transport mechanism for ``execution.mode: object_only``.

The handler is deliberately thin. Every simulator detail (name resolution, the
free-joint vs static dispatch, the world -> parent-local conversion, batched
readbacks) already lives in :class:`~auto_atom.basis.mjwarp.state.MjWarpSceneState`,
so this layer only bridges the runtime's batched ``PoseState`` and ``env_mask``
to that adapter.

One difference from the native handler is worth stating: the native path keeps
one ``MjModel``/``MjData`` pair per replica and loops over them, and its
``_stateful_pose_indices`` exists to collapse a masked write onto a single
physical row for Gaussian-Splatting shared-physics batches. MJWarp has one
device model with genuine per-world state, so there is no aliasing to collapse
and ``env_mask`` maps directly onto ``world_mask``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.contracts import ObjectHandler
from auto_atom.utils.pose import PoseState


@dataclass
class MjWarpObjectHandler(ObjectHandler):
    """One scene object, addressed by its MJCF body name.

    ``name`` is the logical name a task config uses; ``body_name`` is the MJCF
    body it resolves to. They are frequently equal but need not be, which is why
    the native backend keeps them separate too.

    ``freejoint_name`` is an optional override for the joint that drives the
    body. Left unset, the body's own joints are inspected, so a scene owns its
    joint names rather than the backend guessing a convention.
    """

    state: MjWarpSceneState = None  # type: ignore[assignment]
    body_name: str = ""
    freejoint_name: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state is None:
            raise ValueError(
                f"Object '{self.name}' requires an MjWarpSceneState instance."
            )
        if not self.body_name:
            raise ValueError(f"Object '{self.name}' requires a non-empty body_name.")

    def get_pose(self) -> PoseState:
        """Every world's pose for this object, as one batched ``PoseState``."""
        positions, orientations = self.state.get_body_pose_batch(self.body_name)
        return PoseState(position=positions, orientation=orientations)

    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Place this object, one pose per selected world.

        The pose is broadcast to the batch first, so a batch-1 ``PoseState``
        applies to every selected world while a full batch gives each world its
        own sample -- which is what randomization produces.
        """
        nworld = self.state.nworld
        pose = pose.broadcast_to(nworld)
        mask = self._normalize_mask(env_mask, nworld)
        if mask is not None and not mask.any():
            return

        self.state.set_object_pose(
            self.body_name,
            np.asarray(pose.position, dtype=np.float64),
            np.asarray(pose.orientation, dtype=np.float64),
            world_mask=mask,
            freejoint_name=self.freejoint_name,
        )

    @staticmethod
    def _normalize_mask(
        env_mask: Optional[np.ndarray],
        nworld: int,
    ) -> Optional[np.ndarray]:
        """Validate an ``env_mask`` against the world count.

        The message matches the native handler's, which a contract test pins,
        so a caller that keys on it keeps working across backends.
        """
        if env_mask is None:
            return None
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape != (nworld,):
            raise ValueError(f"env_mask must have shape ({nworld},), got {mask.shape}")
        return mask
