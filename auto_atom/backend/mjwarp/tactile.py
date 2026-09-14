"""Tactile sensing for the MJWarp backend.

The end-to-end reads in ``TactileSensorManager`` are all host-side numpy over a
single ``MjData`` -- panel grouping, PCA plane projection, force/torque
composition, wrench summation. The only device dependency is
``mj_data.sensordata`` (and ``site_xpos`` at build time), both of which
``MjWarpSceneState`` already exposes. So this layer reuses the manager verbatim
rather than reimplementing any of that arithmetic; it only bridges MJWarp's
per-world device buffers into the single-world ``MjData`` the manager reads.

The manager is inherently single-world -- it holds one ``MjData`` -- so a batched
read copies each world's ``sensordata`` into a scratch ``MjData``, then calls the
manager. The manager is built once and the scratch rebound per world: its layout
and PCA projection are static, so building per world would recompute them
``nworld`` times.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from auto_atom.basis.mjwarp.state import MjWarpSceneState


class MjWarpTactileReader:
    """Per-world tactile tensors and wrenches, via a reused host manager.

    ``enable=False`` mirrors native's "no tactile" path: the manager is never
    built and reads return zeros, so a task without tactile pays nothing for
    sensors it has not configured.
    """

    def __init__(
        self,
        state: MjWarpSceneState,
        *,
        enable: bool = True,
    ) -> None:
        self.state = state
        self.enable = bool(enable)
        self._manager: Optional[Any] = None
        if not self.enable:
            return

        from auto_atom.basis.mjc.tactile.tactile_sensor import TactileSensorManager

        # Build against the scratch MjData the state already owns. The manager's
        # setup (sensor ids, site ids, layout, PCA coords) reads only the model
        # and the scratch's site_xpos, both valid at construction.
        self._manager = TactileSensorManager(
            state.host_model, state._host_scratch, enable=True
        )

    @property
    def n_panels(self) -> int:
        return 0 if self._manager is None else int(self._manager.n_panels)

    def _bind_world(self, world: int) -> Any:
        """Copy one world's sensordata into the scratch, return the manager.

        Only sensordata is copied: the manager's runtime reads
        (get_finger_wrenches, get_tactile_tensor) touch sensordata alone; site
        coordinates were projected once at build and are cached, so they do not
        need refreshing per world.
        """
        index = self.state._check_world(world)
        scratch = self.state._host_scratch
        device = self.state.data.sensordata.numpy()
        scratch.sensordata[:] = np.asarray(
            device[index], dtype=scratch.sensordata.dtype
        )
        return self._manager

    def get_wrench_tensor(self, world: int) -> np.ndarray:
        """``(n_panels, 6)`` force+torque per panel for one world.

        Zeros of the same shape when tactile is disabled, so a caller does not
        branch on ``enable``.
        """
        if self._manager is None:
            return np.zeros((0, 6), dtype=np.float32)
        return self._bind_world(world).get_wrench_tensor()

    def get_tactile_tensor(self, world: int) -> np.ndarray:
        """``(n_panels, n_points, 6)`` per-taxel force+torque for one world."""
        if self._manager is None:
            return np.zeros((0, 0, 6), dtype=np.float32)
        return self._bind_world(world).get_tactile_tensor()
