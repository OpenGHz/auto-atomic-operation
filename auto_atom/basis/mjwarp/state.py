"""Device-resident scene state for the MJWarp (GPU MuJoCo) backend.

This is the read/write layer the MJWarp backend is built on: it owns the device
``Model``/``Data`` pair and answers the frame queries the runtime asks for, in
the same units, dtypes and conventions as the native MuJoCo path.

Three facts about MJWarp shape this module, all verified rather than assumed
(see ``docs/design/mjwarp-backend-design.md``):

* **The host model stays authoritative for names.** ``put_model`` requires a
  ``mujoco.MjModel`` and the device model carries no name table, so every
  name -> id lookup resolves against the retained host model. The same applies
  to ``vis``/``stat``, which the device model does not expose.
* **Quaternions keep MuJoCo's wxyz order.** ``Data.xquat`` is a ``wp.quat``,
  whose Warp-native order is xyzw, but MJWarp stores MuJoCo's wxyz in it. The
  conversion here is therefore the same wxyz -> xyzw swizzle the native path
  applies, not a no-op.
* **Every state array carries a leading world axis.** ``xpos`` is
  ``(nworld, nbody, 3)`` and ``site_xmat`` is ``(nworld, nsite, 3, 3)`` rather
  than MuJoCo's flat 9, so reads index a world first and flatten after.

``mujoco_warp`` is imported lazily so that importing this module -- and
therefore the package -- does not require the GPU dependency to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mujoco


def _require_mujoco_warp() -> Any:
    """Return the ``mujoco_warp`` module, with an actionable error if absent."""
    try:
        import mujoco_warp
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError(
            "The MJWarp backend requires the 'mujoco_warp' package "
            "(pip install mujoco-warp). It is an optional dependency: the "
            "native MuJoCo backend does not need it."
        ) from exc
    return mujoco_warp


# Model fields the randomization layer writes per environment. Batching these
# lets one shared device model express per-world variation, replacing the
# native path's one-MjModel-per-replica structure.
BATCHED_MODEL_FIELDS: Tuple[str, ...] = (
    "body_pos",
    "body_quat",
    "geom_size",
    "geom_rgba",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
    "jnt_range",
    "qpos0",
)


class MjWarpSceneState:
    """Device ``Model``/``Data`` pair plus the host model that names resolve on.

    ``nworld`` replaces the native path's replica count: one device model with
    ``nworld`` worlds, rather than ``nworld`` separate ``MjModel``/``MjData``
    pairs.

    ``njmax`` must be supplied for scenes whose constraint count exceeds the
    default budget; MJWarp reports ``nefc overflow`` on the first step
    otherwise, and the value scales with ``nworld``.
    """

    def __init__(
        self,
        host_model: "mujoco.MjModel",
        *,
        nworld: int = 1,
        njmax: Optional[int] = None,
        batched_fields: Tuple[str, ...] = BATCHED_MODEL_FIELDS,
    ) -> None:
        if nworld < 1:
            raise ValueError(f"nworld must be >= 1; got {nworld}.")
        mjw = _require_mujoco_warp()
        import mujoco

        self._mjw = mjw
        self.host_model = host_model
        self.nworld = int(nworld)

        batch_sizes = {name: self.nworld for name in batched_fields}
        self.model = mjw.put_model(host_model, batch_sizes=batch_sizes)

        # put_data seeds every world from one host MjData, so the host pass runs
        # first to give all worlds a kinematically consistent starting state.
        host_data = mujoco.MjData(host_model)
        mujoco.mj_forward(host_model, host_data)
        put_data_kwargs: dict[str, Any] = {"nworld": self.nworld}
        if njmax is not None:
            put_data_kwargs["njmax"] = int(njmax)
        self.data = mjw.put_data(host_model, host_data, **put_data_kwargs)

        self._host_scratch = host_data

    # ------------------------------------------------------------------
    # Name resolution (host model; the device model has no name table)
    # ------------------------------------------------------------------

    def _id(self, obj_type: Any, name: str, kind: str) -> int:
        import mujoco

        obj_id = mujoco.mj_name2id(self.host_model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"{kind} '{name}' not found in the MuJoCo model.")
        return int(obj_id)

    def body_id(self, body_name: str) -> int:
        import mujoco

        return self._id(mujoco.mjtObj.mjOBJ_BODY, body_name, "Body")

    def site_id(self, site_name: str) -> int:
        import mujoco

        return self._id(mujoco.mjtObj.mjOBJ_SITE, site_name, "Site")

    # ------------------------------------------------------------------
    # Conversions matching the native path exactly
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
        """Reorder a wxyz quaternion to xyzw, as the native basis does."""
        quat = np.asarray(quat, dtype=np.float32).reshape(-1)
        return quat[[1, 2, 3, 0]]

    @staticmethod
    def _rotmat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to an xyzw quaternion.

        This delegates to ``mju_mat2Quat`` rather than reimplementing the
        conversion, so a site pose read here is bit-identical to the native
        path's instead of merely close. MJWarp hands back a ``(3, 3)`` matrix
        where MuJoCo uses a flat 9, hence the reshape.
        """
        import mujoco

        quat_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_wxyz, np.asarray(rotmat, dtype=np.float64).reshape(9))
        return MjWarpSceneState._quat_wxyz_to_xyzw(quat_wxyz).astype(np.float32)

    def _check_world(self, world_index: int) -> int:
        if not 0 <= world_index < self.nworld:
            raise IndexError(
                f"world_index must be in [0, {self.nworld}); got {world_index}."
            )
        return int(world_index)

    # ------------------------------------------------------------------
    # Frame reads
    # ------------------------------------------------------------------

    def get_body_pose(
        self,
        body_name: str,
        world_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame body pose as ``(position, orientation_xyzw)``."""
        world = self._check_world(world_index)
        body = self.body_id(body_name)
        pos = np.asarray(self.data.xpos.numpy()[world][body], dtype=np.float32)
        quat_wxyz = np.asarray(self.data.xquat.numpy()[world][body], dtype=np.float32)
        return pos, self._quat_wxyz_to_xyzw(quat_wxyz)

    def get_site_pose(
        self,
        site_name: str,
        world_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame site pose as ``(position, orientation_xyzw)``."""
        world = self._check_world(world_index)
        site = self.site_id(site_name)
        pos = np.asarray(self.data.site_xpos.numpy()[world][site], dtype=np.float32)
        quat = self._rotmat_to_quat_xyzw(self.data.site_xmat.numpy()[world][site])
        return pos, quat

    # ------------------------------------------------------------------
    # Randomization-constraint reads
    # ------------------------------------------------------------------

    def camera_id(self, camera_name: str) -> int:
        import mujoco

        cam_id = mujoco.mj_name2id(
            self.host_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id < 0:
            raise KeyError(f"Camera '{camera_name}' not found in the MuJoCo model.")
        return int(cam_id)

    def get_camera_pose(
        self,
        camera_name: str,
        world_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame camera pose as ``(position, orientation_xyzw)``.

        This mirrors the native ``get_camera_model``, which converts
        ``cam_xmat`` with :func:`quaternion_from_matrix_3x3` -- a *different*
        helper from the ``mju_mat2Quat`` used for site poses. Each read matches
        the native helper it corresponds to rather than standardizing on one.
        """
        from auto_atom.utils.pose import quaternion_from_matrix_3x3

        world = self._check_world(world_index)
        cam = self.camera_id(camera_name)
        position = np.asarray(self.data.cam_xpos.numpy()[world][cam], dtype=np.float64)
        orientation = quaternion_from_matrix_3x3(
            np.asarray(
                self.data.cam_xmat.numpy()[world][cam], dtype=np.float64
            ).reshape(3, 3)
        )
        return position, orientation

    def get_camera_fovy_radians(
        self,
        camera_name: str,
        world_index: int = 0,
    ) -> float:
        """Vertical field of view in radians.

        ``cam_fovy`` is a batched model field, so it carries a world axis and
        can differ per world under camera randomization.
        """
        world = self._check_world(world_index)
        cam = self.camera_id(camera_name)
        from math import pi

        return float(self.model.cam_fovy.numpy()[world][cam]) * pi / 180.0

    def default_clip_range_m(self) -> Tuple[float, float]:
        """Model-default near/far clip planes, in metres.

        Read from the *host* model: the device model exposes ``stat`` but not
        ``vis``, and MJWarp's ``RenderContext`` fixes a single ``znear`` at
        creation with no ``zfar`` at all. Returning metres keeps callers free of
        both representations.
        """
        vis_map = self.host_model.vis.map
        extent = float(self.host_model.stat.extent)
        return float(vis_map.znear) * extent, float(vis_map.zfar) * extent

    def get_support_geometry(
        self,
        entity_name: str,
        world_index: int = 0,
    ) -> Any:
        """Conservative bounding sphere over one body's geoms.

        Matches the native implementation including its failure mode: a
        ``KeyError`` for an unknown entity, and a zero-radius sphere at the body
        origin for a body that carries no geoms. ``geom_size`` is batched, so
        the radius can differ per world under geometry randomization.
        """
        from auto_atom.contracts import SupportGeometry

        world = self._check_world(world_index)
        try:
            body = self.body_id(entity_name)
        except ValueError as exc:
            raise KeyError(
                f"Entity '{entity_name}' not found as a MuJoCo body."
            ) from exc

        geom_bodyid = self.model.geom_bodyid.numpy()
        geom_ids = [
            gid
            for gid in range(int(self.host_model.ngeom))
            if int(geom_bodyid[gid]) == body
        ]
        geom_xpos = self.data.geom_xpos.numpy()[world]
        if not geom_ids:
            return SupportGeometry(
                center=np.asarray(
                    self.data.xpos.numpy()[world][body], dtype=np.float64
                ),
                radius=0.0,
            )

        center = np.mean(
            np.asarray([geom_xpos[gid] for gid in geom_ids], dtype=np.float64),
            axis=0,
        )
        geom_size = self.model.geom_size.numpy()[world]
        radius = 0.0
        for gid in geom_ids:
            geom_center = np.asarray(geom_xpos[gid], dtype=np.float64)
            size = np.asarray(geom_size[gid], dtype=np.float64)
            radius = max(
                radius,
                float(np.linalg.norm(geom_center - center))
                + float(np.linalg.norm(size)),
            )
        return SupportGeometry(center=center, radius=radius)

    # ------------------------------------------------------------------
    # Advancing state
    # ------------------------------------------------------------------

    def forward(self) -> None:
        """Refresh derived quantities after a state write (``mj_forward``)."""
        self._mjw.forward(self.model, self.data)

    def step(self) -> None:
        """Advance physics by one timestep for every world (``mj_step``)."""
        self._mjw.step(self.model, self.data)

    def set_free_joint_pose(
        self,
        joint_name: str,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
        world_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write a free joint's pose and zero its velocity.

        ``orientation_xyzw`` follows the runtime's convention and is converted
        to MuJoCo's wxyz on the way into ``qpos``. Velocity is zeroed so a
        kinematic transport does not leave momentum behind -- matching what the
        native object handler does.
        """
        import mujoco

        joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name, "Joint")
        if int(self.host_model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"Joint '{joint_name}' is not a free joint.")

        qpos_adr = int(self.host_model.jnt_qposadr[joint])
        dof_adr = int(self.host_model.jnt_dofadr[joint])
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_xyzw = np.asarray(orientation_xyzw, dtype=np.float64).reshape(4)
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        # Device arrays are read back, edited and reassigned: warp arrays do not
        # support scatter assignment through a numpy view.
        qpos = self.data.qpos.numpy().copy()
        qvel = self.data.qvel.numpy().copy()
        worlds = (
            range(self.nworld)
            if world_mask is None
            else np.flatnonzero(np.asarray(world_mask, dtype=bool))
        )
        for world in worlds:
            qpos[world, qpos_adr : qpos_adr + 3] = pos
            qpos[world, qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
            qvel[world, dof_adr : dof_adr + 6] = 0.0
        self.data.qpos.assign(qpos)
        self.data.qvel.assign(qvel)
        self.forward()
