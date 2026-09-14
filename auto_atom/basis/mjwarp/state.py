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

    # ------------------------------------------------------------------
    # Camera pose writes (randomization samples these)
    # ------------------------------------------------------------------

    def get_camera_mount_pose_batch(
        self,
        camera_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``cam_pos``/``cam_quat`` verbatim, per world.

        This is the offset the camera keeps relative to the body it is mounted
        on, which is what an object-mounted camera's randomization samples: the
        mount frame moves with the object, so a world pose would describe a
        different -- and moving -- thing.
        """
        cam = self.camera_id(camera_name)
        positions = np.asarray(self.model.cam_pos.numpy()[:, cam, :], dtype=np.float64)
        quats_wxyz = np.asarray(
            self.model.cam_quat.numpy()[:, cam, :], dtype=np.float64
        )
        return positions, quats_wxyz[:, [1, 2, 3, 0]]

    def set_camera_mount_pose(
        self,
        camera_name: str,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
        world_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write mount-frame camera extrinsics, per world."""
        cam = self.camera_id(camera_name)
        positions, quats_wxyz = self._pose_rows(position, orientation_xyzw)

        cam_pos = self.model.cam_pos.numpy().copy()
        cam_quat = self.model.cam_quat.numpy().copy()
        for world in self._worlds(world_mask):
            cam_pos[world][cam] = positions[world]
            cam_quat[world][cam] = quats_wxyz[world]
        self.model.cam_pos.assign(cam_pos)
        self.model.cam_quat.assign(cam_quat)
        self.forward()

    def set_camera_pose(
        self,
        camera_name: str,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
        world_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write a *world*-frame camera pose as parent-local extrinsics.

        Same conversion shape as :meth:`set_static_body_pose`, but anchored on
        the camera's own parent (``cam_bodyid``) rather than a body's parent, so
        a camera mounted on a moving body is placed correctly.
        """
        import mujoco

        cam = self.camera_id(camera_name)
        parent = int(self.host_model.cam_bodyid[cam])
        positions, quats_wxyz = self._pose_rows(position, orientation_xyzw)

        cam_pos = self.model.cam_pos.numpy().copy()
        cam_quat = self.model.cam_quat.numpy().copy()
        xpos = self.data.xpos.numpy()
        xmat = self.data.xmat.numpy()
        xquat = self.data.xquat.numpy()

        for world in self._worlds(world_mask):
            parent_pos = np.asarray(xpos[world][parent], dtype=np.float64)
            parent_rot = np.asarray(xmat[world][parent], dtype=np.float64).reshape(3, 3)
            cam_pos[world][cam] = parent_rot.T @ (positions[world] - parent_pos)

            parent_quat_wxyz = np.asarray(xquat[world][parent], dtype=np.float64)
            inverse_parent = np.empty(4, dtype=np.float64)
            mujoco.mju_negQuat(inverse_parent, parent_quat_wxyz)
            local_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(local_quat, inverse_parent, quats_wxyz[world])
            cam_quat[world][cam] = local_quat

        self.model.cam_pos.assign(cam_pos)
        self.model.cam_quat.assign(cam_quat)
        self.forward()

    # ------------------------------------------------------------------
    # Geom / joint frame reads (the deeper get_element_pose fallbacks)
    # ------------------------------------------------------------------

    def get_geom_pose(
        self,
        geom_name: str,
        world_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame geom pose as ``(position, orientation_xyzw)``."""
        import mujoco

        from auto_atom.utils.pose import quaternion_from_matrix_3x3

        world = self._check_world(world_index)
        geom = int(
            mujoco.mj_name2id(self.host_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        )
        if geom < 0:
            raise ValueError(f"Geom '{geom_name}' not found in the MuJoCo model.")
        position = np.asarray(
            self.data.geom_xpos.numpy()[world][geom], dtype=np.float64
        )
        orientation = quaternion_from_matrix_3x3(
            np.asarray(
                self.data.geom_xmat.numpy()[world][geom], dtype=np.float64
            ).reshape(3, 3)
        )
        return position, orientation

    def get_joint_frame_pose(
        self,
        joint_name: str,
        world_index: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Articulated joint anchor and its body's orientation.

        MuJoCo publishes the joint anchor in world coordinates after a forward
        pass, so ``xanchor`` is used rather than the parent body's static
        transform -- the latter loses both the anchor offset and the joint's
        current orientation.
        """
        import mujoco

        from auto_atom.utils.pose import quaternion_from_matrix_3x3

        world = self._check_world(world_index)
        joint = int(
            mujoco.mj_name2id(self.host_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        )
        if joint < 0:
            raise ValueError(f"Joint '{joint_name}' not found in the MuJoCo model.")
        joint_body = int(self.host_model.jnt_bodyid[joint])
        position = np.asarray(self.data.xanchor.numpy()[world][joint], dtype=np.float64)
        orientation = quaternion_from_matrix_3x3(
            np.asarray(
                self.data.xmat.numpy()[world][joint_body], dtype=np.float64
            ).reshape(3, 3)
        )
        return position, orientation

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
    # Batched frame reads (one readback covers every world)
    # ------------------------------------------------------------------

    def get_body_pose_batch(
        self,
        body_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Every world's pose for one body, as ``(nworld, 3)`` / ``(nworld, 4)``.

        This is the shape ``PoseState`` wants, and it costs one device readback
        rather than one per world -- which matters because the runtime reads
        object poses on every control tick.
        """
        body = self.body_id(body_name)
        positions = np.asarray(self.data.xpos.numpy()[:, body, :], dtype=np.float32)
        quats_wxyz = np.asarray(self.data.xquat.numpy()[:, body, :], dtype=np.float32)
        return positions, quats_wxyz[:, [1, 2, 3, 0]]

    def get_site_pose_batch(
        self,
        site_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Every world's pose for one site, in the same batched shape.

        ``mju_mat2Quat`` is scalar, so the conversion loops per world, but the
        device readback still happens once.
        """
        site = self.site_id(site_name)
        positions = np.asarray(
            self.data.site_xpos.numpy()[:, site, :], dtype=np.float32
        )
        mats = self.data.site_xmat.numpy()[:, site]
        quats = np.stack(
            [self._rotmat_to_quat_xyzw(mats[world]) for world in range(self.nworld)]
        )
        return positions, quats

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

        The pose may be one shared pose or one row per world; see
        :meth:`_pose_rows`. Per-world rows are what randomization needs, since
        it samples every environment independently.
        """
        import mujoco

        joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name, "Joint")
        if int(self.host_model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"Joint '{joint_name}' is not a free joint.")

        qpos_adr = int(self.host_model.jnt_qposadr[joint])
        dof_adr = int(self.host_model.jnt_dofadr[joint])
        positions, quats_wxyz = self._pose_rows(position, orientation_xyzw)

        # Device arrays are read back, edited and reassigned: warp arrays do not
        # support scatter assignment through a numpy view. One readback and one
        # assign cover every world, so cost does not scale with world count.
        qpos = self.data.qpos.numpy().copy()
        qvel = self.data.qvel.numpy().copy()
        for world in self._worlds(world_mask):
            qpos[world, qpos_adr : qpos_adr + 3] = positions[world]
            qpos[world, qpos_adr + 3 : qpos_adr + 7] = quats_wxyz[world]
            qvel[world, dof_adr : dof_adr + 6] = 0.0
        self.data.qpos.assign(qpos)
        self.data.qvel.assign(qvel)
        self.forward()

    def _worlds(self, world_mask: Optional[np.ndarray]) -> np.ndarray:
        """World indices a write applies to; all worlds when unmasked."""
        if world_mask is None:
            return np.arange(self.nworld)
        return np.flatnonzero(np.asarray(world_mask, dtype=bool))

    def _pose_rows(
        self,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize a pose argument to per-world rows indexed by world.

        Randomization samples every environment independently and writes the
        results as one batched pose -- ``pose.position[env_index]`` per env --
        so a write has to be able to give each world a *different* pose, not
        just broadcast one. Accepted forms:

        ``(3,)`` / ``(4,)``
            One pose, broadcast to every written world.
        ``(1, 3)`` / ``(1, 4)``
            Same, matching a ``PoseState`` of batch size 1.
        ``(nworld, 3)`` / ``(nworld, 4)``
            Row ``w`` applies to world ``w``. Indexing is by **absolute world
            index**, not by position within the mask, which is how the native
            handler indexes ``pose.position[env_index]``.

        Returns ``(positions, quats_wxyz)``, both ``(nworld, ·)``, with the
        quaternion reordered to MuJoCo's wxyz.
        """
        pos = np.asarray(position, dtype=np.float64)
        quat = np.asarray(orientation_xyzw, dtype=np.float64)
        pos = pos.reshape(1, 3) if pos.ndim == 1 else pos
        quat = quat.reshape(1, 4) if quat.ndim == 1 else quat

        if pos.shape[1:] != (3,) or quat.shape[1:] != (4,):
            raise ValueError(
                "position must be (3,) or (B, 3) and orientation (4,) or (B, 4); "
                f"got {pos.shape} and {quat.shape}."
            )
        if pos.shape[0] != quat.shape[0]:
            raise ValueError(
                "position and orientation must share a batch dimension; got "
                f"{pos.shape[0]} and {quat.shape[0]}."
            )
        if pos.shape[0] == 1:
            pos = np.repeat(pos, self.nworld, axis=0)
            quat = np.repeat(quat, self.nworld, axis=0)
        elif pos.shape[0] != self.nworld:
            raise ValueError(
                f"pose batch must be 1 or nworld ({self.nworld}); got {pos.shape[0]}."
            )
        return pos, quat[:, [3, 0, 1, 2]]

    def resolve_free_joint_id(
        self,
        body_name: str,
        freejoint_name: Optional[str] = None,
    ) -> int:
        """Return the free joint driving ``body_name``, or ``-1`` when static.

        ``freejoint_name`` is an explicit override; otherwise the body's own
        joints are inspected, so a scene owns its joint names
        (``object_free``, ``cup_joint``, ...) instead of the backend guessing a
        naming convention. Topology comes from the host model and is static.
        """
        import mujoco

        if freejoint_name:
            joint = int(
                mujoco.mj_name2id(
                    self.host_model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name
                )
            )
            if joint >= 0:
                return joint

        body = int(
            mujoco.mj_name2id(self.host_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        )
        if body < 0:
            return -1
        first = int(self.host_model.body_jntadr[body])
        for candidate in range(first, first + int(self.host_model.body_jntnum[body])):
            if int(self.host_model.jnt_type[candidate]) == int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                return candidate
        return -1

    def set_static_body_pose(
        self,
        body_name: str,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
        world_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Place a body that has no free joint, by world-frame pose.

        ``body_pos``/``body_quat`` are stored in the *parent* body's frame, and
        static scene assets are frequently nested below another body, so writing
        the requested world pose straight in would silently misplace them. The
        world -> parent-local conversion here mirrors the native object
        handler's, including its use of ``mju_negQuat``/``mju_mulQuat``, so the
        two backends land a body in the same place.

        Both fields are batched, so each world converts against *its own*
        parent pose and a masked write leaves other worlds untouched. This is
        what lets one shared device model express per-world scenery
        randomization that the native path needs one ``MjModel`` per replica
        for.

        The pose may be one shared pose or one row per world; see
        :meth:`_pose_rows`.
        """
        import mujoco

        body = self.body_id(body_name)
        parent = int(self.host_model.body_parentid[body])
        positions, quats_wxyz = self._pose_rows(position, orientation_xyzw)

        body_pos = self.model.body_pos.numpy().copy()
        body_quat = self.model.body_quat.numpy().copy()
        xpos = self.data.xpos.numpy()
        xmat = self.data.xmat.numpy()
        xquat = self.data.xquat.numpy()

        for world in self._worlds(world_mask):
            parent_pos = np.asarray(xpos[world][parent], dtype=np.float64)
            parent_mat = np.asarray(xmat[world][parent], dtype=np.float64).reshape(3, 3)
            body_pos[world][body] = parent_mat.T @ (positions[world] - parent_pos)

            parent_quat_wxyz = np.asarray(xquat[world][parent], dtype=np.float64)
            inverse_parent = np.empty(4, dtype=np.float64)
            mujoco.mju_negQuat(inverse_parent, parent_quat_wxyz)
            local_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(local_quat, inverse_parent, quats_wxyz[world])
            body_quat[world][body] = local_quat

        self.model.body_pos.assign(body_pos)
        self.model.body_quat.assign(body_quat)
        self.forward()

    def set_object_pose(
        self,
        body_name: str,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
        world_mask: Optional[np.ndarray] = None,
        freejoint_name: Optional[str] = None,
    ) -> None:
        """Place a body by world pose, via whichever mechanism it has.

        Dispatches to the free-joint or static path exactly as the native
        object handler does, so a caller does not need to know which kind of
        body it is holding.
        """
        joint = self.resolve_free_joint_id(body_name, freejoint_name)
        if joint >= 0:
            import mujoco

            joint_name = mujoco.mj_id2name(
                self.host_model, mujoco.mjtObj.mjOBJ_JOINT, joint
            )
            self.set_free_joint_pose(joint_name, position, orientation_xyzw, world_mask)
            return
        self.set_static_body_pose(body_name, position, orientation_xyzw, world_mask)
