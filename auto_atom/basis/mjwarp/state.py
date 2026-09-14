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

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mujoco

logger = logging.getLogger(__name__)


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


def adapt_options_for_warp(host_model: "mujoco.MjModel") -> List[str]:
    """Neutralise solver options MJWarp rejects, in place, and report each one.

    The host model seeds ``put_data`` and backs the tactile scratch, so it must
    agree with the device model. Adaptation belongs here to preserve the solver
    options in the MJCF shared with the native backend.

    Only ``noslip_iterations`` is disabled: MJWarp lacks the post-solve pass
    that removes residual tangential slip in contacts. Other unsupported
    features still raise from ``put_model``. Returns one human-readable line
    per adjustment, empty when the model needed none.
    """
    adjustments: List[str] = []

    noslip = int(host_model.opt.noslip_iterations)
    if noslip > 0:
        host_model.opt.noslip_iterations = 0
        adjustments.append(
            f"noslip_iterations {noslip} -> 0 (MJWarp has no noslip solver; "
            "grasped objects may show tangential creep under load)"
        )

    if adjustments:
        logger.warning(
            "MJWarp does not support every solver option this scene requests; "
            "adapted %d option(s): %s",
            len(adjustments),
            "; ".join(adjustments),
        )
    return adjustments


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
    otherwise. The budget is per world; its memory cost scales with ``nworld``.
    """

    def __init__(
        self,
        host_model: "mujoco.MjModel",
        *,
        nworld: int = 1,
        njmax: Optional[int] = None,
        batched_fields: Tuple[str, ...] = BATCHED_MODEL_FIELDS,
        host_data: Optional["mujoco.MjData"] = None,
        nconmax: Optional[int] = None,
    ) -> None:
        if nworld < 1:
            raise ValueError(f"nworld must be >= 1; got {nworld}.")
        mjw = _require_mujoco_warp()
        import mujoco

        self._mjw = mjw
        self.host_model = host_model
        self.nworld = int(nworld)

        # Before put_model, which refuses a model asking for an unimplemented
        # feature. Mutates host_model, so it also covers the put_data seed below
        # and the tactile host scratch.
        self.option_adjustments = adapt_options_for_warp(host_model)

        batch_sizes = {name: self.nworld for name in batched_fields}
        self.model = mjw.put_model(host_model, batch_sizes=batch_sizes)

        # put_data seeds every world from one host MjData, so the host pass runs
        # first to give all worlds a kinematically consistent starting state.
        if host_data is None:
            host_data = mujoco.MjData(host_model)
        mujoco.mj_forward(host_model, host_data)
        put_data_kwargs: dict[str, Any] = {"nworld": self.nworld}
        if njmax is not None:
            put_data_kwargs["njmax"] = int(njmax)
        if nconmax is not None:
            put_data_kwargs["nconmax"] = int(nconmax)
        self.data = mjw.put_data(host_model, host_data, **put_data_kwargs)

        self._host_scratch = host_data
        # Reference-counted so nested deferral blocks collapse into one pass.
        self._defer_depth = 0
        self._forward_pending = False
        # Bodies whose static geoms need world transforms recomputed by hand,
        # because MJWarp's kinematics skips them (see
        # _refresh_pending_static_geoms). Drained after each *real* pass, since
        # the recomputation reads the xpos that pass produces.
        self._pending_static_geom_bodies: set[int] = set()
        # Bodies written since the last kinematics pass, so a write that reads a
        # parent's world pose can tell whether that pose is still valid.
        self._dirty_bodies: set[int] = set()
        # Step deferral is tracked separately from forward deferral: a control
        # tick coalesces steps across envs, while a reset coalesces kinematics
        # passes across entities. The two nest independently.
        self._step_defer_depth = 0
        self._step_pending = False
        self._pending_step_count = 0
        self._pending_step_mask = np.zeros(self.nworld, dtype=bool)
        self._integration_scratch = None

    def integration_state(self) -> Any:
        """Snapshot the complete integration state on the current device."""
        import mujoco
        import warp as wp

        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        snapshot = wp.empty(
            (self.nworld, mujoco.mj_stateSize(self.host_model, signature)),
            dtype=wp.float32,
            device=self.data.qpos.device,
        )
        self._mjw.get_state(self.model, self.data, snapshot, signature)
        return snapshot

    def restore_integration_state(
        self, snapshot: Any, world_mask: Optional[np.ndarray] = None
    ) -> None:
        """Restore a device snapshot, then refresh observable poses and contacts."""
        import mujoco
        import warp as wp

        mask = (
            None
            if world_mask is None
            else wp.array(world_mask, dtype=wp.bool, device=self.data.qpos.device)
        )
        self._mjw.set_state(
            self.model,
            self.data,
            snapshot,
            int(mujoco.mjtState.mjSTATE_INTEGRATION),
            mask,
        )
        self._mark_all_dirty()
        self.forward()

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

        # Converting into the mount body's frame reads its current world pose.
        # A camera mounted on a randomized object (rack_plate mounts plate_cam on
        # the plate) would otherwise convert against the plate's pre-write pose.
        self._require_fresh_parent(parent)

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

    def get_joint_angle(self, joint_name: str, world_index: int = 0) -> float:
        """Scalar ``qpos`` value of a named joint, for one world.

        Door and latch arcs address a hinge by name and read its angle to decide
        how far the effect trajectory has progressed.
        """
        import mujoco

        world = self._check_world(world_index)
        joint = int(
            mujoco.mj_name2id(self.host_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        )
        if joint < 0:
            raise KeyError(f"No joint named '{joint_name}' found in the MuJoCo model.")
        qpos_adr = int(self.host_model.jnt_qposadr[joint])
        return float(self.data.qpos.numpy()[world][qpos_adr])

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
        """Refresh derived quantities after a state write (``mj_forward``).

        Inside :meth:`deferred_forward` this only records that a refresh is due,
        so a sequence of writes costs one kinematics pass instead of one each.
        """
        if self._defer_depth > 0:
            self._forward_pending = True
            return
        self._run_forward()

    def _run_forward(self) -> None:
        """Run one real kinematics pass, then fix up what it skips.

        Every actual ``mjw.forward`` goes through here so the static-geom
        recomputation cannot be missed: it has to read the ``xpos`` the pass just
        produced, so it can only run after a real pass, never after a deferred
        no-op.
        """
        self._mjw.forward(self.model, self.data)
        self._refresh_pending_static_geoms()

    @contextmanager
    def deferred_forward(self) -> Iterator[None]:
        """Coalesce the kinematics passes of several writes into one.

        Each write method refreshes derived state on its own so that it is
        correct when called alone. A reset writes many entities in a row, and
        every intermediate refresh is then wasted work -- measured at 74% of
        reset time on the rack_plate config, where 6 passes ran for 5 entities.

        Correctness is preserved rather than assumed. A write that converts
        against a parent's *current* world pose (static bodies, world-frame
        cameras) calls :meth:`_require_fresh_parent`, which forces the pending
        pass when that parent was itself written inside this block. So deferral
        is safe even for a scene that randomizes both a body and its parent,
        instead of only for configs where the entities happen to be independent.

        Nesting is reference-counted, and the pass runs on exit even if the body
        raises, so a failed reset cannot leave stale kinematics behind.
        """
        self._defer_depth += 1
        try:
            yield
        finally:
            self._defer_depth -= 1
            if self._defer_depth == 0:
                self._dirty_bodies.clear()
                if self._forward_pending:
                    self._forward_pending = False
                    self._run_forward()

    def _mark_dirty(self, body_id: int) -> None:
        """Record that a body's world pose is stale until the next pass."""
        if self._defer_depth > 0:
            self._dirty_bodies.add(int(body_id))

    def _mark_all_dirty(self) -> None:
        """Record that every body's world pose is stale.

        A joint write moves the whole subtree below that joint, so naming one
        body would understate what went stale. Marking everything keeps the
        stale-parent guard conservative: a later write that converts against
        any parent pose flushes rather than trusting a pre-write frame.
        """
        if self._defer_depth > 0:
            self._dirty_bodies.update(range(int(self.host_model.nbody)))

    def _require_fresh_parent(self, body_id: int) -> None:
        """Flush a deferred pass when ``body_id``'s pose is needed but stale.

        Called by writes that read a frame's current world pose to convert into
        it. Without this, deferring a child's placement behind its parent's would
        convert against the parent's pre-write pose and land the child somewhere
        else entirely.
        """
        if self._defer_depth > 0 and int(body_id) in self._dirty_bodies:
            self._forward_pending = False
            self._dirty_bodies.clear()
            self._run_forward()

    def step(self, nstep: int = 1, *, world_mask: Optional[np.ndarray] = None) -> None:
        """Advance selected worlds by ``nstep`` physical timesteps.

        Inside a :meth:`deferred_step` block this only records that a step is
        wanted. Equal-duration requests are merged by world and run when the
        outermost block exits. Outside a block the result is available before
        returning, as required by synchronous controller completion checks.
        """
        if not isinstance(nstep, int) or nstep < 1:
            raise ValueError(f"nstep must be a positive integer; got {nstep}.")
        mask = (
            np.ones(self.nworld, dtype=bool)
            if world_mask is None
            else np.asarray(world_mask, dtype=bool)
        )
        if mask.shape != (self.nworld,):
            raise ValueError(f"world_mask must have shape ({self.nworld},).")
        if not mask.any():
            return
        if self._step_defer_depth > 0:
            if self._step_pending and self._pending_step_count != nstep:
                raise ValueError("Deferred step requests must have the same nstep.")
            self._step_pending = True
            self._pending_step_count = nstep
            self._pending_step_mask |= mask
            return
        self._run_steps(nstep, mask)

    def _run_steps(self, nstep: int, mask: np.ndarray) -> None:
        """Preserve inactive worlds using MJWarp's integration-state contract.

        MJWarp steps all worlds. Its get/set_state API includes time, actuator
        dynamics, warmstart, external forces and mocap inputs as well as qpos
        and qvel, so a masked call can restore the entire inactive simulation
        state. Forward then refreshes derived poses and contacts for readers.
        """
        import mujoco
        import warp as wp

        inactive = None
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if not mask.all():
            if self._integration_scratch is None:
                self._integration_scratch = wp.empty(
                    (self.nworld, mujoco.mj_stateSize(self.host_model, signature)),
                    dtype=wp.float32,
                    device=self.data.qpos.device,
                )
            inactive = wp.array(~mask, dtype=wp.bool, device=self.data.qpos.device)
            self._mjw.get_state(
                self.model, self.data, self._integration_scratch, signature, inactive
            )
        try:
            for _ in range(nstep):
                self._mjw.step(self.model, self.data)
        finally:
            if inactive is not None:
                self._mjw.set_state(
                    self.model,
                    self.data,
                    self._integration_scratch,
                    signature,
                    inactive,
                )
                self.forward()

    @contextmanager
    def deferred_step(self) -> Iterator[None]:
        """Collapse a control tick's per-env steps into one step of all worlds.

        The runtime drives control one environment at a time: ``runtime.py``
        loops over envs and ``stage_execution._mask_for_env`` hands each call a
        **one-hot** mask. The native backend can honour that literally because
        each env is its own ``MjData``, so it steps exactly that one.

        MJWarp cannot: ``mjw.step(m, d)`` takes no mask and advances every
        world. Stepping once per env would therefore advance each world
        ``batch_size`` times per control tick, and all but one of those steps
        would run under a *different* env's command. Nothing would raise -- the
        simulation would just silently run at ``batch_size x timestep`` per tick
        with mismatched commands.

        So a control tick wraps its per-env calls in this block: each call
        writes its own world's ``ctrl`` and asks to step, and exactly one step
        of all worlds happens on exit. Same shape as :meth:`deferred_forward`,
        for the same reason -- a per-entity operation coalesced into the batched
        one the device actually offers.

        Nesting is reference-counted, and the step runs on exit even if the body
        raises, so a failed control tick cannot leave a tick un-stepped.
        """
        self._step_defer_depth += 1
        try:
            yield
        finally:
            self._step_defer_depth -= 1
            if self._step_defer_depth == 0 and self._step_pending:
                self._step_pending = False
                nstep = self._pending_step_count
                mask = self._pending_step_mask.copy()
                self._pending_step_count = 0
                self._pending_step_mask[:] = False
                self._run_steps(nstep, mask)

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def get_contact_geom_pairs(self, world_index: int = 0) -> np.ndarray:
        """Colliding geom-id pairs for one world, shaped ``(ncon, 2)``.

        MJWarp has no per-world ``ncon``: every world's contacts share one flat
        pool of length ``nacon``, tagged by ``contact.worldid``. A naive
        ``range(data.ncon)`` port therefore cannot work -- the field does not
        exist -- and iterating the whole pool would attribute other worlds'
        contacts to this one.

        Verified against native MuJoCo on a two-box scene: filtering by world
        recovers exactly the native contact count and the same geom pairs, and
        ``geom_bodyid`` resolves identically on host and device. That is what the
        grasp check needs, which reads contact *existence* and geom pairing
        rather than contact force.
        """
        world = self._check_world(world_index)
        nacon = int(self.data.nacon.numpy()[0])
        if nacon == 0:
            return np.empty((0, 2), dtype=np.int32)
        selected = self.data.contact.worldid.numpy()[:nacon] == world
        return np.asarray(
            self.data.contact.geom.numpy()[:nacon][selected], dtype=np.int32
        )

    def get_contact_body_pairs(self, world_index: int = 0) -> np.ndarray:
        """Colliding body-id pairs for one world, shaped ``(ncon, 2)``.

        Bodies rather than geoms are what a grasp or contact query compares
        against, since a target is addressed as a body and its whole geom set
        counts.
        """
        pairs = self.get_contact_geom_pairs(world_index)
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        geom_bodyid = np.asarray(self.model.geom_bodyid.numpy(), dtype=np.int32)
        return geom_bodyid[pairs]

    def descendant_body_ids(self, body_name: str) -> frozenset[int]:
        """A body and every body below it in the tree.

        A grasp targets a body, but its collision geoms may hang off child
        bodies (a plate's rim, a mug's handle), so a contact against any body
        in the subtree still counts as touching the target. Resolved on the
        host model, whose ``body_parentid`` is authoritative for topology, and
        the walk relies on MuJoCo numbering children after their parents.
        """
        import mujoco

        target = mujoco.mj_name2id(self.host_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target < 0:
            raise ValueError(f"Body '{body_name}' not found in the compiled model.")
        ids = {target}
        parentid = self.host_model.body_parentid
        for body in range(int(self.host_model.nbody)):
            if body != 0 and int(parentid[body]) in ids:
                ids.add(body)
        return frozenset(ids)

    def operator_body_ids(self, root_body_name: str) -> frozenset[int]:
        """The operator root body and its whole subtree.

        Same subtree walk as :meth:`descendant_body_ids`; named separately
        because the grasp check reads it against a different argument (the
        arm root, not the grasp target) and the intent is worth keeping legible
        at the call site.
        """
        return self.descendant_body_ids(root_body_name)

    def finger_geom_sides(self, operator_body_ids: frozenset[int]) -> dict[int, str]:
        """Map each gripper finger geom id to ``"left"`` or ``"right"``.

        A grasp is confirmed only when *both* a left and a right finger geom
        touch the target, so the two sides have to be distinguishable. Mirrors
        the native name matching: a geom belonging to the operator whose name
        starts with ``left_``/``right_`` or contains ``_left_``/``_right_``
        (the latter covers a gripper attached under an ``<attach prefix=...>``).
        """
        import mujoco

        sides: dict[int, str] = {}
        geom_bodyid = self.host_model.geom_bodyid
        for geom in range(int(self.host_model.ngeom)):
            if int(geom_bodyid[geom]) not in operator_body_ids:
                continue
            name = (
                mujoco.mj_id2name(self.host_model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            )
            if name.startswith("left_") or "_left_" in name:
                sides[geom] = "left"
            elif name.startswith("right_") or "_right_" in name:
                sides[geom] = "right"
        return sides

    def finger_contacts_with_target(
        self,
        world_index: int,
        target_body_ids: frozenset[int],
        finger_sides: dict[int, str],
    ) -> Tuple[bool, bool]:
        """Whether a left and a right finger geom each touch the target.

        Returns ``(left_contact, right_contact)`` for one world. This is the
        contact half of the grasp check; the lateral-distance half needs the
        eef pose and belongs to the operator handler. Built on the world-
        filtered geom-pair scan, so it reads only this world's contacts.
        """
        pairs = self.get_contact_geom_pairs(world_index)
        if pairs.size == 0:
            return False, False
        geom_bodyid = np.asarray(self.model.geom_bodyid.numpy(), dtype=np.int32)
        left = right = False
        for geom1, geom2 in pairs:
            body1 = int(geom_bodyid[geom1])
            body2 = int(geom_bodyid[geom2])
            b1 = body1 in target_body_ids
            b2 = body2 in target_body_ids
            if not b1 and not b2:
                continue
            other = int(geom2) if b1 else int(geom1)
            side = finger_sides.get(other)
            if side == "left":
                left = True
            elif side == "right":
                right = True
            if left and right:
                break
        return left, right

    # ------------------------------------------------------------------
    # Actuators and actuated joints
    # ------------------------------------------------------------------

    def actuator_ids(self, actuator_names: Sequence[str]) -> np.ndarray:
        """Resolve actuator names to ids, erroring with what *is* available.

        Mirrors the native ``_resolve_actuator_indices``: a missing actuator is
        a config error worth naming the alternatives for, not a silent skip
        that would leave a limb unactuated.
        """
        import mujoco

        ids = []
        for name in actuator_names:
            actuator = mujoco.mj_name2id(
                self.host_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if actuator < 0:
                available = [
                    mujoco.mj_id2name(self.host_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    for i in range(int(self.host_model.nu))
                ]
                raise ValueError(
                    f"Actuator '{name}' not found in the compiled model. "
                    f"Available actuators: {available}"
                )
            ids.append(actuator)
        return np.asarray(ids, dtype=np.int32)

    def actuator_joint_indices(
        self,
        actuator_ids: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(qpos_indices, dof_indices)`` for the joints those actuators drive.

        Actuators that drive no joint (``trnid < 0``, e.g. a tendon or general
        transmission) are skipped rather than reported as index ``-1``, which
        is what the native path does -- so the returned arrays may be shorter
        than ``actuator_ids``.
        """
        qpos_indices = []
        dof_indices = []
        for actuator in actuator_ids:
            joint = int(self.host_model.actuator_trnid[int(actuator), 0])
            if joint < 0:
                continue
            qpos_indices.append(int(self.host_model.jnt_qposadr[joint]))
            dof_indices.append(int(self.host_model.jnt_dofadr[joint]))
        return (
            np.asarray(qpos_indices, dtype=np.int32),
            np.asarray(dof_indices, dtype=np.int32),
        )

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def sensor_id(self, sensor_name: str) -> int:
        """Resolve a sensor name to its id on the host model."""
        import mujoco

        return self._id(mujoco.mjtObj.mjOBJ_SENSOR, sensor_name, "Sensor")

    def get_sensor_values(self, sensor_name: str) -> np.ndarray:
        """One sensor's readings for every world, shaped ``(nworld, dim)``.

        ``sensordata`` is a flat per-world buffer addressed by ``sensor_adr`` and
        ``sensor_dim`` on the host model, exactly as on the native path -- MJWarp
        gives it a leading world axis and otherwise leaves the layout alone.
        Verified against native on a touch sensor: 0.6278352 per world against
        native's 0.62783464, i.e. float32 agreement.

        This is what the tactile layer needs. Everything above the read (panel
        grouping, PCA projection, wrench summation) is host-side numpy in
        ``basis/mjc/tactile``, so it ports as a consumer of this rather than
        needing a device reimplementation.
        """
        sensor = self.sensor_id(sensor_name)
        address = int(self.host_model.sensor_adr[sensor])
        dim = int(self.host_model.sensor_dim[sensor])
        values = self.data.sensordata.numpy()[:, address : address + dim]
        return np.asarray(values, dtype=np.float64)

    def get_sensor_values_batch(
        self,
        sensor_names: Sequence[str],
    ) -> np.ndarray:
        """Several sensors at once, ``(nworld, len(sensor_names), dim)``.

        Costs one device readback rather than one per sensor, which matters
        because a tactile array is tens of panels read every tick. Requires the
        sensors to share a dimension, which panels of one array do; a mixed
        request is a caller error rather than something to pad silently.
        """
        if not sensor_names:
            return np.empty((self.nworld, 0, 0), dtype=np.float64)

        spans = []
        for name in sensor_names:
            sensor = self.sensor_id(name)
            spans.append(
                (
                    int(self.host_model.sensor_adr[sensor]),
                    int(self.host_model.sensor_dim[sensor]),
                )
            )
        dims = {dim for _, dim in spans}
        if len(dims) != 1:
            raise ValueError(
                "get_sensor_values_batch requires sensors of equal dimension; "
                f"got {sorted(dims)} for {list(sensor_names)}."
            )

        buffer = self.data.sensordata.numpy()
        dim = spans[0][1]
        out = np.empty((self.nworld, len(spans), dim), dtype=np.float64)
        for index, (address, _) in enumerate(spans):
            out[:, index, :] = buffer[:, address : address + dim]
        return out

    def actuator_joint_names(self, actuator_ids: Sequence[int]) -> list[str]:
        """Names of the joints those actuators drive, in actuator order.

        An IK solver is constructed as
        ``ik_factory(model=model, arm_joint_names=names, **ik_params)``, so the
        solver's joint ordering is defined by the arm actuator ordering in the
        config. Order therefore matters: returning these sorted, or by joint id,
        would silently permute the solver's joint mapping.

        A jointless transmission yields ``"<joint_-1>"`` rather than being
        skipped, mirroring the native helper -- dropping it would shorten the
        list and misalign every name after it, which is worse than a name that
        is visibly wrong.
        """
        import mujoco

        names = []
        for actuator in actuator_ids:
            joint = int(self.host_model.actuator_trnid[int(actuator), 0])
            names.append(
                mujoco.mj_id2name(self.host_model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                or f"<joint_{joint}>"
            )
        return names

    def get_ctrl(self) -> np.ndarray:
        """Every world's actuator command, shaped ``(nworld, nu)``.

        Returned as float64 even though the device stores float32, so callers
        compare against the same dtype the native path hands them.
        """
        return np.asarray(self.data.ctrl.numpy(), dtype=np.float64)

    def set_ctrl(
        self,
        actuator_ids: Sequence[int],
        values: np.ndarray,
        world_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write actuator commands for selected actuators and worlds.

        ``values`` is one row broadcast to every written world, or one row per
        world indexed by **absolute world index** -- the same convention as
        :meth:`_pose_rows`, since a batched controller sends each environment a
        different command.

        Out-of-range commands are deliberately *not* clamped here: MuJoCo
        clamps against ``actuator_ctrlrange`` inside the step and leaves
        ``data.ctrl`` untouched, and MJWarp was verified to match that (same
        settled ``qpos`` for an out-of-range command, ``ctrl`` unchanged on
        readback). Clamping on write would make the two paths disagree about
        what was commanded.
        """
        ids = np.asarray(actuator_ids, dtype=np.int32)
        rows = self._actuator_rows(values, ids.size)
        ctrl = self.data.ctrl.numpy().copy()
        for world in self._worlds(world_mask):
            ctrl[world, ids] = rows[world]
        self.data.ctrl.assign(ctrl)

    def _actuator_rows(self, values: np.ndarray, width: int) -> np.ndarray:
        """Normalize an actuator-command argument to one row per world."""
        array = np.asarray(values, dtype=np.float64)
        array = array.reshape(1, -1) if array.ndim == 1 else array
        if array.shape[1] != width:
            raise ValueError(
                f"Expected {width} value(s) per world; got {array.shape[1]}."
            )
        if array.shape[0] == 1:
            return np.repeat(array, self.nworld, axis=0)
        if array.shape[0] != self.nworld:
            raise ValueError(
                f"Value batch must be 1 or nworld ({self.nworld}); "
                f"got {array.shape[0]}."
            )
        return array

    def get_joint_positions(self, qpos_indices: Sequence[int]) -> np.ndarray:
        """``qpos`` at the given addresses for every world, ``(nworld, n)``."""
        indices = np.asarray(qpos_indices, dtype=np.int32)
        if indices.size == 0:
            return np.empty((self.nworld, 0), dtype=np.float64)
        return np.asarray(self.data.qpos.numpy()[:, indices], dtype=np.float64)

    def get_joint_velocities(self, dof_indices: Sequence[int]) -> np.ndarray:
        """``qvel`` at the given addresses for every world, ``(nworld, n)``."""
        indices = np.asarray(dof_indices, dtype=np.int32)
        if indices.size == 0:
            return np.empty((self.nworld, 0), dtype=np.float64)
        return np.asarray(self.data.qvel.numpy()[:, indices], dtype=np.float64)

    def set_joint_positions(
        self,
        qpos_indices: Sequence[int],
        positions: np.ndarray,
        dof_indices: Optional[Sequence[int]] = None,
        actuator_ids: Optional[Sequence[int]] = None,
        world_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Pin joints to exact angles, the kinematic counterpart of stepping.

        Follows the native kinematic branch: write ``qpos``, zero the matching
        ``qvel`` so no momentum survives the teleport, and mirror the same
        values into ``ctrl`` so switching back to physics does not snap the
        limb toward a stale command.

        The native path additionally runs a settle loop when the model carries
        equality constraints (parallel-linkage grippers), re-pinning the
        actuated joints each step so only passive joints drift. That belongs to
        the home-pose entry point rather than here, because it needs to know
        which joints are the actuated ones to re-pin.
        """
        indices = np.asarray(qpos_indices, dtype=np.int32)
        rows = self._actuator_rows(positions, indices.size)
        worlds = self._worlds(world_mask)

        qpos = self.data.qpos.numpy().copy()
        for world in worlds:
            qpos[world, indices] = rows[world]
        self.data.qpos.assign(qpos)

        if dof_indices is not None:
            dofs = np.asarray(dof_indices, dtype=np.int32)
            if dofs.size:
                qvel = self.data.qvel.numpy().copy()
                for world in worlds:
                    qvel[world, dofs] = 0.0
                self.data.qvel.assign(qvel)

        if actuator_ids is not None:
            ids = np.asarray(actuator_ids, dtype=np.int32)
            if ids.size:
                # Only as many commands as there are actuators to receive them:
                # the native path writes action[:n] into ctrl[all_aidx[:n]].
                width = min(ids.size, indices.size)
                self.set_ctrl(ids[:width], rows[:, :width], world_mask)

        self._mark_all_dirty()
        self.forward()

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
        self._mark_dirty(int(self.host_model.jnt_bodyid[joint]))
        self.forward()

    def get_mocap_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read one mocap body's targets in world coordinates, per world."""
        mocap_id = int(self.host_model.body_mocapid[self.body_id(body_name)])
        if mocap_id < 0:
            raise ValueError(f"Body '{body_name}' is not a mocap body.")
        positions = self.data.mocap_pos.numpy()[:, mocap_id].copy()
        orientations = self.data.mocap_quat.numpy()[:, mocap_id][:, [1, 2, 3, 0]].copy()
        return positions, orientations

    def set_mocap_pose(
        self,
        body_name: str,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
        *,
        world_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Command a mocap target while its welded body continues under physics."""
        mocap_id = int(self.host_model.body_mocapid[self.body_id(body_name)])
        if mocap_id < 0:
            raise ValueError(f"Body '{body_name}' is not a mocap body.")
        worlds = self._worlds(world_mask)
        positions, orientations = self._pose_rows(position, orientation_xyzw)
        target_positions = self.data.mocap_pos.numpy()
        target_orientations = self.data.mocap_quat.numpy()
        target_positions[worlds, mocap_id] = positions[worlds]
        target_orientations[worlds, mocap_id] = orientations[worlds]
        self.data.mocap_pos.assign(target_positions)
        self.data.mocap_quat.assign(target_orientations)

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

        # The conversion below reads the parent's current world pose, so a
        # pending kinematics pass has to land first when the parent itself was
        # just moved.
        self._require_fresh_parent(parent)

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
        self._mark_dirty(body)
        self._pending_static_geom_bodies.add(int(body))
        self.forward()

    def _refresh_pending_static_geoms(self) -> None:
        """Recompute geom world transforms MJWarp's kinematics deliberately skips.

        ``mujoco_warp``'s ``_geom_local_to_global`` kernel opens with an early
        return (``smooth.py:197``) for any geom whose body is welded to world
        (``body_weldid == 0``) and is not mocap-descended: for those, the comment
        states ``geom_xpos``/``geom_xmat`` are computed only once during
        ``make_data``. Under stock MuJoCo that assumption holds, because nothing
        mutates ``body_pos`` at runtime. MJWarp itself breaks it by making
        ``body_pos`` a batch-writable model field, which is exactly what
        randomization writes -- so after a static-body write the body frame moves
        while its collision and render geometry stay behind.

        This applies the same formula that kernel would have applied (its lines
        204-205) to the geoms it skipped, so nothing here is reverse-engineered:

            geom_xpos = xpos + rot(geom_pos, xquat)
            geom_xmat = mat(xquat * geom_quat)

        Only the written body's own geoms are refreshed. Its descendants cannot
        be static in this sense -- a body welded to world has no joint between
        it and world, so any child that moved with it is welded too and is
        covered by the same walk below.
        """
        if not self._pending_static_geom_bodies:
            return

        import mujoco

        weldid = self.host_model.body_weldid
        rootid = self.host_model.body_rootid
        mocapid = self.host_model.body_mocapid

        subtree: set[int] = set()
        for body_id in self._pending_static_geom_bodies:
            subtree |= self.descendant_body_ids(
                mujoco.mj_id2name(
                    self.host_model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)
                )
            )
        self._pending_static_geom_bodies.clear()

        skipped = [
            body
            for body in subtree
            if int(weldid[body]) == 0 and int(mocapid[int(rootid[body])]) == -1
        ]
        if not skipped:
            return

        geom_bodyid = np.asarray(self.host_model.geom_bodyid)
        geoms = np.flatnonzero(np.isin(geom_bodyid, skipped))
        if geoms.size == 0:
            return

        xpos = self.data.xpos.numpy()
        xquat = self.data.xquat.numpy()
        geom_pos = self.model.geom_pos.numpy()
        geom_quat = self.model.geom_quat.numpy()
        geom_xpos = self.data.geom_xpos.numpy().copy()
        geom_xmat = self.data.geom_xmat.numpy().copy()

        for world in range(self.nworld):
            for geom in geoms:
                body = int(geom_bodyid[geom])
                body_quat_wxyz = np.asarray(xquat[world][body], dtype=np.float64)
                # geom_pos/geom_quat may carry a leading batch axis of 1 or
                # nworld; the kernel indexes them modulo their own length.
                local_pos = np.asarray(
                    geom_pos[world % geom_pos.shape[0]][geom], dtype=np.float64
                )
                local_quat = np.asarray(
                    geom_quat[world % geom_quat.shape[0]][geom], dtype=np.float64
                )

                rotated = np.empty(3, dtype=np.float64)
                mujoco.mju_rotVecQuat(rotated, local_pos, body_quat_wxyz)
                geom_xpos[world][geom] = (
                    np.asarray(xpos[world][body], dtype=np.float64) + rotated
                )

                composed = np.empty(4, dtype=np.float64)
                mujoco.mju_mulQuat(composed, body_quat_wxyz, local_quat)
                matrix = np.empty(9, dtype=np.float64)
                mujoco.mju_quat2Mat(matrix, composed)
                geom_xmat[world][geom] = matrix.reshape(geom_xmat[world][geom].shape)

        self.data.geom_xpos.assign(geom_xpos)
        self.data.geom_xmat.assign(geom_xmat)

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
