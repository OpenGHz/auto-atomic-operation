"""MJWarp environment for object-only execution.

This is the ``get_env()`` object an MJWarp backend hands to the runtime. It
satisfies :class:`~auto_atom.contracts.EnvProtocol` and, structurally,
:class:`~auto_atom.contracts.PoseConstraintEnvProtocol`, which is what the
randomization layer needs in order to accept or reject a candidate placement.

Scene composition is deliberately unchanged: ``MjSpec`` compilation happens on
the host and ``put_model`` consumes the compiled result, so config-declared
cameras, object-mounted cameras and layered MJCF all behave exactly as they do
on the native path. The only thing that differs below this seam is where the
state lives.

Rendering is not implemented here yet. MJWarp's ``RenderContext`` fixes one
``znear`` at creation and has no ``zfar``, while the native path switches clip
ranges per output stream, so observation capture needs a context-per-clip-range
design (see ``docs/design/mjwarp-backend-design.md`` 3.3). Nothing on the
``object_only`` execution path renders -- capture is explicit -- so the split is
deliberate rather than an oversight.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.config.env_config import EnvConfig
from auto_atom.contracts import CameraModel, PoseConstraintReport, SupportGeometry
from auto_atom.randomization import RandomizationConstraintEvaluator
from auto_atom.scene_composition import load_composed_scene
from auto_atom.utils.pose import PoseState

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mujoco

logger = logging.getLogger(__name__)


class MjWarpObjectOnlyEnv:
    """Batched MJWarp environment addressed by world rather than by replica.

    ``batch_size`` is the number of MJWarp worlds. Unlike the native batched
    env, which aggregates one ``UnifiedMujocoEnv`` (and one
    ``MjModel``/``MjData`` pair) per row, this holds a single device model whose
    randomized fields are batched -- so per-environment variation is expressed
    inside one model.
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        *,
        host_model: Optional["mujoco.MjModel"] = None,
        njmax: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = EnvConfig.model_validate(kwargs)
        self.config = config

        # Compile on the host: put_model consumes a compiled MjModel, so scene
        # composition and config-declared camera creation are shared verbatim
        # with the native path.
        self.host_model = (
            host_model
            if host_model is not None
            else load_composed_scene(config.scene, cameras=config.camera_elements())
        )
        self.state = MjWarpSceneState(
            self.host_model,
            nworld=int(config.batch_size),
            njmax=njmax,
        )

        self._camera_specs = {camera.name: camera for camera in config.cameras}
        self._resolve_camera_ids()
        self._capture_pose_baselines()
        self._randomization_constraints = RandomizationConstraintEvaluator()
        self._interest_object_operations: Dict[str, str] = {}

        if config.name:
            # Registered under the config name so a task file's ``backend:``
            # builder can find the env Hydra already instantiated, which is how
            # the native env is wired too. Imported here rather than at module
            # scope because ``runtime`` pulls in the contracts and config
            # layers, and this module sits below them.
            from auto_atom.runtime import ComponentRegistry

            ComponentRegistry.register_env(config.name, self)

    # ------------------------------------------------------------------
    # EnvProtocol
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.state.nworld

    @property
    def n_substeps(self) -> int:
        """Physics steps advanced per ``update()``.

        Derived from ``sim_freq / update_freq`` as the native basis does, so
        ``dt_per_update`` agrees across backends for the same config.
        """
        if self.config.sim_freq is None or self.config.update_freq is None:
            return 1
        return int(self.config.sim_freq / self.config.update_freq)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _resolve_camera_ids(self) -> None:
        """Fail loudly for a configured camera the compiled model lacks.

        Reported with the available names, because the usual cause is a config
        naming a camera the scene does not author and whose declaration did not
        create one either.
        """
        import mujoco

        self._camera_ids: Dict[str, int] = {}
        for name in self._camera_specs:
            cam_id = mujoco.mj_name2id(
                self.host_model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            if cam_id < 0:
                available = [
                    mujoco.mj_id2name(self.host_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    for i in range(int(self.host_model.ncam))
                ]
                raise ValueError(
                    f"Camera '{name}' not found in the compiled model. "
                    f"Available cameras: {available}"
                )
            self._camera_ids[name] = cam_id

    def _capture_pose_baselines(self) -> None:
        """Record the model-level poses :meth:`reset` restores.

        ``reset_data`` only clears dynamic state; it leaves ``body_pos``,
        ``body_quat``, ``cam_pos`` and ``cam_quat`` alone. Static-object
        placement and camera randomization all write those fields, so without a
        baseline one reset's structural pose silently becomes the next reset's
        starting point. The native basis restores the same four arrays for the
        same reason.
        """
        self._baselines = {
            "body_pos": self.state.model.body_pos.numpy().copy(),
            "body_quat": self.state.model.body_quat.numpy().copy(),
            "cam_pos": self.state.model.cam_pos.numpy().copy(),
            "cam_quat": self.state.model.cam_quat.numpy().copy(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore the structural baseline, then clear dynamic state."""
        for field, baseline in self._baselines.items():
            getattr(self.state.model, field).assign(baseline)

        mjw = self.state._mjw
        if int(self.host_model.nkey) > 0:
            mjw.reset_data_keyframe(self.state.model, self.state.data, 0)
        else:
            mjw.reset_data(self.state.model, self.state.data)
        self.state.forward()
        self._randomization_constraints.reset()

    def close(self) -> None:
        """Release references; Warp owns device memory and frees it itself."""
        self._randomization_constraints.reset()

    def get_logger(self) -> logging.Logger:
        return logger

    # ------------------------------------------------------------------
    # Frame reads (batched, matching the native env's shapes)
    # ------------------------------------------------------------------

    def get_body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.state.get_body_pose_batch(body_name)

    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.state.get_site_pose_batch(site_name)

    def get_element_pose(self, name: str, env_index: int = 0) -> PoseState:
        """Pose of a named element, for one world.

        Resolution order is site -> body -> geom -> joint, matching the native
        backend. Sites come first because a task addresses semantic frames
        (``object_site``, ``rack_target_site``) that are usually sites attached
        to a body of the same or similar name; the deeper levels exist because
        ``controlled_frame`` and door/latch arcs address geoms and joints too.
        """
        import mujoco

        for obj_type, read in (
            (mujoco.mjtObj.mjOBJ_SITE, self.state.get_site_pose),
            (mujoco.mjtObj.mjOBJ_BODY, self.state.get_body_pose),
            (mujoco.mjtObj.mjOBJ_GEOM, self.state.get_geom_pose),
            (mujoco.mjtObj.mjOBJ_JOINT, self.state.get_joint_frame_pose),
        ):
            if mujoco.mj_name2id(self.host_model, obj_type, name) >= 0:
                position, orientation = read(name, env_index)
                return PoseState(position=position, orientation=orientation)

        raise KeyError(f"Element '{name}' not found as a site, body, geom or joint.")

    # ------------------------------------------------------------------
    # PoseConstraintEnvProtocol
    # ------------------------------------------------------------------

    def get_camera_model(
        self,
        camera_name: str,
        env_index: int = 0,
    ) -> CameraModel:
        """Pinhole model for one camera, with the effective metric clip range.

        The clip range is the *intersection* across every enabled image stream,
        matching the native derivation: a visibility-constrained object has to
        survive all of them, so the near plane is the largest and the far plane
        the smallest of the active ranges.
        """
        if camera_name not in self._camera_ids:
            raise KeyError(f"Camera '{camera_name}' not found in the MuJoCo model.")
        spec = self._camera_specs[camera_name]
        position, orientation = self.state.get_camera_pose(camera_name, env_index)
        default_near_m, default_far_m = self.state.default_clip_range_m()

        active_ranges: list[Tuple[float, float]] = []
        if spec.enable_color or spec.enable_mask or spec.enable_heat_map:
            active_ranges.append(
                (default_near_m, default_far_m)
                if spec.rgb_clip_range_m is None
                else tuple(float(value) for value in spec.rgb_clip_range_m)
            )
        if spec.enable_depth:
            active_ranges.append(
                (default_near_m, default_far_m)
                if spec.depth_clip_range_m is None
                else tuple(float(value) for value in spec.depth_clip_range_m)
            )
        if active_ranges:
            near_m = max(near for near, _ in active_ranges)
            far_m = min(far for _, far in active_ranges)
        else:
            near_m, far_m = default_near_m, default_far_m

        return CameraModel(
            name=camera_name,
            pose=PoseState(position=position, orientation=orientation),
            width=spec.width,
            height=spec.height,
            fovy_radians=self.state.get_camera_fovy_radians(camera_name, env_index),
            near=near_m,
            far=far_m,
        )

    def get_support_geometry(
        self,
        entity_name: str,
        env_index: int = 0,
    ) -> SupportGeometry:
        return self.state.get_support_geometry(entity_name, env_index)

    def evaluate_pose_constraints(
        self,
        candidate_poses: Mapping[str, PoseState],
        *,
        env_index: int = 0,
        constraints: Any = None,
        ancestors: Optional[Mapping[str, Set[str]]] = None,
        target_names: Optional[Set[str]] = None,
    ) -> PoseConstraintReport:
        """Accept or reject one candidate placement set.

        The arithmetic lives in the backend-neutral
        :class:`RandomizationConstraintEvaluator`; this only supplies the two
        simulator-specific reads, exactly as the native basis does. Reusing the
        evaluator is what makes ``visible_in`` / ``separated`` semantics
        identical across backends rather than reimplemented per backend.
        """
        return self._randomization_constraints.evaluate(
            candidate_poses,
            constraints,
            camera_model_of=lambda name: self.get_camera_model(name, env_index),
            support_geometry_of=lambda name: self.get_support_geometry(name, env_index),
            all_camera_names=self._visibility_witness_camera_names(),
            ancestors=ancestors,
            target_names=target_names,
        )

    @property
    def object_camera_names(self) -> frozenset[str]:
        """Cameras mounted on an object, which therefore ride it."""
        return frozenset(
            name
            for name, spec in self._camera_specs.items()
            if getattr(spec, "role", "scene") == "object"
        )

    def _visibility_witness_camera_names(self) -> Tuple[str, ...]:
        """Cameras eligible to witness an object for ``visible_in: all``.

        An object-mounted camera rides the very object it would have to witness,
        so it is excluded -- the same rule the native basis applies.
        """
        mounted = self.object_camera_names
        return tuple(name for name in self._camera_ids if name not in mounted)

    # ------------------------------------------------------------------
    # Camera pose access (randomization writes these)
    # ------------------------------------------------------------------

    def camera_names(self) -> list[str]:
        return list(self._camera_ids)

    def get_camera_pose(self, camera_name: str, env_index: int = 0) -> PoseState:
        position, orientation = self.state.get_camera_pose(camera_name, env_index)
        return PoseState(position=position, orientation=orientation)

    def set_camera_pose(
        self,
        camera_name: str,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write world-frame camera poses as parent-local extrinsics."""
        pose = pose.broadcast_to(self.batch_size)
        self.state.set_camera_pose(
            camera_name,
            np.asarray(pose.position, dtype=np.float64),
            np.asarray(pose.orientation, dtype=np.float64),
            world_mask=self._normalize_mask(env_mask),
        )

    def get_camera_mount_pose(self, camera_name: str) -> PoseState:
        """Batched mount-frame extrinsics: the offset from the mount body.

        This is what an object-mounted camera's randomization samples -- the
        mount frame travels with the object, so a world pose would name a
        different and moving thing.
        """
        position, orientation = self.state.get_camera_mount_pose_batch(camera_name)
        return PoseState(position=position, orientation=orientation)

    def set_camera_mount_pose(
        self,
        camera_name: str,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        pose = pose.broadcast_to(self.batch_size)
        self.state.set_camera_mount_pose(
            camera_name,
            np.asarray(pose.position, dtype=np.float64),
            np.asarray(pose.orientation, dtype=np.float64),
            world_mask=self._normalize_mask(env_mask),
        )

    def _normalize_mask(
        self,
        env_mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Validate a mask against the world count, as the handlers do."""
        if env_mask is None:
            return None
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape != (self.batch_size,):
            raise ValueError(
                f"env_mask must have shape ({self.batch_size},), got {mask.shape}"
            )
        return mask

    def set_interest_objects_and_operations(
        self,
        object_names: Sequence[str],
        operation_names: Sequence[str],
    ) -> None:
        """Record the task-focus objects, as the native basis validates them."""
        if len(object_names) != len(operation_names):
            raise ValueError(
                "object_names and operation_names must have the same length, got "
                f"{list(object_names)} and {list(operation_names)}"
            )
        for operation in operation_names:
            if operation not in self.config.heatmap_operations:
                raise ValueError(
                    f"Operation '{operation}' is not configured in operations: "
                    f"{self.config.heatmap_operations}"
                )
        self._interest_object_operations = dict(zip(object_names, operation_names))
