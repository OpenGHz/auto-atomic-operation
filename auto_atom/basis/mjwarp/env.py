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

Observation capture renders through :mod:`auto_atom.basis.mjwarp.render`, which
groups cameras into one ``RenderContext`` per requested clip range -- a context
fixes ``znear`` at creation and has no ``zfar``, while the native path switches
clip range per output stream (see ``docs/design/mjwarp-backend-design.md`` 3.3).
Capture is explicit: nothing on the ``object_only`` execution path renders, so a
run that never asks for an observation never builds a context.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.config.env_config import EnvConfig
from auto_atom.contracts import CameraModel, PoseConstraintReport, SupportGeometry
from auto_atom.randomization import RandomizationConstraintEvaluator
from auto_atom.scene_composition import load_composed_scene
from auto_atom.utils.pose import PoseState

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mujoco

    from auto_atom.basis.mjwarp.render import MjWarpBatchRenderer

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
        # Per world: (object, operation) for that world's active stage.
        self._interest_object_operations: List[Tuple[str, str]] = [
            ("", "") for _ in range(int(config.batch_size))
        ]
        self._renderer: Optional["MjWarpBatchRenderer"] = None
        self._mask_pairs: Optional[Dict[str, set]] = None

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
        """Resolve configured camera names against the compiled model.

        Gated on the camera sensor, matching the native basis and the rule
        ``EnvConfig.camera_elements`` documents: a camera declaration is inert
        without ``DataType.CAMERA``, so its name is neither materialized into the
        scene nor resolved against the model. Resolving unconditionally would
        reject a perfectly valid sensor-free config whose declared cameras were
        deliberately never created.

        When the sensor *is* enabled, a missing camera is an error reported with
        the available names, because the usual cause is a config naming a camera
        that neither the scene authored nor a declaration created.
        """
        import mujoco

        from auto_atom.config.env_config import DataType

        self._camera_ids: Dict[str, int] = {}
        if DataType.CAMERA not in self.config.enabled_sensors:
            return

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
    # Observation capture (ObservationEnvProtocol)
    # ------------------------------------------------------------------

    @property
    def renderer(self) -> "MjWarpBatchRenderer":
        """The batch renderer, built on first use.

        Deferred because building a ``RenderContext`` compiles kernels, and the
        ``object_only`` execution path never renders unless a caller explicitly
        captures an observation.
        """
        if self._renderer is None:
            from auto_atom.basis.mjwarp.render import (
                CameraRenderRequest,
                MjWarpBatchRenderer,
            )

            requests = [
                CameraRenderRequest(
                    name=spec.name,
                    width=int(spec.width),
                    height=int(spec.height),
                    want_color=bool(spec.enable_color),
                    # Masks and heat maps are both built from one segmentation
                    # pass, exactly as the native path does.
                    want_segmentation=bool(spec.enable_mask or spec.enable_heat_map),
                    want_depth=bool(spec.enable_depth),
                    rgb_clip_range_m=spec.rgb_clip_range_m,
                    depth_clip_range_m=spec.depth_clip_range_m,
                )
                for spec in self._camera_specs.values()
            ]
            self._renderer = MjWarpBatchRenderer(self.state, requests)
        return self._renderer

    def capture_observation(self) -> Dict[str, Dict[str, Any]]:
        """Batched camera observation in the runtime's key/payload contract.

        Every entry is ``{"data": array with a leading batch axis, "t": (B,)}``,
        which is what the native batched env produces by stacking per-replica
        captures. MJWarp renders all worlds in one pass, so the batch axis comes
        straight from the device rather than from stacking.
        """
        from auto_atom.basis.mjc.mujoco_env import KeyCreator
        from auto_atom.config.env_config import DataType

        if DataType.CAMERA not in self.config.enabled_sensors or not self._camera_specs:
            return {}

        keys = KeyCreator(self.config.structured)
        rendered = self.renderer.render()
        # data.time carries a world axis, so each world reports its own clock
        # rather than world 0's broadcast across the batch.
        timestamps = np.asarray(self.state.data.time.numpy(), dtype=np.float64)
        observation: Dict[str, Dict[str, Any]] = {}

        def emit(key: str, data: np.ndarray) -> None:
            observation[key] = {"data": data, "t": timestamps}

        for name, spec in self._camera_specs.items():
            streams = rendered.get(name, {})
            if spec.enable_color and "color" in streams:
                emit(keys.create_color_key(name), streams["color"])
            if spec.enable_depth and "depth" in streams:
                emit(keys.create_depth_key(name), streams["depth"])

            segmentation = streams.get("segmentation")
            if segmentation is None:
                continue
            if spec.enable_mask:
                emit(
                    keys.create_mask_key(name),
                    self._binary_masks(segmentation),
                )
            if spec.enable_heat_map:
                emit(
                    keys.create_heat_map_key(name),
                    self._operation_masks(segmentation),
                )

        return observation

    # ------------------------------------------------------------------
    # Mask construction (shared arithmetic with the native path)
    # ------------------------------------------------------------------

    def _mask_object_pairs(self) -> Dict[str, set]:
        """``(object_type, object_id)`` pairs identifying each mask object.

        Built once and cached: model topology is static. A configured mask object
        the model does not contain is an error rather than an empty mask, which
        is the native behaviour -- a silently empty mask channel is very hard to
        notice downstream.
        """
        if self._mask_pairs is not None:
            return self._mask_pairs

        import mujoco

        pairs: Dict[str, set] = {}
        for object_name in self.config.mask_objects:
            found: set = set()
            geom_id = mujoco.mj_name2id(
                self.host_model, mujoco.mjtObj.mjOBJ_GEOM, object_name
            )
            if geom_id >= 0:
                found.add((int(mujoco.mjtObj.mjOBJ_GEOM), int(geom_id)))
            body_id = mujoco.mj_name2id(
                self.host_model, mujoco.mjtObj.mjOBJ_BODY, object_name
            )
            if body_id >= 0:
                geom_bodyid = self.state.model.geom_bodyid.numpy()
                for candidate in range(int(self.host_model.ngeom)):
                    if int(geom_bodyid[candidate]) == body_id:
                        found.add((int(mujoco.mjtObj.mjOBJ_GEOM), int(candidate)))
            if not found:
                raise ValueError(
                    f"Mask object '{object_name}' not found as a geom or body in "
                    "the MuJoCo model."
                )
            pairs[object_name] = found

        self._mask_pairs = pairs
        return pairs

    def _pair_mask(self, segmentation: np.ndarray, pairs: set) -> np.ndarray:
        """Boolean mask over ``(B, H, W)`` for one object's geom/type pairs."""
        selected = np.zeros(segmentation.shape[:3], dtype=bool)
        for object_type, object_id in pairs:
            selected |= (segmentation[..., 0] == object_id) & (
                segmentation[..., 1] == object_type
            )
        return selected

    def _binary_masks(self, segmentation: np.ndarray) -> np.ndarray:
        """Union of every configured mask object, as ``uint8``."""
        masks = np.zeros(segmentation.shape[:3], dtype=np.uint8)
        for pairs in self._mask_object_pairs().values():
            masks[self._pair_mask(segmentation, pairs)] = 1
        return masks

    def _operation_masks(self, segmentation: np.ndarray) -> np.ndarray:
        """One channel per configured operation, marked per world.

        Channels follow ``heatmap_operations`` order so a consumer indexes them
        identically on either backend. Each world is marked from *its own*
        interest entry, because worlds run independently and may be on different
        stages at the same tick.
        """
        operations = list(self.config.heatmap_operations)
        masks = np.zeros((*segmentation.shape[:3], len(operations)), dtype=np.uint8)
        pairs_by_object = self._mask_object_pairs()

        for world, (object_name, operation) in enumerate(
            self._interest_object_operations
        ):
            if not object_name or operation not in operations:
                continue
            pairs = pairs_by_object.get(object_name)
            if not pairs:
                continue
            selected = self._pair_mask(segmentation[world : world + 1], pairs)[0]
            masks[world][selected, operations.index(operation)] = 1
        return masks

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
        """Record each world's current task focus, for the heat-map channels.

        The lists are **per world**, not a paired object/operation list: index
        ``i`` is world ``i``'s active stage object and operation, with ``""`` for
        a world that has no active stage. That is what the runtime passes (see
        ``TaskRunner`` and the native batched env), so a world's heat map marks
        the object *that world* is currently working on.

        A length-1 list broadcasts, matching the native batched env, which is how
        ``EnvConfig.interests`` supplies a single scene-wide default.
        """
        if len(object_names) != len(operation_names):
            raise ValueError(
                "object_names and operation_names must have the same length, got "
                f"{list(object_names)} and {list(operation_names)}"
            )
        if len(object_names) not in {1, self.batch_size}:
            raise ValueError(
                "Expected either broadcast length 1 or per-env length equal to "
                f"batch_size ({self.batch_size}), got {len(object_names)}."
            )

        objects = (
            list(object_names) * self.batch_size
            if len(object_names) == 1
            else list(object_names)
        )
        operations = (
            list(operation_names) * self.batch_size
            if len(operation_names) == 1
            else list(operation_names)
        )
        for operation in operations:
            if operation and operation not in self.config.heatmap_operations:
                raise ValueError(
                    f"Operation '{operation}' is not configured in operations: "
                    f"{self.config.heatmap_operations}"
                )
        self._interest_object_operations = [
            (obj, op) if obj and op else ("", "")
            for obj, op in zip(objects, operations)
        ]
