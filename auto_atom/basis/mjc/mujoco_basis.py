"""Low-level MuJoCo environment wrapper.

``MujocoBasis`` owns the MuJoCo model/data lifecycle, operator index
resolution, sensor queries, rendering, viewer management and physics
stepping.  It deliberately does **not** provide ``step(action)`` or
``capture_observation()`` — those higher-level concepts live in the
``UnifiedMujocoEnv`` subclass defined in ``mujoco_env.py``.
"""

import copy
import logging
import threading
import time
from contextlib import contextmanager
from math import pi, tan
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

import mujoco
import numpy as np
from pydantic import PositiveFloat

from auto_atom.basis.mjc.model_initialization import apply_initial_joint_positions
from auto_atom.basis.mjc.tactile.tactile_sensor import TactileSensorManager
from auto_atom.config.env_config import DataType, EnvConfig
from auto_atom.contracts import (
    CameraModel,
    RandomizationConstraintReport,
    SupportGeometry,
)
from auto_atom.scene_composition import (
    SceneArtifact,
    SceneConfig,
    compile_scene,
    load_composed_scene,
)
from auto_atom.utils.pose import (
    PoseState,
    quaternion_from_matrix_3x3,
    quaternion_to_rotation_matrix,
)


class MujocoBasis:
    """Low-level MuJoCo wrapper: model/data access, rendering, physics."""

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        *,
        scene_artifact: SceneArtifact | None = None,
        **kwargs,
    ):
        self.get_logger().info("Initializing...")
        if config is None:
            config = EnvConfig.model_validate(kwargs)
        self.config = config
        self._info = None
        self.scene_artifact = scene_artifact or (
            compile_scene(config.scene) if config.scene.layers else None
        )
        self.model, self.data = self._load_model(config.scene, self.scene_artifact)
        if config.sim_freq is not None:
            self.model.opt.timestep = 1.0 / config.sim_freq

        # MuJoCo's body/camera pose arrays are model (not data) state.  Task
        # randomization and initial-pose overrides legitimately mutate these
        # arrays, but ``mj_resetData(Keyframe)`` does not restore them.  Keep
        # an immutable snapshot of the composed XML model so every reset starts
        # from the same structural scene before higher layers re-apply their
        # configured overrides.
        self._model_body_pos_baseline = np.asarray(
            self.model.body_pos, dtype=np.float64
        ).copy()
        self._model_body_quat_baseline = np.asarray(
            self.model.body_quat, dtype=np.float64
        ).copy()
        self._model_cam_pos_baseline = np.asarray(
            self.model.cam_pos, dtype=np.float64
        ).copy()
        self._model_cam_quat_baseline = np.asarray(
            self.model.cam_quat, dtype=np.float64
        ).copy()
        self.get_logger().info(
            f"Using timestep of {self.model.opt.timestep:.6f} seconds"
        )
        self._n_substeps = (
            int(config.sim_freq / config.update_freq)
            if config.sim_freq is not None and config.update_freq is not None
            else 1
        )
        self._ctrl_interp = config.ctrl_interpolation and self._n_substeps > 1
        self._prev_ctrl: np.ndarray | None = None

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._sync_mocap_to_freejoint()

        # Bind pre-step callbacks (already instantiated by Hydra).
        self._pre_step_callbacks: list[Callable] = []
        for cb in config.pre_step_callbacks:
            cb = copy.deepcopy(cb)
            if hasattr(cb, "bind"):
                cb.bind(self.model, self.data)
            self._pre_step_callbacks.append(cb)

        # Pre-compute per-operator actuator and joint index arrays.
        self._operators = config.operators
        self._camera_hidden_geom_ids = (
            self._resolve_operator_render_geom_ids()
            if config.hide_operators_in_camera
            else frozenset()
        )
        if config.hide_operators_in_camera:
            self.get_logger().info(
                "Hiding %d operator geoms from native camera rendering.",
                len(self._camera_hidden_geom_ids),
            )
        self._op_arm_aidx: dict[str, np.ndarray] = {}
        self._op_eef_aidx: dict[str, np.ndarray] = {}
        self._op_arm_qidx: dict[str, np.ndarray] = {}
        self._op_eef_qidx: dict[str, np.ndarray] = {}
        self._op_arm_vidx: dict[str, np.ndarray] = {}
        self._op_eef_vidx: dict[str, np.ndarray] = {}
        self._op_output_names: dict[str, tuple[str, str]] = {}
        self._op_eef_mapper: dict[str, Any] = {}
        for op in self._operators.values():
            arm_aidx = self._resolve_actuator_indices(op.arm_actuators)
            eef_aidx = self._resolve_actuator_indices(op.eef_actuators)
            arm_qidx, arm_vidx = self._actuator_joint_indices(arm_aidx.tolist())
            eef_qidx, eef_vidx = self._actuator_joint_indices(eef_aidx.tolist())
            self._op_arm_aidx[op.name] = arm_aidx
            self._op_eef_aidx[op.name] = eef_aidx
            self._op_arm_qidx[op.name] = arm_qidx
            self._op_eef_qidx[op.name] = eef_qidx
            self._op_arm_vidx[op.name] = arm_vidx
            self._op_eef_vidx[op.name] = eef_vidx
            self._op_output_names[op.name] = (
                op.arm_output_name or op.name,
                op.eef_output_name,
            )
            mapper = copy.deepcopy(op.eef_mapper) if op.eef_mapper is not None else None
            if mapper is not None and hasattr(mapper, "bind"):
                mapper.bind(self.model, self.data)
            self._op_eef_mapper[op.name] = mapper

        self._camera_specs = {c.name: c for c in config.cameras}
        self._renderers: Dict[str, mujoco.Renderer] = {}
        self._camera_ids = {}
        self._renderer_scene_option = mujoco.MjvOption()
        self._renderer_scene_option.sitegroup[:] = 0
        self._interest_object_operations: dict[str, str] = {}
        self._mask_object_pairs = self._build_mask_object_pairs(config.mask_objects)
        self.set_interest_objects_and_operations(*config.interests)

        logger = self.get_logger()

        if DataType.CAMERA in config.enabled_sensors:
            logger.info(f"Setting up cameras: {list(self._camera_specs.keys())}")
            max_w = max((s.width for s in config.cameras), default=0)
            max_h = max((s.height for s in config.cameras), default=0)
            if max_w > self.model.vis.global_.offwidth:
                self.model.vis.global_.offwidth = max_w
            if max_h > self.model.vis.global_.offheight:
                self.model.vis.global_.offheight = max_h
            for name, spec in self._camera_specs.items():
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if cam_id < 0:
                    raise ValueError(
                        f"Camera '{name}' not found in the Mujoco model. Available cameras: {[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]}"
                    )
                self._camera_ids[name] = cam_id
                if spec.has_native_output:
                    self._renderers[name] = mujoco.Renderer(
                        self.model,
                        height=spec.height,
                        width=spec.width,
                    )

        # _camera_parent_frame: cam_name -> ("site"|"body", obj_id, frame_name)
        self._camera_parent_frame: dict[str, tuple[str, int, str]] = {}
        # _camera_frame_site_ids: cam_name -> site_id when same-named site exists
        self._camera_frame_site_ids: dict[str, int] = {}
        for name, cam_id in self._camera_ids.items():
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                self._camera_frame_site_ids[name] = site_id
        for name, spec in self._camera_specs.items():
            if spec.parent_frame:
                entry = self._resolve_frame(spec.parent_frame)
                if entry is None:
                    raise ValueError(
                        f"Camera '{name}': parent_frame '{spec.parent_frame}' not found as site or body."
                    )
                self._camera_parent_frame[name] = entry

        # Per-operator sensor IDs resolved from config (-1 when not specified).
        self._imu_ids: dict[str, dict[str, int]] = {}
        self._pose_ids: dict[str, dict[str, int]] = {}
        self._pose_site_ids: dict[str, int] = {}
        self._pose_validated_components: set[str] = set()
        self._wrench_ids: dict[str, dict[str, int]] = {}
        for op in self._operators.values():
            self._imu_ids[op.name] = {
                "acc": self._sensor_id(op.imu_acc) if op.imu_acc else -1,
                "gyro": self._sensor_id(op.imu_gyro) if op.imu_gyro else -1,
                "quat": self._sensor_id(op.imu_quat) if op.imu_quat else -1,
            }
            self._pose_ids[op.name] = {
                "pos": self._sensor_id(op.pose_sensor_pos)
                if op.pose_sensor_pos
                else -1,
                "quat": self._sensor_id(op.pose_sensor_quat)
                if op.pose_sensor_quat
                else -1,
            }
            if op.pose_site:
                site_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, op.pose_site
                )
                if site_id < 0:
                    raise ValueError(
                        f"Operator '{op.name}': pose_site '{op.pose_site}' not found in the model."
                    )
                self._pose_site_ids[op.name] = int(site_id)
            else:
                self._pose_site_ids[op.name] = -1
            self._wrench_ids[op.name] = {
                "force": self._sensor_id(op.wrench_force) if op.wrench_force else -1,
                "torque": self._sensor_id(op.wrench_torque) if op.wrench_torque else -1,
            }

        # Auto-detect camera parent frames for cameras attached to non-world bodies.
        for name, cam_id in self._camera_ids.items():
            if name in self._camera_parent_frame:
                continue  # explicit parent_frame already set
            body_id = int(self.model.cam_bodyid[cam_id])
            if body_id != 0:  # 0 is world body → world frame, no transform needed
                body_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                entry = self._resolve_frame(body_name)
                if entry is not None:
                    self._camera_parent_frame[name] = entry
                    logger.info(
                        f"Camera '{name}' auto-detected parent frame: {entry[0]} '{entry[2]}'"
                    )

        self._apply_camera_calibrations()

        self._tactile_manager = None
        if (
            DataType.TACTILE in config.enabled_sensors
            or DataType.WRENCH in config.enabled_sensors
        ):
            self._init_tactile_manager()
        self._last_time = None

        # Apply initial_joint_positions now that operator indices (needed by
        # the constraint-settle loop inside reset()) are bound. Without this,
        # the env's observable state stays at qpos0 until the caller invokes
        # reset() explicitly, so freejoint-based home poses wouldn't be seen.
        # Do not dispatch a virtual ``reset`` from the base constructor:
        # Gaussian-rendering subclasses initialise their reset-only fields
        # after ``super().__init__`` returns.  The core reset is deliberately
        # separate from the post-reset hook so no subclass code runs before
        # subclass construction has completed.
        MujocoBasis._reset_core(self)

        self._viewer_update_defer_depth = 0
        self._viewer_update_pending = False
        self._viewer = None
        self._viewer_thread: threading.Thread | None = None
        if config.viewer is not None:
            self._launch_viewer()

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def _launch_viewer(self) -> None:
        import mujoco.viewer as _mj_viewer

        existing_threads = set(threading.enumerate())
        self._viewer = _mj_viewer.launch_passive(self.model, self.data)
        new_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in existing_threads
            and (
                "_launch_internal" in thread.name
                or getattr(getattr(thread, "_target", None), "__name__", "")
                == "_launch_internal"
            )
        ]
        self._viewer_thread = new_threads[-1] if new_threads else None
        cfg = self.config.viewer
        if cfg.lookat is not None:
            self._viewer.cam.lookat[:] = cfg.lookat
        if cfg.distance is not None:
            self._viewer.cam.distance = cfg.distance
        if cfg.azimuth is not None:
            self._viewer.cam.azimuth = cfg.azimuth
        if cfg.elevation is not None:
            self._viewer.cam.elevation = cfg.elevation
        self._sync_viewer()

    def _viewer_running(self) -> bool:
        if self._viewer is None:
            return False
        is_running = getattr(self._viewer, "is_running", None)
        if callable(is_running):
            try:
                return bool(is_running())
            except Exception:
                return False
        return True

    def _sync_viewer(self) -> bool:
        if self._viewer_update_defer_depth > 0:
            self._viewer_update_pending = True
            return False
        try:
            self._viewer.sync()
            return True
        except Exception:
            return False

    @contextmanager
    def defer_viewer_updates(self) -> Iterator[None]:
        """Coalesce viewer refreshes and skip per-update delays in this scope."""
        self._viewer_update_defer_depth += 1
        try:
            yield
        finally:
            self._viewer_update_defer_depth -= 1
            if self._viewer_update_defer_depth == 0 and self._viewer_update_pending:
                self._viewer_update_pending = False
                if self._viewer_running():
                    self._sync_viewer()

    def _shutdown_viewer(self) -> None:
        viewer = self._viewer
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if not self._viewer_running():
                break
            time.sleep(0.01)
        viewer_thread = self._viewer_thread
        if (
            viewer_thread is not None
            and viewer_thread is not threading.current_thread()
            and viewer_thread.is_alive()
        ):
            viewer_thread.join(timeout=2.0)
        self._viewer = None
        self._viewer_thread = None

    def refresh_viewer(self) -> None:
        """Redraw the passive viewer without advancing physics."""
        if self._viewer_running():
            self._sync_viewer()

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _build_mask_object_pairs(
        self, object_names: List[str]
    ) -> dict[str, set[tuple[int, int]]]:
        object_pairs: dict[str, set[tuple[int, int]]] = {}
        for object_name in object_names:
            pairs: set[tuple[int, int]] = set()

            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, object_name
            )
            if geom_id >= 0:
                pairs.add((int(mujoco.mjtObj.mjOBJ_GEOM), int(geom_id)))

            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, object_name
            )
            if body_id >= 0:
                for geom_idx in range(self.model.ngeom):
                    if int(self.model.geom_bodyid[geom_idx]) == body_id:
                        pairs.add((int(mujoco.mjtObj.mjOBJ_GEOM), int(geom_idx)))

            if not pairs:
                raise ValueError(
                    f"Mask object '{object_name}' not found as a geom or body in the Mujoco model."
                )
            object_pairs[object_name] = pairs

        return object_pairs

    def set_interest_objects_and_operations(
        self, object_names: List[str], operation_names: List[str]
    ) -> None:
        if len(object_names) != len(operation_names):
            raise ValueError(
                f"object_names and operation_names must have the same length, got {object_names=} and {operation_names=}"
            )

        interest_object_operations: dict[str, str] = {}
        for object_name, operation_name in zip(object_names, operation_names):
            if self.config.mask_objects and object_name not in self._mask_object_pairs:
                raise ValueError(
                    f"Object '{object_name}' is not configured in mask_objects: {self.config.mask_objects}"
                )
            if operation_name not in self.config.heatmap_operations:
                raise ValueError(
                    f"Operation '{operation_name}' is not configured in operations: {self.config.heatmap_operations}"
                )
            interest_object_operations[object_name] = operation_name

        self._interest_object_operations = interest_object_operations

    def _build_operation_mask(self, segmentation: np.ndarray) -> np.ndarray:
        operation_mask = np.zeros(
            (*segmentation.shape[:2], len(self.config.heatmap_operations)),
            dtype=np.uint8,
        )
        if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
            return operation_mask

        for object_name, operation_name in self._interest_object_operations.items():
            channel_idx = self.config.heatmap_operations.index(operation_name)
            pair_mask = np.zeros(segmentation.shape[:2], dtype=bool)
            for objtype, objid in self._mask_object_pairs.get(object_name, set()):
                pair_mask |= (segmentation[..., 0] == objid) & (
                    segmentation[..., 1] == objtype
                )
            operation_mask[pair_mask, channel_idx] = 1

        return operation_mask

    def _build_binary_mask(self, segmentation: np.ndarray) -> np.ndarray:
        binary_mask = np.zeros(segmentation.shape[:2], dtype=np.uint8)
        if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
            return binary_mask

        for pairs in self._mask_object_pairs.values():
            pair_mask = np.zeros(segmentation.shape[:2], dtype=bool)
            for objtype, objid in pairs:
                pair_mask |= (segmentation[..., 0] == objid) & (
                    segmentation[..., 1] == objtype
                )
            binary_mask[pair_mask] = 1

        return binary_mask

    # ------------------------------------------------------------------
    # Model loading & sensor/actuator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(
        scene: SceneConfig,
        artifact: SceneArtifact | None = None,
    ) -> tuple[Any, Any]:
        model = load_composed_scene(scene, artifact)
        data = mujoco.MjData(model)
        return model, data

    def _init_tactile_manager(self) -> None:
        self._tactile_manager = TactileSensorManager(
            self.model,
            self.data,
            enable=DataType.TACTILE in self.config.enabled_sensors,
        )
        if self._tactile_manager.n_panels == 0:
            raise ValueError(
                "DataType.TACTILE is enabled but no tactile sensors "
                "(sites with 'touch_point') were found in the model."
            )

    def _resolve_frame(self, name: str) -> tuple[str, int, str] | None:
        """Resolve *name* to a frame: site takes priority over body.

        Returns ``("site", site_id, name)`` or ``("body", body_id, name)``,
        or ``None`` if the name is not found as either.
        """
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return ("site", int(site_id), name)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return ("body", int(body_id), name)
        return None

    def _frame_pose_world(
        self, frame: tuple[str, int, str]
    ) -> tuple[np.ndarray, np.ndarray]:
        kind, frame_id, _name = frame
        if kind == "site":
            return (
                np.asarray(self.data.site_xpos[frame_id], dtype=np.float64),
                np.asarray(self.data.site_xmat[frame_id], dtype=np.float64).reshape(
                    3, 3
                ),
            )
        return (
            np.asarray(self.data.xpos[frame_id], dtype=np.float64),
            np.asarray(self.data.xmat[frame_id], dtype=np.float64).reshape(3, 3),
        )

    def _apply_camera_calibrations(self) -> None:
        """Apply YAML camera calibration before capturing reset baselines."""
        calibrations = {
            name: spec.calibration
            for name, spec in self._camera_specs.items()
            if spec.calibration is not None and name in self._camera_ids
        }
        if not calibrations:
            return

        mujoco.mj_forward(self.model, self.data)
        for name, calibration in calibrations.items():
            assert calibration is not None
            cam_id = self._camera_ids[name]
            if calibration.fovy_deg is not None:
                self.model.cam_fovy[cam_id] = float(calibration.fovy_deg)
            extrinsics = calibration.extrinsics
            if extrinsics is None:
                continue

            parent = self._camera_parent_frame.get(name)
            if parent is None:
                body_id = int(self.model.cam_bodyid[cam_id])
                if body_id == 0:
                    parent_pos = np.zeros(3, dtype=np.float64)
                    parent_rot = np.eye(3, dtype=np.float64)
                else:
                    parent_pos = np.asarray(self.data.xpos[body_id], dtype=np.float64)
                    parent_rot = np.asarray(
                        self.data.xmat[body_id], dtype=np.float64
                    ).reshape(3, 3)
            else:
                parent_pos, parent_rot = self._frame_pose_world(parent)

            current_pos = parent_rot.T @ (
                np.asarray(self.data.cam_xpos[cam_id], dtype=np.float64) - parent_pos
            )
            current_rot = parent_rot.T @ np.asarray(
                self.data.cam_xmat[cam_id], dtype=np.float64
            ).reshape(3, 3)
            relative_pos = (
                np.asarray(extrinsics.position, dtype=np.float64)
                if extrinsics.position is not None
                else current_pos
            )
            relative_rot = (
                quaternion_to_rotation_matrix(
                    np.asarray(extrinsics.orientation, dtype=np.float64)
                )
                if extrinsics.orientation is not None
                else current_rot
            )
            world_pos = parent_pos + parent_rot @ relative_pos
            world_rot = parent_rot @ relative_rot

            body_id = int(self.model.cam_bodyid[cam_id])
            body_pos = np.asarray(self.data.xpos[body_id], dtype=np.float64)
            body_rot = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(
                3, 3
            )
            self.model.cam_pos[cam_id] = body_rot.T @ (world_pos - body_pos)
            world_quat_xyzw = quaternion_from_matrix_3x3(world_rot)
            world_quat_wxyz = np.asarray(
                [world_quat_xyzw[3], *world_quat_xyzw[:3]], dtype=np.float64
            )
            body_quat_wxyz = np.asarray(self.data.xquat[body_id], dtype=np.float64)
            inverse_body_quat = np.empty(4, dtype=np.float64)
            local_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_negQuat(inverse_body_quat, body_quat_wxyz)
            mujoco.mju_mulQuat(local_quat, inverse_body_quat, world_quat_wxyz)
            self.model.cam_quat[cam_id] = local_quat
            mujoco.mj_forward(self.model, self.data)

        self._model_cam_pos_baseline = np.asarray(
            self.model.cam_pos, dtype=np.float64
        ).copy()
        self._model_cam_quat_baseline = np.asarray(
            self.model.cam_quat, dtype=np.float64
        ).copy()

    def _sensor_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(sid)

    def _sensor_data(self, sensor_id: int) -> np.ndarray:
        if sensor_id < 0:
            return np.zeros((0,), dtype=np.float32)
        idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return np.asarray(self.data.sensordata[idx : idx + dim], dtype=np.float32)

    def _resolve_actuator_indices(self, actuator_names: List[str]) -> np.ndarray:
        """Return actuator index array for the given actuator names.

        Raises ``ValueError`` if any name is not found in the model.
        """
        indices = []
        for name in actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                available = [
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    for i in range(self.model.nu)
                ]
                raise ValueError(
                    f"Actuator '{name}' not found in the Mujoco model. "
                    f"Available actuators: {available}"
                )
            indices.append(int(aid))
        return np.asarray(indices, dtype=np.int32)

    def _actuator_joint_indices(
        self, actuator_indices: List[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(qpos_indices, dof_indices)`` for joints driven by the given actuators."""
        joint_indices = []
        velocity_indices = []
        for actuator_idx in actuator_indices:
            joint_id = int(self.model.actuator_trnid[actuator_idx, 0])
            if joint_id < 0:
                continue
            joint_indices.append(int(self.model.jnt_qposadr[joint_id]))
            velocity_indices.append(int(self.model.jnt_dofadr[joint_id]))
        return (
            np.asarray(joint_indices, dtype=np.int32),
            np.asarray(velocity_indices, dtype=np.int32),
        )

    def _actuator_joint_names(self, operator_name: str) -> List[str]:
        """Return joint names for the arm actuators of the given operator."""
        arm_aidx = self._op_arm_aidx[operator_name]
        names = []
        for aidx in arm_aidx:
            jid = int(self.model.actuator_trnid[aidx, 0])
            names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid))
        return names

    def _split_component_joint_state_indices(
        self, operator_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return pre-computed index arrays for ``operator_name``.

        Returns ``(arm_qidx, eef_qidx, arm_vidx, eef_vidx, arm_aidx, eef_aidx)``.
        """
        return (
            self._op_arm_qidx[operator_name],
            self._op_eef_qidx[operator_name],
            self._op_arm_vidx[operator_name],
            self._op_eef_vidx[operator_name],
            self._op_arm_aidx[operator_name],
            self._op_eef_aidx[operator_name],
        )

    # ------------------------------------------------------------------
    # Pose queries (world frame)
    # ------------------------------------------------------------------

    def _pose_rot9d(self, operator_name: str) -> np.ndarray:
        site_id = self._pose_site_ids.get(operator_name, -1)
        if site_id < 0:
            raise ValueError(f"No pose site found for operator '{operator_name}'")
        return self.data.site_xmat[site_id]

    def _quat_wxyz_to_xyzw(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32).reshape(-1)
        return quat[[1, 2, 3, 0]]

    def _rotmat_to_quat_xyzw(self, rotmat: np.ndarray) -> np.ndarray:
        quat_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_wxyz, np.asarray(rotmat, dtype=np.float64).reshape(9))
        return self._quat_wxyz_to_xyzw(quat_wxyz).astype(np.float32)

    def _site_pose(self, operator_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = self._pose_site_ids.get(operator_name, -1)
        if site_id < 0:
            raise ValueError(f"No pose site found for operator '{operator_name}'")
        pos = np.asarray(self.data.site_xpos[site_id], dtype=np.float32)
        quat = self._rotmat_to_quat_xyzw(self.data.site_xmat[site_id])
        return pos, quat

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in the Mujoco model.")
        pos = np.asarray(self.data.xpos[body_id], dtype=np.float32)
        quat_wxyz = np.asarray(self.data.xquat[body_id], dtype=np.float32)
        quat_xyzw = self._quat_wxyz_to_xyzw(quat_wxyz)
        return pos, quat_xyzw

    def get_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in the Mujoco model.")
        pos = np.asarray(self.data.site_xpos[site_id], dtype=np.float32)
        quat = self._rotmat_to_quat_xyzw(self.data.site_xmat[site_id])
        return pos, quat

    def _quats_equivalent_xyzw(
        self, quat_a: np.ndarray, quat_b: np.ndarray, atol: float = 1e-5
    ) -> bool:
        quat_a = np.asarray(quat_a, dtype=np.float32).reshape(-1)
        quat_b = np.asarray(quat_b, dtype=np.float32).reshape(-1)
        return np.allclose(quat_a, quat_b, atol=atol) or np.allclose(
            quat_a, -quat_b, atol=atol
        )

    def _validate_pose_sensor_matches_site(
        self, operator_name: str, pos: np.ndarray, quat_xyzw: np.ndarray
    ) -> None:
        if operator_name in self._pose_validated_components:
            return

        site_pos, site_quat_xyzw = self._site_pose(operator_name)
        if not np.allclose(pos, site_pos, atol=1e-5) or not self._quats_equivalent_xyzw(
            quat_xyzw, site_quat_xyzw, atol=1e-5
        ):
            raise ValueError(
                "Pose sensor does not match pose site for operator "
                f"'{operator_name}': sensor_pos={pos.tolist()}, site_pos={site_pos.tolist()}, "
                f"sensor_quat_xyzw={quat_xyzw.tolist()}, "
                f"site_quat_xyzw={site_quat_xyzw.tolist()}"
            )

        self._pose_validated_components.add(operator_name)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _restore_model_pose_baseline(self) -> None:
        """Restore model-level body and camera poses captured at load time.

        ``mj_resetData`` only resets dynamic ``MjData`` values; it intentionally
        leaves ``MjModel.body_pos/body_quat`` and ``cam_pos/cam_quat`` alone.
        Those arrays are nevertheless mutated by static-object placement,
        operator-base relocation, and camera randomization.  Restoring them at
        the low-level reset seam prevents one episode's structural pose from
        becoming the implicit baseline of the next episode.
        """
        self.model.body_pos[...] = self._model_body_pos_baseline
        self.model.body_quat[...] = self._model_body_quat_baseline
        self.model.cam_pos[...] = self._model_cam_pos_baseline
        self.model.cam_quat[...] = self._model_cam_quat_baseline

    def _after_reset(self) -> None:
        """Hook for higher-level wrappers to restore derived reset state."""
        return None

    def _sync_mocap_to_freejoint(self) -> None:
        """Synchronize mocap bodies with their weld-connected physical bodies.

        After ``mj_resetDataKeyframe`` the physical body has correct xpos/xquat
        (computed via ``mj_forward`` from the keyframe qpos), but mocap_pos and
        mocap_quat are NOT set by the keyframe.  This copies the physical body's
        world pose to the corresponding mocap body so the weld constraint starts
        in equilibrium.
        """
        for i in range(self.model.neq):
            if self.model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD:
                continue
            b1 = int(self.model.eq_obj1id[i])
            b2 = int(self.model.eq_obj2id[i])
            mid1 = int(self.model.body_mocapid[b1])
            mid2 = int(self.model.body_mocapid[b2])
            if mid1 >= 0:
                mocap_id, phys_id = mid1, b2
            elif mid2 >= 0:
                mocap_id, phys_id = mid2, b1
            else:
                continue
            self.data.mocap_pos[mocap_id] = self.data.xpos[phys_id].copy()
            self.data.mocap_quat[mocap_id] = self.data.xquat[phys_id].copy()

    def _reset_core(self) -> None:
        """Restore model/data state without invoking subclass lifecycle hooks."""
        self._restore_model_pose_baseline()
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        actuator_ids = (
            int(actuator_id)
            for operator in self._operators.values()
            for indices in (
                self._op_arm_aidx[operator.name],
                self._op_eef_aidx[operator.name],
            )
            for actuator_id in indices
        )
        apply_initial_joint_positions(
            self.model,
            self.data,
            self.config.initial_joint_positions,
            actuator_ids,
        )
        self._sync_mocap_to_freejoint()
        self._prev_ctrl = None

    def reset(self) -> None:
        """Restore low-level state and notify higher-level wrappers."""
        self._reset_core()
        self._after_reset()

    def _snapshot_ctrl(self) -> None:
        """Capture current ctrl as the interpolation baseline for the next update."""
        if self._ctrl_interp and self._prev_ctrl is None:
            self._prev_ctrl = self.data.ctrl.copy()

    def update(self):
        if self._ctrl_interp:
            new_ctrl = self.data.ctrl.copy()
            old_ctrl = self._prev_ctrl if self._prev_ctrl is not None else new_ctrl
            for i in range(self._n_substeps):
                alpha = (i + 1) / self._n_substeps
                self.data.ctrl[:] = old_ctrl + alpha * (new_ctrl - old_ctrl)
                for cb in self._pre_step_callbacks:
                    cb(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
            self._prev_ctrl = new_ctrl
        else:
            for _ in range(self._n_substeps):
                for cb in self._pre_step_callbacks:
                    cb(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
        if self._viewer_running():
            refreshed = self._sync_viewer()
            if refreshed and self.config.viewer.step_delay > 0.0:
                time.sleep(self.config.viewer.step_delay)

    def is_updated(self) -> bool:
        current_time = self.data.time
        if self._last_time != current_time:
            self._last_time = current_time
            return True
        return False

    # ------------------------------------------------------------------
    # Camera render visibility
    # ------------------------------------------------------------------

    @contextmanager
    def _camera_clip_scope(
        self,
        clip_range_m: Tuple[PositiveFloat, PositiveFloat] | None,
    ) -> Iterator[None]:
        """Apply one output stream's metric clip range for a render pass."""
        if clip_range_m is None:
            yield
            return
        extent = float(self.model.stat.extent)
        if not np.isfinite(extent) or extent <= 0.0:
            raise ValueError(
                "Cannot apply camera clip range: model.stat.extent must be positive"
            )
        previous_znear = float(self.model.vis.map.znear)
        previous_zfar = float(self.model.vis.map.zfar)
        near_m, far_m = (float(value) for value in clip_range_m)
        try:
            self.model.vis.map.znear = near_m / extent
            self.model.vis.map.zfar = far_m / extent
            yield
        finally:
            self.model.vis.map.znear = previous_znear
            self.model.vis.map.zfar = previous_zfar

    def _resolve_operator_render_geom_ids(self) -> frozenset[int]:
        """Return geom IDs belonging to configured operator body subtrees."""
        root_names = {
            body_name
            for operator in self._operators.values()
            for body_name in (operator.root_body, operator.mocap_body)
            if body_name
        }
        if not root_names:
            self.get_logger().warning(
                "hide_operators_in_camera is enabled, but no operator has a "
                "root_body or mocap_body."
            )
            return frozenset()

        root_ids: set[int] = set()
        for body_name in root_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(
                    "hide_operators_in_camera could not find operator body "
                    f"'{body_name}' in the MuJoCo model."
                )
            root_ids.add(int(body_id))

        hidden_body_ids = set(root_ids)
        for body_id in range(1, self.model.nbody):
            ancestor_id = body_id
            while ancestor_id > 0:
                if ancestor_id in root_ids:
                    hidden_body_ids.add(body_id)
                    break
                ancestor_id = int(self.model.body_parentid[ancestor_id])

        return frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in hidden_body_ids
        )

    def _hide_operator_geoms_from_camera_scene(self, renderer: Any) -> None:
        """Remove configured operator geoms from one populated render scene.

        Mutating ``MjvGeom.type`` affects only this renderer's transient scene;
        it does not alter ``MjModel`` and therefore cannot change simulation
        physics or the passive viewer.
        """
        hidden_geom_ids = getattr(self, "_camera_hidden_geom_ids", frozenset())
        if not hidden_geom_ids:
            return

        geom_obj_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        hidden_type = int(mujoco.mjtGeom.mjGEOM_NONE)
        for scene_geom in renderer.scene.geoms[: renderer.scene.ngeom]:
            if (
                int(scene_geom.objtype) == geom_obj_type
                and int(scene_geom.objid) in hidden_geom_ids
            ):
                scene_geom.type = hidden_type

    # ------------------------------------------------------------------
    # Info / lifecycle
    # ------------------------------------------------------------------

    def _get_camera_info(self) -> Dict[str, dict]:
        info = {}
        for cam_name, cam_id in self._camera_ids.items():
            spec = self._camera_specs[cam_name]
            fovy_deg = float(self.model.cam_fovy[cam_id])
            fovy_rad = fovy_deg * pi / 180.0
            f = (spec.height / 2.0) / tan(fovy_rad / 2.0)

            camera_info = {
                "header": {"frame_id": cam_name},
                "width": spec.width,
                "height": spec.height,
                "distortion_model": "plumb_bob",
                "d": [0.0, 0.0, 0.0, 0.0, 0.0],
                "k": [
                    f,
                    0.0,
                    spec.width / 2.0,
                    0.0,
                    f,
                    spec.height / 2.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "r": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "p": [
                    f,
                    0.0,
                    spec.width / 2.0,
                    0.0,
                    0.0,
                    f,
                    spec.height / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            }
            info[cam_name] = camera_info
        return info

    def get_camera_model(self, camera_name: str) -> CameraModel:
        """Return the current world-frame camera model for constraints."""
        cam_id = self._camera_ids.get(camera_name)
        if cam_id is None:
            raise KeyError(f"Camera '{camera_name}' not found in the MuJoCo model.")
        mujoco.mj_forward(self.model, self.data)
        spec = self._camera_specs[camera_name]
        pose = PoseState(
            position=np.asarray(self.data.cam_xpos[cam_id], dtype=np.float64),
            orientation=quaternion_from_matrix_3x3(
                np.asarray(self.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
            ),
        )
        fovy = float(self.model.cam_fovy[cam_id]) * pi / 180.0
        default_near_m = float(self.model.vis.map.znear * self.model.stat.extent)
        default_far_m = float(self.model.vis.map.zfar * self.model.stat.extent)
        active_clip_ranges: list[tuple[float, float]] = []
        if spec.enable_color or spec.enable_mask or spec.enable_heat_map:
            active_clip_ranges.append(
                (default_near_m, default_far_m)
                if spec.rgb_clip_range_m is None
                else tuple(float(value) for value in spec.rgb_clip_range_m)
            )
        if spec.enable_depth:
            active_clip_ranges.append(
                (default_near_m, default_far_m)
                if spec.depth_clip_range_m is None
                else tuple(float(value) for value in spec.depth_clip_range_m)
            )
        if active_clip_ranges:
            # A visibility-constrained object must survive every enabled image stream.
            near_m = max(near_m for near_m, _ in active_clip_ranges)
            far_m = min(far_m for _, far_m in active_clip_ranges)
        else:
            near_m, far_m = default_near_m, default_far_m
        return CameraModel(
            name=camera_name,
            pose=pose,
            width=spec.width,
            height=spec.height,
            fovy_radians=fovy,
            near=near_m,
            far=far_m,
        )

    def get_support_geometry(self, entity_name: str) -> SupportGeometry:
        """Return a conservative sphere around the named body's current geoms."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, entity_name)
        if body_id < 0:
            raise KeyError(f"Entity '{entity_name}' not found as a MuJoCo body.")
        mujoco.mj_forward(self.model, self.data)
        geom_ids = [
            gid
            for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) == body_id
        ]
        if not geom_ids:
            return SupportGeometry(
                center=np.asarray(self.data.xpos[body_id], dtype=np.float64),
                radius=0.0,
            )
        center = np.mean(
            np.asarray(
                [self.data.geom_xpos[gid] for gid in geom_ids], dtype=np.float64
            ),
            axis=0,
        )
        radius = 0.0
        for gid in geom_ids:
            geom_center = np.asarray(self.data.geom_xpos[gid], dtype=np.float64)
            size = np.asarray(self.model.geom_size[gid], dtype=np.float64)
            geom_radius = float(np.linalg.norm(size))
            radius = max(
                radius, float(np.linalg.norm(geom_center - center)) + geom_radius
            )
        return SupportGeometry(center=center, radius=radius)

    def evaluate_randomization_constraints(
        self,
        candidate_poses: Mapping[str, PoseState],
        *,
        env_index: int = 0,
        constraints: Any = None,
        ancestors: Optional[Mapping[str, Set[str]]] = None,
        target_names: Optional[Set[str]] = None,
    ) -> RandomizationConstraintReport:
        """Evaluate frustum and support-separation constraints for a candidate."""
        if constraints is None:
            return RandomizationConstraintReport(valid=True)
        violations: list[str] = []
        minimum_clearance = float("inf")
        visible = getattr(constraints, "visible_in", None)
        if visible is not None:
            if visible.mode.value != "frustum":
                raise NotImplementedError(
                    "MuJoCo randomization currently supports frustum visibility; "
                    "segmentation visibility belongs to the backend hardening phase."
                )
            if visible.geometry.value == "support_hull":
                raise NotImplementedError(
                    "MuJoCo randomization currently uses a bounding-sphere support; "
                    "support-hull visibility belongs to the backend hardening phase."
                )
            cameras = (
                list(self._camera_ids)
                if visible.cameras == "all"
                else list(visible.cameras)
            )
            visibility_candidates = (
                candidate_poses
                if target_names is None
                else {
                    name: pose
                    for name, pose in candidate_poses.items()
                    if name in target_names
                }
            )
            for entity_name, pose in visibility_candidates.items():
                geometry = self.get_support_geometry(entity_name)
                delta = np.asarray(pose.position[0], dtype=np.float64) - geometry.center
                # The support sphere follows the proposed pose translation.
                center = np.asarray(pose.position[0], dtype=np.float64)
                radius = (
                    0.0
                    if visible.geometry.value == "center"
                    else float(geometry.radius)
                )
                for camera_name in cameras:
                    camera = self.get_camera_model(camera_name)
                    cam_pos = camera.pose.position[0]
                    rotation = np.asarray(
                        quaternion_to_rotation_matrix(camera.pose.orientation[0]),
                        dtype=np.float64,
                    )
                    # MuJoCo camera x-axis is right, y-axis up, z-axis backward.
                    camera_point = rotation.T @ (center - cam_pos)
                    depth = -float(camera_point[2])
                    half_fovy = camera.fovy_radians / 2.0
                    half_fovx = np.arctan(
                        np.tan(half_fovy) * (camera.width / camera.height)
                    )
                    depth_margin = radius / max(depth, 1e-9)
                    px = camera.width * 0.5 + camera_point[0] / max(depth, 1e-9) * (
                        camera.width * 0.5 / np.tan(half_fovx)
                    )
                    py = camera.height * 0.5 - camera_point[1] / max(depth, 1e-9) * (
                        camera.height * 0.5 / np.tan(half_fovy)
                    )
                    margin = float(visible.margin_px)
                    if depth <= camera.near + radius or depth >= camera.far - radius:
                        violations.append(f"{entity_name}:outside_depth:{camera_name}")
                    if (
                        px - depth_margin * camera.width < margin
                        or px + depth_margin * camera.width > camera.width - margin
                        or py - depth_margin * camera.height < margin
                        or py + depth_margin * camera.height > camera.height - margin
                    ):
                        violations.append(f"{entity_name}:outside_view:{camera_name}")

        separated = getattr(constraints, "separated", None)
        if separated is not None and separated.scope == "scene":
            raise NotImplementedError(
                "MuJoCo scene-scope separation requires geom-level broad-phase "
                "support and is reserved for the backend hardening phase."
            )
        if separated is not None and separated.scope == "randomized":
            names = list(candidate_poses)
            for index, left_name in enumerate(names):
                left_geometry = self.get_support_geometry(left_name)
                left_center = np.asarray(
                    candidate_poses[left_name].position[0], dtype=np.float64
                )
                for right_name in names[index + 1 :]:
                    if ancestors and (
                        right_name in ancestors.get(left_name, set())
                        or left_name in ancestors.get(right_name, set())
                    ):
                        continue
                    right_geometry = self.get_support_geometry(right_name)
                    right_center = np.asarray(
                        candidate_poses[right_name].position[0], dtype=np.float64
                    )
                    clearance = (
                        float(np.linalg.norm(left_center - right_center))
                        - left_geometry.radius
                        - right_geometry.radius
                        - float(separated.clearance)
                    )
                    minimum_clearance = min(minimum_clearance, clearance)
                    if clearance < 0.0:
                        violations.append(f"{left_name}:collides:{right_name}")
        return RandomizationConstraintReport(
            valid=not violations,
            violations=tuple(violations),
            minimum_clearance=minimum_clearance,
        )

    def _get_camera_extrinsics(self) -> Dict[str, dict]:
        extrinsics = {}
        for cam_name, cam_id in self._camera_ids.items():
            # Camera frame: prefer same-named site (e.g. optical convention) over cam_xmat.
            cam_site_id = self._camera_frame_site_ids.get(cam_name)
            if cam_site_id is not None:
                cam_rot = np.asarray(self.data.site_xmat[cam_site_id]).reshape(3, 3)
                cam_pos = np.asarray(self.data.site_xpos[cam_site_id])
            else:
                cam_rot = np.asarray(self.data.cam_xmat[cam_id]).reshape(3, 3)
                cam_pos = np.asarray(self.data.cam_xpos[cam_id])

            # Parent frame: express camera pose relative to it.
            parent = self._camera_parent_frame.get(cam_name)
            if parent is not None:
                kind, ref_id, frame_name = parent
                if kind == "site":
                    ref_rot = np.asarray(self.data.site_xmat[ref_id]).reshape(3, 3)
                    ref_pos = np.asarray(self.data.site_xpos[ref_id])
                else:
                    ref_rot = np.asarray(self.data.xmat[ref_id]).reshape(3, 3)
                    ref_pos = np.asarray(self.data.xpos[ref_id])
                cam_rot = ref_rot.T @ cam_rot
                cam_pos = ref_rot.T @ (cam_pos - ref_pos)
                extrinsics_frame = frame_name
            else:
                extrinsics_frame = "world"
            extrinsics[cam_name] = {
                "frame": extrinsics_frame,
                "translation": cam_pos,
                "rotation_matrix": cam_rot,
            }
        return extrinsics

    def get_info(self, cached: bool = True) -> dict[str, Any]:
        if cached:
            if self._info is None:
                self._info = self.get_info(cached=False)
            return self._info
        mujoco.mj_forward(self.model, self.data)
        info: dict[str, Any] = self.config.model_dump(
            mode="json", exclude={"cameras", "pre_step_callbacks"}
        )
        info["cameras"] = {}
        camera_info = self._get_camera_info()
        camera_extrinsics = self._get_camera_extrinsics()
        for cam_name in self._camera_ids:
            spec = self._camera_specs[cam_name]
            info["cameras"][cam_name] = {
                "camera_info": {
                    stream_type: camera_info[cam_name]
                    for stream_type in ("color", "depth")
                },
                # TODO: should separate extrinsics for color and depth?
                "camera_extrinsics": camera_extrinsics[cam_name],
                "rgb_clip_range_m": (
                    list(spec.rgb_clip_range_m)
                    if spec.rgb_clip_range_m is not None
                    else None
                ),
                "depth_clip_range_m": (
                    list(spec.depth_clip_range_m)
                    if spec.depth_clip_range_m is not None
                    else None
                ),
                "noise": (
                    spec.noise.model_dump(mode="json")
                    if spec.noise is not None
                    else None
                ),
            }
        return info

    def close(self) -> None:
        for renderer in self._renderers.values():
            if hasattr(renderer, "close"):
                renderer.close()
        self._renderers.clear()
        if self._viewer is not None:
            hold = self.config.viewer.hold_seconds
            if hold > 0.0:
                deadline = time.time() + hold
                while time.time() < deadline and self._viewer_running():
                    self._sync_viewer()
                    time.sleep(min(0.05, hold))
            self._shutdown_viewer()

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)
