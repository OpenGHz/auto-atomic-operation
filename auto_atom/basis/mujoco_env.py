from enum import Enum
from math import tan, pi
from pathlib import Path
from typing import Any, Dict, List, Set
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from auto_atom.utils.transformations import euler_from_matrix
from auto_atom.backend.mjc.tactile.tactile_sensor import TactileSensorManager
import os
import time
import numpy as np
import mujoco
import logging


class DataType(str, Enum):
    CAMERA = "camera"
    IMU = "imu"
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_EFFORT = "joint_effort"
    TACTILE = "tactile"
    WRENCH = "wrench"
    POSE = "pose"


class CameraSpec(BaseModel, frozen=True):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    name: str
    """The camera name as defined in the Mujoco model."""
    width: int = 640
    """The rendered image width in pixels."""
    height: int = 480
    """The rendered image height in pixels."""
    enable_color: bool = True
    """Whether to include RGB images in the captured observation."""
    enable_depth: bool = True
    """Whether to include depth images in the captured observation."""
    depth_max: float = 5.0
    """Maximum valid depth in metres; pixels beyond this distance are set to 0."""
    enable_mask: bool = False
    """Whether to include a binary segmentation mask for configured objects."""
    enable_heat_map: bool = False
    """Whether to include per-operation heat maps derived from object masks."""
    parent_frame: str = ""
    """Name of the site or body whose frame is used as the reference for
    camera extrinsics.  When set, the name is resolved as a site first; if no
    site with that name exists it is resolved as a body.  If empty, the
    reference frame is auto-detected from the camera's attached body (again
    preferring a same-named site over the body itself).  Cameras attached
    directly to the world body keep the world frame."""


class ViewerConfig(BaseModel, frozen=True):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    lookat: List[float] | None = None
    """Camera lookat point [x, y, z]. Uses Mujoco default if None."""
    distance: float | None = None
    """Camera distance. Uses Mujoco default if None."""
    azimuth: float | None = None
    """Camera azimuth angle in degrees. Uses Mujoco default if None."""
    elevation: float | None = None
    """Camera elevation angle in degrees. Uses Mujoco default if None."""
    step_delay: float = 0.0
    """Seconds to sleep after each update step."""
    hold_seconds: float = 0.0
    """Seconds to keep the viewer open after close() is called."""
    disable: bool = False
    """Whether to disable launching the viewer, even if a ViewerConfig is provided."""


class OperatorBinding(BaseModel, frozen=True):
    """Binds a logical operator name to the actuators and sensors defined in the XML model."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    name: str
    """Logical operator name, used as the key prefix in observation dicts."""

    arm_actuators: List[str] = Field(default_factory=list)
    """Actuator names (as defined in the XML) that drive the main arm/body joints."""

    eef_actuators: List[str] = Field(default_factory=list)
    """Actuator names (as defined in the XML) that drive the end-effector/gripper joints."""

    arm_output_name: str = ""
    """Observation key prefix for arm joint data. Defaults to the operator ``name``."""

    eef_output_name: str = "eef"
    """Observation key prefix for end-effector joint data."""

    pose_site: str = ""
    """Site name (as defined in the XML) used for end-effector pose output.
    Required when POSE sensor is enabled for this operator."""

    imu_acc: str = ""
    """Sensor name for IMU linear acceleration (accelerometer)."""

    imu_gyro: str = ""
    """Sensor name for IMU angular velocity (gyroscope)."""

    imu_quat: str = ""
    """Sensor name for IMU orientation quaternion."""

    pose_sensor_pos: str = ""
    """Sensor name for end-effector position, cross-validated against pose_site when provided."""

    pose_sensor_quat: str = ""
    """Sensor name for end-effector orientation quaternion, cross-validated against pose_site."""

    wrench_force: str = ""
    """Sensor name for end-effector force. Falls back to tactile-derived wrench if empty."""

    wrench_torque: str = ""
    """Sensor name for end-effector torque. Falls back to tactile-derived wrench if empty."""

    tactile_prefixes: List[str] = Field(default_factory=list)
    """Tactile panel name prefixes that belong to this operator's end-effector.
    An empty list means all panels are assigned to this operator (only valid for a single operator)."""


class EnvConfig(BaseModel, frozen=True):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    model_path: Path
    """The path to the Mujoco XML model file used to create the environment."""
    operators: List[OperatorBinding] = Field(default_factory=list)
    """Operator definitions mapping logical names to XML actuators and sensors."""
    enabled_sensors: Set[DataType] = Field(default_factory=set)
    """The sensor categories that should be exposed in captured observations."""
    cameras: List[CameraSpec] = Field(default_factory=list)
    """The camera specifications to initialize when camera output is enabled."""
    mask_objects: List[str] = Field(default_factory=list)
    """The object names that are eligible for binary mask and heat-map generation."""
    operations: List[str] = Field(default_factory=list)
    """The operation names used to assign channels in generated heat maps."""
    stamp_ns: bool = True
    """Whether observation timestamps should be emitted in nanoseconds instead of seconds."""
    sim_freq: float | None = None
    """Physics simulation frequency in Hz. If None, uses the timestep defined in the XML model."""
    update_freq: float | None = None
    """Control update frequency in Hz. Must be <= sim_freq. If None, defaults to sim_freq (n_substeps=1)."""
    initial_joint_positions: Dict[str, float] = Field(default_factory=dict)
    """Joint name → qpos value overrides applied after every reset (after the keyframe)."""
    viewer: ViewerConfig | None = None
    """Viewer configuration. If None, the passive viewer is not launched."""
    structured: bool = False
    """Whether the observation value data should be flattened to 1D arrays when possible, e.g. for joint states."""

    @model_validator(mode="after")
    def validate_frequencies(self):
        if self.update_freq is not None:
            if self.sim_freq is None:
                raise ValueError("sim_freq must be set when update_freq is set.")
            if self.update_freq > self.sim_freq:
                raise ValueError(
                    f"update_freq ({self.update_freq} Hz) must be <= sim_freq ({self.sim_freq} Hz)."
                )
            if self.sim_freq % self.update_freq != 0:
                raise ValueError(
                    f"sim_freq ({self.sim_freq} Hz) must be divisible by update_freq ({self.update_freq} Hz)."
                )
        return self

    @model_validator(mode="after")
    def validate_operations(self):
        for cam_cfg in self.cameras:
            if cam_cfg.enable_heat_map:
                for field in ("operations", "mask_objects"):
                    if not getattr(self, field):
                        raise ValueError(f"{field} must be set when enable_heat_map")
            if cam_cfg.enable_mask and not self.mask_objects:
                raise ValueError("mask_objects must be set when enable_mask")
        return self

    @field_validator("viewer")
    @classmethod
    def validate_viewer(cls, v: ViewerConfig | None) -> ViewerConfig | None:
        if v is not None and v.disable:
            return None
        return v


class UnifiedMujocoEnv:
    def __init__(self, config: EnvConfig):
        self.get_logger().info("Initializing...")
        self.config = config
        self.model, self.data = self._load_model(config.model_path)

        if config.sim_freq is not None:
            self.model.opt.timestep = 1.0 / config.sim_freq
        self.get_logger().info(
            f"Using timestep of {self.model.opt.timestep:.6f} seconds"
        )
        self._n_substeps = (
            int(config.sim_freq / config.update_freq)
            if config.sim_freq is not None and config.update_freq is not None
            else 1
        )

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._sync_mocap_to_freejoint()

        # Pre-compute per-operator actuator and joint index arrays.
        self._operators = config.operators
        self._op_arm_aidx: dict[str, np.ndarray] = {}
        self._op_eef_aidx: dict[str, np.ndarray] = {}
        self._op_arm_qidx: dict[str, np.ndarray] = {}
        self._op_eef_qidx: dict[str, np.ndarray] = {}
        self._op_arm_vidx: dict[str, np.ndarray] = {}
        self._op_eef_vidx: dict[str, np.ndarray] = {}
        self._op_output_names: dict[str, tuple[str, str]] = {}
        for op in self._operators:
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

        self._camera_specs = {c.name: c for c in config.cameras}
        self._renderers: Dict[str, mujoco.Renderer] = {}
        self._camera_ids = {}
        self._renderer_scene_option = mujoco.MjvOption()
        self._renderer_scene_option.sitegroup[:] = 0
        self._interest_object_operations: dict[str, str] = {}
        self._mask_object_pairs = self._build_mask_object_pairs(config.mask_objects)

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
        for op in self._operators:
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

        self._tactile_manager = None
        if (
            DataType.TACTILE in config.enabled_sensors
            or DataType.WRENCH in config.enabled_sensors
        ):
            self._init_tactile_manager()
        self._last_time = None

        self._viewer = None
        if config.viewer is not None:
            self._launch_viewer()

    def _launch_viewer(self) -> None:
        import mujoco.viewer as _mj_viewer

        self._viewer = _mj_viewer.launch_passive(self.model, self.data)
        # self._viewer.opt.sitegroup[:] = 0
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

    def _sync_viewer(self) -> None:
        try:
            self._viewer.sync()
        except Exception:
            pass

    def _shutdown_viewer(self) -> None:
        try:
            self._viewer.close()
        except Exception:
            pass
        deadline = time.time() + 1.0
        while time.time() < deadline:
            if not self._viewer_running():
                break
            time.sleep(0.01)
        time.sleep(0.05)
        self._viewer = None

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
                "object_names and operation_names must have the same length."
            )

        interest_object_operations: dict[str, str] = {}
        for object_name, operation_name in zip(object_names, operation_names):
            if self.config.mask_objects and object_name not in self._mask_object_pairs:
                raise ValueError(
                    f"Object '{object_name}' is not configured in mask_objects: {self.config.mask_objects}"
                )
            if operation_name not in self.config.operations:
                raise ValueError(
                    f"Operation '{operation_name}' is not configured in operations: {self.config.operations}"
                )
            interest_object_operations[object_name] = operation_name

        self._interest_object_operations = interest_object_operations

    def _build_operation_mask(self, segmentation: np.ndarray) -> np.ndarray:
        operation_mask = np.zeros(
            (*segmentation.shape[:2], len(self.config.operations)), dtype=np.uint8
        )
        if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
            return operation_mask

        for object_name, operation_name in self._interest_object_operations.items():
            channel_idx = self.config.operations.index(operation_name)
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

    @staticmethod
    def _load_model(model_path: Path) -> tuple[Any, Any]:
        original_dir = os.getcwd()
        xml_path = Path(model_path).resolve()
        xml_dir = xml_path.parent
        try:
            os.chdir(xml_dir)
            model = mujoco.MjModel.from_xml_path(xml_path.name)
            data = mujoco.MjData(model)
        finally:
            os.chdir(original_dir)
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

    def reset(self) -> None:
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        for joint_name, qpos_val in self.config.initial_joint_positions.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                self.data.qpos[int(self.model.jnt_qposadr[jid])] = qpos_val
        mujoco.mj_forward(self.model, self.data)
        self._sync_mocap_to_freejoint()

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        n = min(len(action), self.model.nu)
        if n > 0:
            ctrl = np.asarray(self.data.ctrl, dtype=np.float64)
            ctrl[:n] = action[:n]
            if self.model.nu > 0:
                low = self.model.actuator_ctrlrange[:n, 0]
                high = self.model.actuator_ctrlrange[:n, 1]
                ctrl[:n] = np.clip(ctrl[:n], low, high)
            self.data.ctrl[:n] = ctrl[:n]
        self.update()

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        return self._collect_obs(self.config.structured)

    def _collect_obs(self, structured: bool) -> dict[str, dict[str, Any]]:
        t = int(self.data.time * 1e9) if self.config.stamp_ns else float(self.data.time)
        obs: dict[str, dict[str, Any]] = {}

        for op in self._operators:
            arm_qidx = self._op_arm_qidx[op.name]
            eef_qidx = self._op_eef_qidx[op.name]
            arm_vidx = self._op_arm_vidx[op.name]
            eef_vidx = self._op_eef_vidx[op.name]
            arm_aidx = self._op_arm_aidx[op.name]
            eef_aidx = self._op_eef_aidx[op.name]
            arm_name, eef_name = self._op_output_names[op.name]

            if DataType.JOINT_POSITION in self.config.enabled_sensors:
                if structured:
                    for prefix, limb, qidx, vidx, aidx in [
                        ("enc", arm_name, arm_qidx, arm_vidx, arm_aidx),
                        ("enc", eef_name, eef_qidx, eef_vidx, eef_aidx),
                        ("action", arm_name, arm_qidx, arm_vidx, arm_aidx),
                        ("action", eef_name, eef_qidx, eef_vidx, eef_aidx),
                    ]:
                        obs[f"{prefix}/{limb}/joint_state"] = {
                            "data": {
                                "position": np.asarray(
                                    self.data.qpos[qidx], dtype=np.float32
                                ),
                                "velocity": np.asarray(
                                    self.data.qvel[vidx], dtype=np.float32
                                ),
                                "effort": np.asarray(
                                    self.data.ctrl[aidx], dtype=np.float32
                                ),
                            },
                            "t": t,
                        }
                else:
                    for limb, qidx in [(arm_name, arm_qidx), (eef_name, eef_qidx)]:
                        obs[f"{limb}/joint_state/position"] = {
                            "data": np.asarray(self.data.qpos[qidx], dtype=np.float32),
                            "t": t,
                        }
                    for limb, aidx in [(arm_name, arm_aidx), (eef_name, eef_aidx)]:
                        obs[f"action/{limb}/joint_state/position"] = {
                            "data": np.asarray(self.data.ctrl[aidx], dtype=np.float32),
                            "t": t,
                        }

            if not structured:
                if DataType.JOINT_VELOCITY in self.config.enabled_sensors:
                    for limb, vidx in [(arm_name, arm_vidx), (eef_name, eef_vidx)]:
                        obs[f"{limb}/joint_state/velocity"] = {
                            "data": np.asarray(self.data.qvel[vidx], dtype=np.float32),
                            "t": t,
                        }
                if DataType.JOINT_EFFORT in self.config.enabled_sensors:
                    for limb, aidx in [(arm_name, arm_aidx), (eef_name, eef_aidx)]:
                        obs[f"{limb}/joint_state/effort"] = {
                            "data": np.asarray(self.data.ctrl[aidx], dtype=np.float32),
                            "t": t,
                        }

            if DataType.POSE in self.config.enabled_sensors:
                site_id = self._pose_site_ids.get(op.name, -1)
                if site_id >= 0:
                    pos_id = self._pose_ids[op.name]["pos"]
                    quat_id = self._pose_ids[op.name]["quat"]
                    if pos_id >= 0 and quat_id >= 0:
                        pos = self._sensor_data(pos_id)
                        quat = self._quat_wxyz_to_xyzw(self._sensor_data(quat_id))
                        self._validate_pose_sensor_matches_site(op.name, pos, quat)
                    else:
                        pos, quat = self._site_pose(op.name)
                    rot9d = self._pose_rot9d(op.name)
                    if structured:
                        obs[f"{op.name}/pose"] = {
                            "data": {
                                "position": pos.astype(np.float32),
                                "orientation": quat.astype(np.float32),
                            },
                            "t": t,
                        }
                    else:
                        obs[f"{op.name}/pose/position"] = {
                            "data": pos.astype(np.float32),
                            "t": t,
                        }
                        obs[f"{op.name}/pose/orientation"] = {
                            "data": quat.astype(np.float32),
                            "t": t,
                        }
                    obs[f"{op.name}/pose/rotation"] = {
                        "data": euler_from_matrix(rot9d.reshape(3, 3)),
                        "t": t,
                    }
                    obs[f"{op.name}/pose/rotation_6d"] = {
                        "data": rot9d[:6].astype(np.float32),
                        "t": t,
                    }

            if DataType.IMU in self.config.enabled_sensors:
                acc_id = self._imu_ids[op.name]["acc"]
                gyro_id = self._imu_ids[op.name]["gyro"]
                quat_id = self._imu_ids[op.name]["quat"]
                if acc_id >= 0 and gyro_id >= 0 and quat_id >= 0:
                    acc = self._sensor_data(acc_id)
                    gyro = self._sensor_data(gyro_id)
                    imu_quat = self._sensor_data(quat_id)
                    if structured:
                        obs[f"{op.name}/imu"] = {
                            "data": {
                                "linear_acceleration": acc,
                                "angular_velocity": gyro,
                                "orientation": imu_quat,
                            },
                            "t": t,
                        }
                    else:
                        obs[f"{op.name}/imu/linear_acceleration"] = {
                            "data": acc,
                            "t": t,
                        }
                        obs[f"{op.name}/imu/angular_velocity"] = {"data": gyro, "t": t}
                        obs[f"{op.name}/imu/orientation"] = {"data": imu_quat, "t": t}

            if DataType.WRENCH in self.config.enabled_sensors:
                force = self._sensor_data(self._wrench_ids[op.name]["force"])
                torque = self._sensor_data(self._wrench_ids[op.name]["torque"])
                if force.size == 0 or torque.size == 0:
                    force, torque = self._wrench_from_tactile(op)
                force = np.asarray(force, dtype=np.float32)
                torque = np.asarray(torque, dtype=np.float32)
                if structured:
                    obs[f"{op.name}/wrench"] = {
                        "data": {"force": force, "torque": torque},
                        "t": t,
                    }
                else:
                    obs[f"{op.name}/wrench/force"] = {"data": force, "t": t}
                    obs[f"{op.name}/wrench/torque"] = {"data": torque, "t": t}

        if (
            DataType.TACTILE in self.config.enabled_sensors
            and self._tactile_manager is not None
        ):
            tactile_data = self._tactile_manager.get_data().get("tactile")
            if tactile_data is not None:
                for component, data in self._group_tactile_by_component(
                    tactile_data
                ).items():
                    key_component = (
                        component.replace("_", "/", 1) if structured else component
                    )
                    obs[f"{key_component}/tactile/point_cloud2"] = {
                        "data": data,
                        "t": t,
                    }

        if DataType.CAMERA in self.config.enabled_sensors:
            for cam_name, renderer in self._renderers.items():
                cam_id = self._camera_ids[cam_name]
                spec = self._camera_specs[cam_name]
                renderer.update_scene(
                    self.data,
                    camera=cam_id,
                    scene_option=self._renderer_scene_option,
                )
                renderer.disable_depth_rendering()
                renderer.disable_segmentation_rendering()
                obs_cam_name = (
                    "camera/" + cam_name.split("_")[0] if structured else cam_name
                )
                if spec.enable_color:
                    obs[f"{obs_cam_name}/color/image_raw"] = {
                        "data": np.asarray(renderer.render(), dtype=np.uint8),
                        "t": t,
                    }
                if spec.enable_depth:
                    renderer.enable_depth_rendering()
                    depth = np.asarray(renderer.render(), dtype=np.float32)
                    renderer.disable_depth_rendering()
                    depth[depth > spec.depth_max] = 0.0
                    obs[f"{obs_cam_name}/aligned_depth_to_color/image_raw"] = {
                        "data": depth,
                        "t": t,
                    }
                if spec.enable_mask or spec.enable_heat_map:
                    renderer.enable_segmentation_rendering()
                    segmentation = np.asarray(renderer.render(), dtype=np.int32)
                    renderer.disable_segmentation_rendering()
                    if spec.enable_mask:
                        obs[f"{obs_cam_name}/mask/image_raw"] = {
                            "data": self._build_binary_mask(segmentation),
                            "t": t,
                        }
                    if spec.enable_heat_map:
                        obs[f"{obs_cam_name}/mask/heat_map"] = {
                            "data": self._build_operation_mask(segmentation),
                            "t": t,
                        }

        if structured:
            return {f"/robot/{key}": value for key, value in obs.items()}
        return obs

    def is_updated(self) -> bool:
        current_time = self.data.time
        if self._last_time != current_time:
            self._last_time = current_time
            return True
        return False

    def update(self):
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)
        if self._viewer_running():
            self._sync_viewer()
            if self.config.viewer.step_delay > 0.0:
                time.sleep(self.config.viewer.step_delay)

    def refresh_viewer(self) -> None:
        """Redraw the passive viewer without advancing physics."""
        if self._viewer_running():
            self._sync_viewer()

    def _wrench_from_tactile(
        self, op: OperatorBinding
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._tactile_manager is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        wrenches = self._tactile_manager.get_finger_wrenches()
        force = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        for panel_name, panel_wrench in wrenches.items():
            if not op.tactile_prefixes:
                matches = len(self._operators) == 1
            else:
                matches = any(panel_name.startswith(pfx) for pfx in op.tactile_prefixes)
            if not matches:
                continue
            panel_wrench = np.asarray(panel_wrench, dtype=np.float64).reshape(-1)
            if panel_wrench.shape[0] >= 6:
                force += panel_wrench[:3]
                torque += panel_wrench[3:6]
        return force.astype(np.float32), torque.astype(np.float32)

    def _find_operator_for_tactile_panel(
        self, panel_prefix: str
    ) -> OperatorBinding | None:
        for op in self._operators:
            if not op.tactile_prefixes:
                if len(self._operators) == 1:
                    return op
            elif any(panel_prefix.startswith(pfx) for pfx in op.tactile_prefixes):
                return op
        return None

    def _group_tactile_by_component(
        self, tactile_tensor: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        tactile_tensor = np.asarray(tactile_tensor, dtype=np.float32)
        grouped = {}
        rows = 8
        cols = 5
        max_points = rows * cols
        for i, panel_prefix in enumerate(self._tactile_manager.panel_order):
            op = self._find_operator_for_tactile_panel(panel_prefix)
            if op is None:
                continue
            _, eef_name = self._op_output_names[op.name]
            panel_data = tactile_tensor[i]
            packed_points = np.zeros((max_points, 6), dtype=np.float32)

            for j in range(min(len(panel_data), max_points)):
                row = j // cols
                col = j % cols
                packed_points[j, 0] = col * 0.005
                packed_points[j, 1] = row * 0.005
                packed_points[j, 2] = 0.0
                packed_points[j, 3] = panel_data[j, 0]
                packed_points[j, 4] = panel_data[j, 1]
                packed_points[j, 5] = panel_data[j, 2]

            feats = ("x", "y", "z", "fx", "fy", "fz")
            field_size = 4
            fields = [
                {
                    "name": name,
                    "offset": field_size * idx,
                    "datatype": "FLOAT32",
                    "count": 1,
                }
                for idx, name in enumerate(feats)
            ]

            # Derive a label from the panel prefix (strip trailing underscore).
            panel_label = panel_prefix.rstrip("_")
            key = f"{eef_name}_{panel_label}" if panel_label else eef_name

            sim_time = self.data.time
            sec = int(sim_time)
            nanosec = int((sim_time - sec) * 1e9)
            grouped[key] = {
                "header": {
                    "frame_id": f"{key}_tactile",
                    "stamp": {"sec": sec, "nanosec": nanosec},
                },
                "height": rows,
                "width": cols,
                "fields": fields,
                "is_bigendian": False,
                "point_step": field_size * len(feats),
                "data": packed_points.tobytes(),
                "is_dense": True,
            }

        return dict(grouped)

    def get_info(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)
        info: dict[str, Any] = self.config.model_dump(mode="json", exclude={"cameras"})
        info["cameras"] = {}
        for cam_name, cam_id in self._camera_ids.items():
            spec = self._camera_specs[cam_name]
            fovy_deg = float(self.model.cam_fovy[cam_id])
            fovy_rad = fovy_deg * pi / 180.0
            f = (spec.height / 2.0) / tan(fovy_rad / 2.0)

            camera_info = {
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

            info["cameras"][cam_name] = {
                "camera_info": {
                    stream_type: camera_info for stream_type in ("color", "depth")
                },
                # TODO: should separate extrinsics for color and depth?
                "camera_extrinsics": {
                    "frame": extrinsics_frame,
                    "translation": cam_pos,
                    "rotation_matrix": cam_rot,
                },
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
