"""Low-level MuJoCo environment wrapper.

``MujocoBasis`` owns the MuJoCo model/data lifecycle, operator index
resolution, sensor queries, rendering, viewer management and physics
stepping.  It deliberately does **not** provide ``step(action)`` or
``capture_observation()`` — those higher-level concepts live in the
``UnifiedMujocoEnv`` subclass defined in ``mujoco_env.py``.
"""

import copy
import logging
import time
from enum import Enum
from math import pi, tan
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import mujoco
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    field_serializer,
    field_validator,
    model_validator,
)

from auto_atom.basis.mjc.tactile.tactile_sensor import TactileSensorManager


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
    is_static: bool = False
    """Whether this camera's GS background is rendered once and cached (static)
    or re-rendered every frame (dynamic).  Set to ``False`` for moving cameras
    such as hand-mounted cameras whose viewpoint changes each timestep."""

    @property
    def has_native_output(self) -> bool:
        """Whether any native MuJoCo render output is requested for this camera.

        When False (e.g. every channel has been reassigned to GS rendering) the
        env can skip allocating a ``mujoco.Renderer`` for this camera entirely.
        """
        return (
            self.enable_color
            or self.enable_depth
            or self.enable_mask
            or self.enable_heat_map
        )


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

    name: str = ""
    """Logical operator name, auto-populated from the ``EnvConfig.operators``
    dict key.  Can be left empty in YAML; the model validator fills it in."""

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

    root_body: str = ""
    """Root body name for the operator's base frame.
    When non-empty, triggers auto-registration in ``UnifiedMujocoEnv.__init__``."""

    mocap_body: str = ""
    """Mocap body name for non-joint-mode operators."""

    freejoint: str = ""
    """Freejoint name for mocap operators."""

    eef_mapper: Optional[Any] = None
    """Optional mapper that remaps EEF observation and control values.

    Instantiate via Hydra ``_target_``.  Must provide:

    - ``bind(model, data)`` — called once after model load to resolve
      geom/joint indices, build lookup tables, etc.
    - ``obs_map(model, data, raw: np.ndarray) -> np.ndarray`` — forward
      map from raw joint qpos / ctrl to user space (e.g. finger distance).
    - ``ctrl_map(model, data, user: np.ndarray) -> np.ndarray`` — inverse
      map from user space back to actuator ctrl values.

    When ``None`` (default), raw qpos / ctrl values are used as-is.
    """

    @field_serializer("eef_mapper")
    @classmethod
    def _serialize_eef_mapper(cls, v: Any, _info: Any) -> Optional[str]:
        if v is None:
            return None
        return f"{type(v).__module__}.{type(v).__qualname__}"

    ik_factory: Optional[ImportString] = None
    """Import path to IK solver class/callable.  Called as
    ``ik_factory(model=model, arm_joint_names=names, **ik_params)``."""

    ik_params: Dict[str, Any] = Field(default_factory=dict)
    """Solver-specific keyword arguments passed to *ik_factory*."""


class EnvConfig(BaseModel, frozen=True):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    name: str = ""
    """Optional registry name. When set, the constructed batched env self-registers under this name."""
    model_path: Path
    """Path to the scene XML. If ``robot_paths`` is non-empty, the scene is
    composed by injecting each robot XML as an ``<include>`` sibling under
    ``<mujoco>`` at load time; otherwise the scene file is loaded as-is."""
    robot_paths: List[Path] = Field(default_factory=list)
    """Optional robot XMLs to compose with ``model_path``. Leave empty when
    ``model_path`` already embeds its robot (legacy monolithic scenes)."""
    operators: Dict[str, OperatorBinding] = Field(default_factory=dict)
    """Operator definitions keyed by logical name, mapping to XML actuators and sensors."""
    enabled_sensors: Set[DataType] = Field(default_factory=set)
    """The sensor categories that should be exposed in captured observations."""
    cameras: List[CameraSpec] = Field(default_factory=list)
    """The camera specifications to initialize when camera output is enabled."""
    mask_objects: List[str] = Field(default_factory=list)
    """The object names that are eligible for binary mask and heat-map generation."""
    operations: List[str] = Field(default_factory=list)
    """The operation names corresponding to the mask objects."""
    heatmap_operations: List[str] = Field(default_factory=list)
    """The operation names used to assign channels in generated heat maps."""
    stamp_ns: bool = True
    """Whether observation timestamps should be emitted in nanoseconds instead of seconds."""
    sim_freq: float | None = None
    """Physics simulation frequency in Hz. If None, uses the timestep defined in the XML model."""
    update_freq: float | None = None
    """Control update frequency in Hz. Must be <= sim_freq. If None, defaults to sim_freq (n_substeps=1)."""
    ctrl_interpolation: bool = False
    """Linearly interpolate ctrl across substeps when n_substeps > 1 to prevent PD overshoot."""
    initial_joint_positions: Dict[str, float | List[float]] = Field(
        default_factory=dict
    )
    """Joint name → qpos value overrides applied after every reset (after the
    keyframe). Use a scalar for 1-DOF joints (slide/hinge); use a list for
    multi-DOF joints — 4 values for ball joints (quat wxyz), 7 values for free
    joints (pos xyz + quat wxyz). Multi-DOF entries are written *after* the
    parallel-linkage settle loop so the weld/equality drift does not
    override them."""
    viewer: ViewerConfig | None = None
    """Viewer configuration. If None, the passive viewer is not launched."""
    structured: bool = False
    """Whether the observation value data should be flattened to 1D arrays when possible, e.g. for joint states."""
    batch_size: int = 1
    """Number of homogeneous env replicas to construct for batched execution."""
    viewer_env_index: int = 0
    """Which env replica owns the viewer when ``batch_size > 1``."""
    pre_step_callbacks: List[Any] = Field(default_factory=list)
    """Pre-step callback objects invoked before every ``mj_step()``.

    Each entry should be a callable with signature ``cb(model, data)``.
    If the object has a ``bind(model, data)`` method, it is called once
    during env initialization to resolve joint indices, etc.

    In YAML, specify each callback with a Hydra ``_target_`` pointing to a
    class whose ``__init__`` accepts only configuration parameters (no
    ``model``/``data``).  The framework calls ``bind()`` after loading the
    model.
    """
    interests: Tuple[List[str], List[str]] = ([], [])
    """A tuple of two lists: the first list contains the names of interest objects, and the second list contains the names of interest operations."""

    @model_validator(mode="after")
    def populate_operator_names(self):
        for key, binding in self.operators.items():
            if not binding.name:
                self.operators[key] = binding.model_copy(update={"name": key})
            elif binding.name != key:
                raise ValueError(
                    f"Operator key '{key}' does not match binding name '{binding.name}'"
                )
        return self

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
        enable_heat_map = False
        enable_mask = False
        for cam_cfg in self.cameras:
            if cam_cfg.enable_heat_map:
                enable_heat_map = True
            if cam_cfg.enable_mask:
                enable_mask = True
            if enable_mask and enable_heat_map:
                break
        if enable_heat_map:
            if not self.heatmap_operations:
                self.heatmap_operations.extend(self.operations)
            for field in ("operations", "mask_objects"):
                if field not in self.model_fields_set:
                    raise ValueError(f"{field} must be set when enable_heat_map")
        if enable_mask and not self.mask_objects:
            raise ValueError("mask_objects must be set when enable_mask")
        return self

    @model_validator(mode="after")
    def validate_interests(self):
        if not self.interests[0]:
            self.interests[0].extend(self.mask_objects)
        if not self.interests[1]:
            self.interests[1].extend(self.operations)
        object_names, operation_names = self.interests
        if len(operation_names) == 1 and len(object_names) > 1:
            operation_names[:] = operation_names * len(object_names)
        return self

    @field_validator("viewer")
    @classmethod
    def validate_viewer(cls, v: ViewerConfig | None) -> ViewerConfig | None:
        if v is not None and v.disable:
            return None
        return v

    @model_validator(mode="after")
    def validate_batch(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if self.viewer_env_index < 0 or self.viewer_env_index >= self.batch_size:
            raise ValueError(
                f"viewer_env_index ({self.viewer_env_index}) must be in [0, {self.batch_size})"
            )
        return self


class MujocoBasis:
    """Low-level MuJoCo wrapper: model/data access, rendering, physics."""

    def __init__(self, config: Optional[EnvConfig] = None, **kwargs):
        self.get_logger().info("Initializing...")
        if config is None:
            config = EnvConfig.model_validate(kwargs)
        self.config = config
        self._info = None
        self.model, self.data = self._load_model(config.model_path, config.robot_paths)
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
        self.reset()

        self._viewer = None
        if config.viewer is not None:
            self._launch_viewer()

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    def _launch_viewer(self) -> None:
        import mujoco.viewer as _mj_viewer

        self._viewer = _mj_viewer.launch_passive(self.model, self.data)
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
        model_path: Path, robot_paths: List[Path] | None = None
    ) -> tuple[Any, Any]:
        from auto_atom.utils.scene_loader import load_scene

        model = load_scene(model_path, robot_paths or [])
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
        qpos_widths = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
        multi_dof_entries: list[tuple[int, np.ndarray]] = []
        pin_addrs: list[int] = []
        for joint_name, value in self.config.initial_joint_positions.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                continue
            addr = int(self.model.jnt_qposadr[jid])
            width = qpos_widths[int(self.model.jnt_type[jid])]
            if isinstance(value, (list, tuple)):
                arr = np.asarray(value, dtype=float)
                if arr.size != width:
                    raise ValueError(
                        f"initial_joint_positions['{joint_name}'] has {arr.size} "
                        f"values but joint has {width} qpos slots"
                    )
                multi_dof_entries.append((addr, arr))
            else:
                if width != 1:
                    raise ValueError(
                        f"initial_joint_positions['{joint_name}'] is scalar but joint "
                        f"has {width} qpos slots; use a list to set all slots"
                    )
                self.data.qpos[addr] = value
                pin_addrs.append(addr)
        if pin_addrs and self.model.neq > 0:
            # Equality constraints (parallel linkage grippers, etc.) are only
            # resolved during mj_step.  Pin the configured joints and step so
            # passive joints settle to a constraint-consistent state.
            for op in self._operators.values():
                for aidx in [
                    self._op_arm_aidx[op.name],
                    self._op_eef_aidx[op.name],
                ]:
                    for ai in aidx:
                        ji = self.model.actuator_trnid[ai, 0]
                        if ji >= 0:
                            self.data.ctrl[ai] = self.data.qpos[
                                self.model.jnt_qposadr[ji]
                            ]
            saved_gravity = self.model.opt.gravity.copy()
            self.model.opt.gravity[:] = 0
            target = self.data.qpos[pin_addrs].copy()
            # Pin all free-joint qpos during settle so bodies with a freejoint
            # (objects on the table, mocap-driven arm bases) do not drift from
            # contact-solver repulsion or residual constraint forces.
            free_addrs: list[int] = []
            for j in range(self.model.njnt):
                if int(self.model.jnt_type[j]) == 0:  # mjJNT_FREE
                    a = int(self.model.jnt_qposadr[j])
                    free_addrs.extend(range(a, a + 7))
            free_snapshot = self.data.qpos[free_addrs].copy() if free_addrs else None
            for _ in range(500):
                mujoco.mj_step(self.model, self.data)
                self.data.qpos[pin_addrs] = target
                if free_snapshot is not None:
                    self.data.qpos[free_addrs] = free_snapshot
            self.data.qvel[:] = 0.0
            self.model.opt.gravity[:] = saved_gravity
        for addr, arr in multi_dof_entries:
            self.data.qpos[addr : addr + arr.size] = arr
        mujoco.mj_forward(self.model, self.data)
        self._sync_mocap_to_freejoint()
        self._prev_ctrl = None

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
            self._sync_viewer()
            if self.config.viewer.step_delay > 0.0:
                time.sleep(self.config.viewer.step_delay)

    def is_updated(self) -> bool:
        current_time = self.data.time
        if self._last_time != current_time:
            self._last_time = current_time
            return True
        return False

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
            info["cameras"][cam_name] = {
                "camera_info": {
                    stream_type: camera_info[cam_name]
                    for stream_type in ("color", "depth")
                },
                # TODO: should separate extrinsics for color and depth?
                "camera_extrinsics": camera_extrinsics[cam_name],
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
