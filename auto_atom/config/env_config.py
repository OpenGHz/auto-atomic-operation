"""Simulation-agnostic environment / sensor configuration models.

These pydantic models describe how a simulated environment is configured —
cameras, sensor noise, viewer, operator bindings, and the top-level
``EnvConfig``. They are intentionally free of any physics/renderer import so
every simulation backend (native MuJoCo, Gaussian-Splatting rendering, …)
shares one configuration surface.

Split out of the former ``auto_atom.basis.mjc.mujoco_basis`` monolith.
"""

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)

from auto_atom.scene_composition import SceneConfig


class DataType(str, Enum):
    CAMERA = "camera"
    IMU = "imu"
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_EFFORT = "joint_effort"
    TACTILE = "tactile"
    WRENCH = "wrench"
    POSE = "pose"


class CameraExtrinsicsConfig(BaseModel, frozen=True):
    """Optional camera pose overrides expressed in the camera's parent frame."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    position: Tuple[float, float, float] | None = None
    """Camera position ``[x, y, z]`` in metres relative to ``parent_frame``."""
    orientation: Tuple[float, float, float, float] | None = None
    """Camera orientation quaternion ``[x, y, z, w]`` relative to ``parent_frame``."""

    @model_validator(mode="after")
    def validate_orientation(self) -> "CameraExtrinsicsConfig":
        if self.position is None and self.orientation is None:
            raise ValueError("camera extrinsics must set position or orientation")
        if self.orientation is not None:
            if any(not math.isfinite(value) for value in self.orientation):
                raise ValueError("camera extrinsics orientation must be finite")
            if math.fsum(value * value for value in self.orientation) <= 1.0e-24:
                raise ValueError("camera extrinsics orientation must be non-zero")
        if self.position is not None and any(
            not math.isfinite(value) for value in self.position
        ):
            raise ValueError("camera extrinsics position must be finite")
        return self


class CameraCalibrationConfig(BaseModel, frozen=True):
    """YAML-owned projection and extrinsic calibration for one camera."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    projection: Literal["mujoco_perspective"] = "mujoco_perspective"
    """Projection model supported by the native MuJoCo and GS adapters."""
    fovy_deg: PositiveFloat | None = None
    """Vertical field of view in degrees; omitted preserves the XML value."""
    extrinsics: CameraExtrinsicsConfig | None = None
    """Optional pose override relative to the camera's ``parent_frame``."""

    @model_validator(mode="after")
    def validate_fovy(self) -> "CameraCalibrationConfig":
        if self.fovy_deg is not None and self.fovy_deg >= 180.0:
            raise ValueError("camera calibration fovy_deg must be less than 180")
        return self


class RGBNoiseConfig(BaseModel, frozen=True):
    """Sensor-noise parameters applied to an RGB camera stream."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    gaussian_std: NonNegativeFloat = 0.0
    """Gaussian readout-noise standard deviation in normalized RGB units."""
    shot_noise_scale: PositiveFloat | None = None
    """Optional Poisson shot-noise scale in normalized RGB units."""
    dropout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    """Independent per-pixel dropout probability."""
    exposure: NonNegativeFloat = 1.0
    """Static exposure multiplier applied before RGB noise."""
    gain: NonNegativeFloat = 1.0
    """Static sensor gain multiplier applied before RGB noise."""
    quantization_levels: PositiveInt | None = None
    """Optional number of normalized RGB intervals used for quantization."""
    illumination_std: NonNegativeFloat = 0.0
    """Per-capture low-frequency illumination multiplier standard deviation."""
    gain_std: NonNegativeFloat = 0.0
    """Per-capture low-frequency gain multiplier standard deviation."""
    distribution: "NoiseDistributionConfig" = Field(
        default_factory=lambda: NoiseDistributionConfig()
    )
    """Distribution used for additive Gaussian or heavy-tailed RGB noise."""
    temporal: "TemporalNoiseConfig" = Field(
        default_factory=lambda: TemporalNoiseConfig()
    )
    """Low-cost frame-level temporal jitter, AR(1), and drift parameters."""


class NoiseDistributionConfig(BaseModel, frozen=True):
    """Distribution parameters shared by one RGB or depth stream."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: Literal["gaussian", "student_t"] = "gaussian"
    """Additive noise distribution."""
    degrees_of_freedom: PositiveFloat = 5.0
    """Student-t degrees of freedom; values above two have finite variance."""

    @model_validator(mode="after")
    def validate_degrees_of_freedom(self) -> "NoiseDistributionConfig":
        if self.kind == "student_t" and self.degrees_of_freedom <= 2.0:
            raise ValueError(
                "student_t degrees_of_freedom must be greater than 2 for finite variance"
            )
        return self


class TemporalNoiseConfig(BaseModel, frozen=True):
    """Low-cost frame-level temporal correlation parameters."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    jitter_std: NonNegativeFloat = 0.0
    """Stationary standard deviation of the frame-level AR(1) offset."""
    ar1_coefficient: float = Field(default=0.0, ge=-1.0, le=1.0)
    """AR(1) coefficient for the frame-level offset."""
    drift_std: NonNegativeFloat = 0.0
    """Standard deviation of each slowly changing drift innovation."""
    drift_decay: float = Field(default=1.0, ge=0.0, le=1.0)
    """Retention factor applied to the previous drift value."""


class DepthNoiseConfig(BaseModel, frozen=True):
    """Sensor-noise parameters applied to a metric depth stream."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    gaussian_std_m: NonNegativeFloat = 0.0
    """Fixed Gaussian depth-noise standard deviation in metres."""
    relative_std: NonNegativeFloat = 0.0
    """Distance-proportional depth-noise coefficient."""
    bias_m: float = 0.0
    """Signed depth-bias coefficient in metres, scaled by ``distance**bias_distance_power``."""
    bias_distance_power: NonNegativeFloat = 0.0
    """Exponent used to scale ``bias_m`` with the original measured distance."""
    scale: PositiveFloat = 1.0
    """Multiplicative depth scale applied before additive noise."""
    resolution_reference: Tuple[PositiveInt, PositiveInt] | None = None
    """Reference ``[width, height]`` for resolution-scaled depth errors."""
    resolution_power: float = 0.0
    """Exponent for ``(reference_pixels / actual_pixels)`` applied to depth errors."""
    quantization_step_m: PositiveFloat | None = None
    """Optional positive depth quantization step in metres."""
    invalid_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("invalid_probability", "dropout_probability"),
    )
    """Independent probability of invalidating a valid depth pixel."""
    distribution: NoiseDistributionConfig = Field(
        default_factory=NoiseDistributionConfig
    )
    """Distribution used for additive depth noise."""
    temporal: TemporalNoiseConfig = Field(default_factory=TemporalNoiseConfig)
    """Low-cost frame-level temporal jitter, AR(1), and drift parameters."""

    @model_validator(mode="after")
    def validate_bias(self) -> "DepthNoiseConfig":
        if not math.isfinite(self.bias_m):
            raise ValueError("depth bias_m must be finite")
        if not math.isfinite(self.resolution_power):
            raise ValueError("depth resolution_power must be finite")
        return self


class CameraNoiseConfig(BaseModel, frozen=True):
    """Independent RGB and depth sensor-noise configuration for one camera."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    rgb: RGBNoiseConfig | None = None
    """RGB noise configuration; omitted to leave RGB unchanged."""
    depth: DepthNoiseConfig | None = None
    """Depth noise configuration; omitted to leave depth unchanged."""


# The RGB model is declared before the shared nested models for readability of
# the public camera configuration.  Resolve its forward references once all
# nested models are available so YAML dictionaries validate normally.
RGBNoiseConfig.model_rebuild()


class CameraSpec(BaseModel, frozen=True):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    name: str
    """The camera name as defined in the Mujoco model."""
    role: Literal["scene", "operator"] = "scene"
    """Semantic ownership used to remove operator-mounted cameras together
    with an operator-only scene layer."""
    width: int = 640
    """The rendered image width in pixels."""
    height: int = 480
    """The rendered image height in pixels."""
    enable_color: bool = True
    """Whether to include RGB images in the captured observation."""
    enable_depth: bool = True
    """Whether to include depth images in the captured observation."""
    calibration: CameraCalibrationConfig | None = None
    """Optional YAML-owned projection and extrinsic calibration."""
    rgb_clip_range_m: Tuple[PositiveFloat, PositiveFloat] | None = None
    """RGB and semantic-mask clip range ``[near, far]`` in metres.

    MuJoCo stores near/far values as scene-extent multipliers globally on the
    model. When this is set, the backend temporarily converts the range to
    those multipliers around the RGB and semantic-mask pass, then restores the
    previous model values. ``None`` preserves the XML scene's native clipping
    behavior.
    """
    depth_clip_range_m: Tuple[PositiveFloat, PositiveFloat] | None = None
    """Depth clip range ``[near, far]`` in metres.

    This range is independent from ``rgb_clip_range_m`` so the simulated depth
    sensor can model a stricter measurable range than its aligned RGB stream.
    ``None`` preserves the XML scene's native clipping behavior.
    """
    noise: CameraNoiseConfig | None = None
    """Optional independent RGB and depth sensor-noise configuration."""
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

    @model_validator(mode="after")
    def validate_clip_ranges(self) -> "CameraSpec":
        for stream, clip_range in (
            ("rgb", self.rgb_clip_range_m),
            ("depth", self.depth_clip_range_m),
        ):
            if clip_range is None:
                continue
            near_m, far_m = clip_range
            if near_m >= far_m:
                raise ValueError(
                    f"{stream}_clip_range_m must satisfy near < far, "
                    f"got [{near_m}, {far_m}]"
                )
        return self


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
    """Seconds to sleep after each rendered update step. TaskRunner skips this
    delay while coalescing internal viewer updates at public boundaries."""
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
    model_config = ConfigDict(
        validate_assignment=True,
        use_attribute_docstrings=True,
        extra="forbid",
    )

    name: str = ""
    """Optional registry name. When set, the constructed batched env self-registers under this name."""
    scene: SceneConfig
    """Generic host scene and ordered composition layers."""
    operators: Dict[str, OperatorBinding] = Field(default_factory=dict)
    """Operator definitions keyed by logical name, mapping to XML actuators and sensors."""
    enabled_sensors: Set[DataType] = Field(default_factory=set)
    """The sensor categories that should be exposed in captured observations."""
    cameras: List[CameraSpec] = Field(default_factory=list)
    """The camera specifications to initialize when camera output is enabled."""
    hide_operators_in_camera: bool = False
    """Hide configured operators from native MuJoCo camera rendering.

    Geoms below every operator's ``root_body`` and ``mocap_body`` are removed
    from the offscreen scene used for RGB, depth, masks, and heat maps.  The
    MuJoCo model remains unchanged, so physics, contacts, tactile sensing,
    control, and the passive viewer are unaffected.
    """
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
    """Scene-level joint qpos overrides applied after every reset.

    Use ``null`` in a task override to clear inherited defaults when all
    operator-owned joints are declared under ``task_operators``. A scalar is
    used for a 1-DOF joint; a list supplies all qpos slots for a ball or
    freejoint. Operator-scoped values are applied later through each
    operator's ``initial_state.joint_positions``.
    """

    @field_validator("initial_joint_positions", mode="before")
    @classmethod
    def _normalize_initial_joint_positions(cls, value: object) -> object:
        """Treat Hydra ``null`` as an intentional clear of inherited defaults."""
        return {} if value is None else value

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
