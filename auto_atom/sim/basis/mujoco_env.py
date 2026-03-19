from enum import Enum
from math import tan, pi
from pathlib import Path
from typing import Any, Dict, List, Set
from pydantic import BaseModel, ConfigDict, Field, model_validator
from auto_atom.utils.transformations import euler_from_matrix
import os
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
    width: int = 640
    height: int = 480
    enable_color: bool = True
    enable_depth: bool = True
    enable_mask: bool = False
    enable_heat_map: bool = False


class EnvConfig(BaseModel, frozen=True):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    model_path: Path
    arm_mode: str = "single"
    enabled_sensors: Set[DataType] = Field(default_factory=set)
    cameras: List[CameraSpec] = Field(default_factory=list)
    mask_objects: List[str] = Field(default_factory=list)
    operations: List[str] = Field(default_factory=list)
    stamp_ns: bool = True

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


class UnifiedMujocoEnv:
    def __init__(self, config: EnvConfig):
        self.get_logger().info("Initializing...")
        self.config = config
        self.model, self.data = self._load_model(config.model_path)

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._components = (
            ["arm"] if config.arm_mode == "single" else ["left_arm", "right_arm"]
        )
        self._prefix = {"arm": "", "left_arm": "left_", "right_arm": "right_"}

        self._camera_specs = {c.name: c for c in config.cameras}
        self._renderers: Dict[str, mujoco.Renderer] = {}
        self._camera_ids = {}
        self._interest_object_operations: dict[str, str] = {}
        self._mask_object_pairs = self._build_mask_object_pairs(config.mask_objects)

        logger = self.get_logger()

        if DataType.CAMERA in config.enabled_sensors:
            logger.info(f"Setting up cameras: {list(self._camera_specs.keys())}")
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

        self._imu_ids = {}
        self._pose_ids = {}
        self._pose_site_ids = {}
        self._pose_validated_components: set[str] = set()
        self._wrench_ids = {}
        for comp in self._components:
            prefix = self._prefix[comp]
            self._imu_ids[comp] = {
                "acc": self._sensor_id(f"{prefix}imu_acc"),
                "gyro": self._sensor_id(f"{prefix}imu_gyro"),
                "quat": self._sensor_id(f"{prefix}imu_quat"),
            }
            self._pose_ids[comp] = {
                "pos": self._sensor_id(f"{prefix}global_gripper_pos"),
                "quat": self._sensor_id(f"{prefix}global_gripper_quat"),
            }
            pose_site_name = f"{prefix}eef_pose" if prefix else "eef_pose"
            self._pose_site_ids[comp] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, pose_site_name
            )
            self._wrench_ids[comp] = {
                "force": self._sensor_id(f"{prefix}eef_force"),
                "torque": self._sensor_id(f"{prefix}eef_torque"),
            }

        self._tactile_manager = None
        if (
            DataType.TACTILE in config.enabled_sensors
            or DataType.WRENCH in config.enabled_sensors
        ):
            self._init_tactile_manager()
        self._last_time = None

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
        try:
            import importlib
            import sys

            tactile_dir = Path(__file__).resolve().parents[1] / "tactile"
            if str(tactile_dir) not in sys.path:
                sys.path.insert(0, str(tactile_dir))
            tactile_module = importlib.import_module("tactile_sensor")
            tactile_manager_cls = getattr(tactile_module, "TactileSensorManager")

            self._tactile_manager = tactile_manager_cls(
                self.model,
                self.data,
                enable=DataType.TACTILE in self.config.enabled_sensors,
            )
        except Exception:
            self._tactile_manager = None

    def _sensor_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(sid)

    def _sensor_data(self, sensor_id: int) -> np.ndarray:
        if sensor_id < 0:
            return np.zeros((0,), dtype=np.float32)
        idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return np.asarray(self.data.sensordata[idx : idx + dim], dtype=np.float32)

    def _component_q_indices(self, component: str) -> np.ndarray:
        if component == "arm":
            return np.arange(min(7, self.model.nq), dtype=np.int32)

        prefix = self._prefix[component]
        indices = []
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if name.startswith(prefix):
                qpos_adr = int(self.model.jnt_qposadr[jid])
                indices.append(qpos_adr)
        indices = sorted(set(indices))
        if not indices:
            return np.arange(min(7, self.model.nq), dtype=np.int32)
        return np.asarray(indices[:7], dtype=np.int32)

    def _component_actuator_indices(self, component: str) -> np.ndarray:
        prefix = self._prefix[component]
        if component == "arm":
            return np.arange(min(7, self.model.nu), dtype=np.int32)

        indices = []
        for aid in range(self.model.nu):
            name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            )
            if name.startswith(prefix):
                indices.append(aid)
        if not indices:
            return np.arange(min(7, self.model.nu), dtype=np.int32)
        return np.asarray(indices[:7], dtype=np.int32)

    def _component_output_names(self, component: str) -> tuple[str, str]:
        if component == "arm":
            return "arm", "eef"
        if component == "left_arm":
            return "left_arm", "left_eef"
        if component == "right_arm":
            return "right_arm", "right_eef"
        return component, f"{component}_eef"

    def _split_component_joint_state_indices(
        self, component: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actuator_indices = self._component_actuator_indices(component)
        arm_actuator_indices = []
        eef_actuator_indices = []

        for actuator_idx in actuator_indices:
            actuator_name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator_idx)
                )
                or ""
            )
            if "fingers" in actuator_name:
                eef_actuator_indices.append(int(actuator_idx))
            else:
                arm_actuator_indices.append(int(actuator_idx))

        def _joint_and_velocity_indices(
            selected_actuator_indices: list[int],
        ) -> tuple[np.ndarray, np.ndarray]:
            joint_indices = []
            velocity_indices = []
            for actuator_idx in selected_actuator_indices:
                joint_id = int(self.model.actuator_trnid[actuator_idx, 0])
                if joint_id < 0:
                    continue
                joint_indices.append(int(self.model.jnt_qposadr[joint_id]))
                velocity_indices.append(int(self.model.jnt_dofadr[joint_id]))
            return (
                np.asarray(joint_indices, dtype=np.int32),
                np.asarray(velocity_indices, dtype=np.int32),
            )

        arm_qidx, arm_vidx = _joint_and_velocity_indices(arm_actuator_indices)
        eef_qidx, eef_vidx = _joint_and_velocity_indices(eef_actuator_indices)

        return (
            arm_qidx,
            eef_qidx,
            arm_vidx,
            eef_vidx,
            np.asarray(arm_actuator_indices, dtype=np.int32),
            np.asarray(eef_actuator_indices, dtype=np.int32),
        )

    def _pose_rot9d(self, component: str) -> np.ndarray:
        site_id = self._pose_site_ids.get(component, -1)
        if site_id < 0:
            raise ValueError(f"No pose site found for component '{component}'")
        return self.data.site_xmat[site_id]

    def _mujoco_quat_wxyz_to_xyzw(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32).reshape(-1)
        return quat[[1, 2, 3, 0]]

    def _rotmat_to_quat_xyzw(self, rotmat: np.ndarray) -> np.ndarray:
        quat_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_wxyz, np.asarray(rotmat, dtype=np.float64).reshape(9))
        return self._mujoco_quat_wxyz_to_xyzw(quat_wxyz).astype(np.float32)

    def _site_pose(self, component: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = self._pose_site_ids.get(component, -1)
        if site_id < 0:
            raise ValueError(f"No pose site found for component '{component}'")
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
        self, component: str, pos: np.ndarray, quat_xyzw: np.ndarray
    ) -> None:
        if component in self._pose_validated_components:
            return

        site_pos, site_quat_xyzw = self._site_pose(component)
        if not np.allclose(pos, site_pos, atol=1e-5) or not self._quats_equivalent_xyzw(
            quat_xyzw, site_quat_xyzw, atol=1e-5
        ):
            raise ValueError(
                "Pose sensor does not match pose site for component "
                f"'{component}': sensor_pos={pos.tolist()}, site_pos={site_pos.tolist()}, "
                f"sensor_quat_xyzw={quat_xyzw.tolist()}, "
                f"site_quat_xyzw={site_quat_xyzw.tolist()}"
            )

        self._pose_validated_components.add(component)

    def _rotmat_to_euler_xyz(self, rotmat: np.ndarray) -> np.ndarray:
        rotmat = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
        sy = np.sqrt(rotmat[0, 0] ** 2 + rotmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(rotmat[2, 1], rotmat[2, 2])
            pitch = np.arctan2(-rotmat[2, 0], sy)
            yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
        else:
            roll = np.arctan2(-rotmat[1, 2], rotmat[1, 1])
            pitch = np.arctan2(-rotmat[2, 0], sy)
            yaw = 0.0

        return np.asarray([roll, pitch, yaw], dtype=np.float32)

    def _rotmat_to_rotation_6d(self, rotmat: np.ndarray) -> np.ndarray:
        rotmat = np.asarray(rotmat, dtype=np.float32).reshape(3, 3)
        return rotmat[:, :2].reshape(-1).astype(np.float32)

    def reset(self) -> None:
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

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
        t = (
            float(self.data.time)
            if not self.config.stamp_ns
            else int(self.data.time * 1e9)
        )
        obs: dict[str, dict[str, Any]] = {}

        for component in self._components:
            arm_qidx, eef_qidx, arm_vidx, eef_vidx, arm_aidx, eef_aidx = (
                self._split_component_joint_state_indices(component)
            )
            arm_name, eef_name = self._component_output_names(component)

            if DataType.JOINT_POSITION in self.config.enabled_sensors:
                obs[f"{arm_name}/joint_state/position"] = {
                    "data": np.asarray(self.data.qpos[arm_qidx], dtype=np.float32),
                    "t": t,
                }
                obs[f"{eef_name}/joint_state/position"] = {
                    "data": np.asarray(self.data.qpos[eef_qidx], dtype=np.float32),
                    "t": t,
                }
                obs[f"action/{arm_name}/joint_state/position"] = {
                    "data": np.asarray(self.data.ctrl[arm_aidx], dtype=np.float32),
                    "t": t,
                }
                obs[f"action/{eef_name}/joint_state/position"] = {
                    "data": np.asarray(self.data.ctrl[eef_aidx], dtype=np.float32),
                    "t": t,
                }
            if DataType.JOINT_VELOCITY in self.config.enabled_sensors:
                obs[f"{arm_name}/joint_state/velocity"] = {
                    "data": np.asarray(self.data.qvel[arm_vidx], dtype=np.float32),
                    "t": t,
                }
                obs[f"{eef_name}/joint_state/velocity"] = {
                    "data": np.asarray(self.data.qvel[eef_vidx], dtype=np.float32),
                    "t": t,
                }
            if DataType.JOINT_EFFORT in self.config.enabled_sensors:
                obs[f"{arm_name}/joint_state/effort"] = {
                    "data": np.asarray(self.data.ctrl[arm_aidx], dtype=np.float32),
                    "t": t,
                }
                obs[f"{eef_name}/joint_state/effort"] = {
                    "data": np.asarray(self.data.ctrl[eef_aidx], dtype=np.float32),
                    "t": t,
                }

            if DataType.POSE in self.config.enabled_sensors:
                pos = self._sensor_data(self._pose_ids[component]["pos"])
                quat = self._mujoco_quat_wxyz_to_xyzw(
                    self._sensor_data(self._pose_ids[component]["quat"])
                )
                rot9d = self._pose_rot9d(component)
                self._validate_pose_sensor_matches_site(component, pos, quat)
                obs[f"{component}/pose/position"] = {
                    "data": pos.astype(np.float32),
                    "t": t,
                }
                obs[f"{component}/pose/orientation"] = {
                    "data": quat.astype(np.float32),
                    "t": t,
                }
                obs[f"{component}/pose/rotation"] = {
                    "data": euler_from_matrix(rot9d.reshape(3, 3)),
                    "t": t,
                }
                obs[f"{component}/pose/rotation_6d"] = {
                    "data": rot9d[:6].astype(np.float32),
                    "t": t,
                }

            if DataType.IMU in self.config.enabled_sensors:
                acc = self._sensor_data(self._imu_ids[component]["acc"])
                gyro = self._sensor_data(self._imu_ids[component]["gyro"])
                quat = self._sensor_data(self._imu_ids[component]["quat"])
                obs[f"{component}/imu/linear_acceleration"] = {
                    "data": acc,
                    "t": t,
                }
                obs[f"{component}/imu/angular_velocity"] = {
                    "data": gyro,
                    "t": t,
                }
                obs[f"{component}/imu/orientation"] = {
                    "data": quat,
                    "t": t,
                }

            if DataType.WRENCH in self.config.enabled_sensors:
                force = self._sensor_data(self._wrench_ids[component]["force"])
                torque = self._sensor_data(self._wrench_ids[component]["torque"])
                if force.size == 0 or torque.size == 0:
                    force, torque = self._wrench_from_tactile(component)
                obs[f"{component}/wrench/force"] = {
                    "data": np.asarray(force, dtype=np.float32),
                    "t": t,
                }
                obs[f"{component}/wrench/torque"] = {
                    "data": np.asarray(torque, dtype=np.float32),
                    "t": t,
                }

        if (
            DataType.TACTILE in self.config.enabled_sensors
            and self._tactile_manager is not None
        ):
            tactile_data = self._tactile_manager.get_data().get("tactile")
            if tactile_data is not None:
                for component, data in self._group_tactile_by_component(
                    tactile_data
                ).items():
                    obs[f"{component}/tactile/point_cloud_raw"] = {
                        "data": data,
                        "t": t,
                    }

        if DataType.CAMERA in self.config.enabled_sensors:
            for cam_name, renderer in self._renderers.items():
                cam_id = self._camera_ids[cam_name]
                spec = self._camera_specs[cam_name]
                renderer.update_scene(self.data, camera=cam_id)
                renderer.disable_depth_rendering()
                renderer.disable_segmentation_rendering()
                if spec.enable_color:
                    obs[f"{cam_name}/color/image_raw"] = {
                        "data": np.asarray(renderer.render(), dtype=np.uint8),
                        "t": t,
                    }
                if spec.enable_depth:
                    renderer.enable_depth_rendering()
                    depth = renderer.render()
                    renderer.disable_depth_rendering()
                    obs[f"{cam_name}/aligned_depth_to_color/image_raw"] = {
                        "data": np.asarray(depth, dtype=np.float32),
                        "t": t,
                    }
                if spec.enable_mask or spec.enable_heat_map:
                    renderer.enable_segmentation_rendering()
                    segmentation = np.asarray(renderer.render(), dtype=np.int32)
                    renderer.disable_segmentation_rendering()
                    if spec.enable_mask:
                        obs[f"{cam_name}/mask/image_raw"] = {
                            "data": self._build_binary_mask(segmentation),
                            "t": t,
                        }
                    if spec.enable_heat_map:
                        obs[f"{cam_name}/mask/heat_map"] = {
                            "data": self._build_operation_mask(segmentation),
                            "t": t,
                        }

        return obs

    def is_updated(self) -> bool:
        current_time = self.data.time
        if self._last_time != current_time:
            self._last_time = current_time
            return True
        return False

    def update(self):
        mujoco.mj_step(self.model, self.data)

    def _wrench_from_tactile(self, component: str) -> tuple[np.ndarray, np.ndarray]:
        if self._tactile_manager is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        wrenches = self._tactile_manager.get_finger_wrenches()
        force = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        for panel_name, panel_wrench in wrenches.items():
            if component == "arm":
                matches = True
            elif component == "left_arm":
                matches = panel_name.startswith("left_")
            else:
                matches = panel_name.startswith("right_")
            if not matches:
                continue
            panel_wrench = np.asarray(panel_wrench, dtype=np.float64).reshape(-1)
            if panel_wrench.shape[0] >= 6:
                force += panel_wrench[:3]
                torque += panel_wrench[3:6]
        return force.astype(np.float32), torque.astype(np.float32)

    def _group_tactile_by_component(
        self, tactile_tensor: np.ndarray
    ) -> dict[str, np.ndarray]:
        tactile_tensor = np.asarray(tactile_tensor, dtype=np.float32)
        if tactile_tensor.ndim != 3 or self._tactile_manager is None:
            return {}

        grouped: dict[str, list[np.ndarray]] = {k: [] for k in self._components}
        for i, panel_name in enumerate(self._tactile_manager.panel_order):
            if i >= tactile_tensor.shape[0]:
                break
            panel = tactile_tensor[i]
            if "left_" in panel_name:
                comp = "left_arm" if "left_arm" in grouped else "arm"
            elif "right_" in panel_name:
                comp = "right_arm" if "right_arm" in grouped else "arm"
            else:
                comp = "arm"
            grouped.setdefault(comp, []).append(panel)

        out = {}
        for comp, blocks in grouped.items():
            if blocks:
                out[comp] = np.concatenate(blocks, axis=0)
        return out

    def get_info(self) -> dict[str, Any]:
        mujoco.mj_forward(self.model, self.data)
        info: dict[str, Any] = {
            "model_path": str(self.config.model_path),
            "arm_mode": self.config.arm_mode,
            "enabled_sensors": [s.value for s in self.config.enabled_sensors],
            "cameras": {},
        }

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

            cam_rot = np.asarray(self.data.cam_xmat[cam_id]).reshape(3, 3)
            cam_pos = np.asarray(self.data.cam_xpos[cam_id])

            info["cameras"][cam_name] = {
                "camera_info": {
                    stream_type: camera_info for stream_type in ("color", "depth")
                },
                # TODO: should separate extrinsics for color and depth?
                "camera_extrinsics": {
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

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)
