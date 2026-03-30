import numpy as np

from auto_atom.basis.mjc.mujoco_env import (
    BatchedUnifiedMujocoEnv,
    CameraSpec,
    DataType,
    EnvConfig,
    OperatorBinding,
)


def main():
    cam_spec = CameraSpec(
        name="hand_cam",
        width=1280,
        height=720,
        enable_color=True,
        enable_depth=True,
        enable_mask=True,
        enable_heat_map=True,
    )
    env = BatchedUnifiedMujocoEnv(
        EnvConfig(
            model_path="assets/xmls/scenes/pick_and_place/demo.xml",
            batch_size=2,
            operators=[
                OperatorBinding(
                    name="arm",
                    arm_actuators=[],
                    eef_actuators=["fingers_actuator"],
                    eef_output_name="eef",
                    pose_site="eef_pose",
                    imu_acc="imu_acc",
                    imu_gyro="imu_gyro",
                    imu_quat="imu_quat",
                    pose_sensor_pos="global_gripper_pos",
                    pose_sensor_quat="global_gripper_quat",
                    tactile_prefixes=["left_", "right_"],
                )
            ],
            enabled_sensors=[
                DataType.CAMERA,
                DataType.JOINT_POSITION,
                DataType.JOINT_VELOCITY,
                DataType.JOINT_EFFORT,
                DataType.POSE,
                DataType.TACTILE,
            ],
            cameras=[
                cam_spec.model_copy(update={"parent_frame": "eef_pose"}),
                cam_spec.model_copy(update={"name": "front_cam"}),
                cam_spec.model_copy(update={"name": "side_cam"}),
            ],
            mask_objects=["source_block", "target_pedestal"],
            operations=["pick", "place", "push", "pull", "press"],
            structured=True,
        )
    )

    try:
        env.set_interest_objects_and_operations(
            ["source_block", "target_pedestal"], ["pick", "place"]
        )
        env.reset()
        obs = env.capture_observation()
        cam_name = "/robot/camera/hand"
        mask = obs[cam_name + "/mask/image_raw"]["data"]
        heat_map = obs[cam_name + "/mask/heat_map"]["data"]
        assert mask.shape == (2, 720, 1280)
        assert mask.dtype == np.uint8
        assert heat_map.shape == (2, 720, 1280, len(env.config.operations))
        assert heat_map.dtype == np.uint8
        tactile = obs["/robot/eef/left/tactile/point_cloud2"]["data"]
        assert tactile["data"].shape[0] == 2
        pose = obs["/robot/arm/pose"]["data"]
        assert pose["position"].shape == (2, 3)
        assert pose["orientation"].shape == (2, 4)
    finally:
        env.close()


if __name__ == "__main__":
    main()
