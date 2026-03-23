import numpy as np
from auto_atom.basis.mujoco_env import (
    CameraSpec,
    DataType,
    EnvConfig,
    OperatorBinding,
    UnifiedMujocoEnv,
)
from pprint import pprint


def main():
    cam_spec = CameraSpec(
        name="hand_cam",
        width=1280,
        height=720,
        enable_color=False,
        enable_depth=False,
        enable_mask=True,
        enable_heat_map=True,
    )
    flatten_like = False
    env = UnifiedMujocoEnv(
        EnvConfig(
            model_path="assets/xmls/scenes/pick_and_place/demo.xml",
            operators=[
                OperatorBinding(
                    name="arm",
                    arm_actuators=[
                        "act_root_x",
                        "act_root_y",
                        "act_root_z",
                        "act_root_rot_z",
                        "act_root_rot_y",
                        "act_root_rot_x",
                    ],
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
                # DataType.IMU
            ],
            cameras=[
                cam_spec.model_copy(update={"parent_frame": "eef_pose"}),
                cam_spec.model_copy(update={"name": "front_cam"}),
                cam_spec.model_copy(update={"name": "side_cam"}),
            ],
            mask_objects=["source_block", "target_pedestal"],
            operations=["pick", "place", "push", "pull", "press"],
            structured=not flatten_like,
        )
    )

    try:
        env.set_interest_objects_and_operations(
            ["source_block", "target_pedestal"], ["pick", "place"]
        )
        env.reset()

        info = env.get_info()
        # pprint("Info:")
        # pprint(info)

        obs = env.capture_observation()
        print("keys:", sorted(obs.keys()))

        cam_name = "hand_cam" if flatten_like else "camera/hand"
        cam_name = "/robot/" + cam_name
        mask = obs[cam_name + "/mask/image_raw"]["data"]
        heat_map = obs[cam_name + "/mask/heat_map"]["data"]
        channel_sum = heat_map.sum(axis=(0, 1))

        print("mask_shape:", mask.shape)
        print("mask_dtype:", mask.dtype)
        print("mask_sum:", int(mask.sum()))
        print("heat_map_shape:", heat_map.shape)
        print("heat_map_dtype:", heat_map.dtype)
        print("channel_sum:", channel_sum.tolist())
        tactile_key = (
            "eef_left/tactile/point_cloud2"
            if flatten_like
            else "/robot/eef/left/tactile/point_cloud2"
        )
        print("tactile:", obs[tactile_key]["data"].keys())
        # print("arm/joint_state/position:", obs["arm/joint_state/position"]["data"])

        assert mask.shape == (720, 1280)
        assert mask.dtype == np.uint8
        assert int(mask.sum()) > 0
        assert heat_map.shape == (720, 1280, len(env.config.operations))
        assert heat_map.dtype == np.uint8
        assert channel_sum[0] > 0, "source_block should activate the pick channel"
        assert channel_sum[1] > 0, "target_pedestal should activate the place channel"
        assert np.all(channel_sum[[2, 3, 4]] == 0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
