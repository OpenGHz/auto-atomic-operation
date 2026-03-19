import numpy as np

from auto_atom.sim.basis.mujoco_env import (
    CameraSpec,
    DataType,
    EnvConfig,
    UnifiedMujocoEnv,
)


def main():
    env = UnifiedMujocoEnv(
        EnvConfig(
            model_path="third_party/xml/scene_single_arm.xml",
            arm_mode="single",
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
                CameraSpec(
                    name="hand_cam",
                    width=128,
                    height=96,
                    enable_color=False,
                    enable_depth=False,
                    enable_mask=True,
                    enable_heat_map=True,
                )
            ],
            mask_objects=["floor_1_room_main", "left_follower"],
            operations=["grasp", "push", "place", "pull", "rotate"],
        )
    )

    try:
        env.set_interest_objects_and_operations(
            ["floor_1_room_main", "left_follower"], ["grasp", "place"]
        )
        env.reset()

        obs = env.capture_observation()
        mask = obs["hand_cam/mask/image_raw"]["data"]
        heat_map = obs["hand_cam/mask/heat_map"]["data"]
        channel_sum = heat_map.sum(axis=(0, 1))

        print("keys:", sorted(obs.keys()))
        print("mask_shape:", mask.shape)
        print("mask_dtype:", mask.dtype)
        print("mask_sum:", int(mask.sum()))
        print("heat_map_shape:", heat_map.shape)
        print("heat_map_dtype:", heat_map.dtype)
        print("channel_sum:", channel_sum.tolist())
        print("tactile:", obs["arm_eef_left/tactile/point_cloud2"]["data"].keys())

        assert mask.shape == (96, 128)
        assert mask.dtype == np.uint8
        assert int(mask.sum()) > 0
        assert heat_map.shape == (96, 128, len(env.config.operations))
        assert heat_map.dtype == np.uint8
        assert channel_sum[0] > 0, "floor_1_room_main should activate the grasp channel"
        assert channel_sum[2] > 0, "left_follower should activate the place channel"
        assert np.all(channel_sum[[1, 3, 4]] == 0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
