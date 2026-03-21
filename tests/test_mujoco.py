import numpy as np
from auto_atom.basis.mujoco_env import (
    CameraSpec,
    DataType,
    EnvConfig,
    UnifiedMujocoEnv,
)


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
    env = UnifiedMujocoEnv(
        EnvConfig(
            model_path="third_party/xml/scene_pick_place_demo.xml",
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
                cam_spec,
                cam_spec.model_copy(update={"name": "front_cam"}),
                cam_spec.model_copy(update={"name": "side_cam"}),
            ],
            mask_objects=["source_block", "target_pedestal"],
            operations=["pick", "place", "push", "pull", "press"],
        )
    )

    try:
        env.set_interest_objects_and_operations(
            ["source_block", "target_pedestal"], ["pick", "place"]
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
