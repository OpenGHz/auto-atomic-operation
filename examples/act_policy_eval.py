"""把训练好的 lerobot ACT 模型接入 aao 仿真器做闭环评估。

改编自 ``policy_eval_example.py``：循环骨架完全一致，只是把
``RecordedDemoPolicy``（回放录制动作）换成 ``ACTPolicyAdapter``
（每步调用训练好的 ACT 模型出动作）。

数据流（单步）::

    env.capture_observation()                      # aao 仿真观测
      -> 拼 observation.state(10) + 两路 RGB        # 适配成 lerobot 输入
      -> policy.select_action(obs)  -> action(8)    # ACT 推理(内部带动作分块)
      -> 拆成 position(3)+quat(4)+gripper(1)         # 适配回 aao 动作
      -> env.apply_pose_action("arm", ...)          # 驱动仿真

用法（在 auto-atomic-operation 根目录下运行）::

    python examples/act_policy_eval.py \
        --checkpoint /home/richie/airdc/outputs/train/2026-06-03/12-14-11_act/checkpoints/last/pretrained_model \
        --config-name <你采集 aao_pick_data 时用的 task 配置名> \
        --batch-size 1 --num-rollouts 10

注意：
- ``--config-name`` 必须和你采集训练数据时用的 aao task 配置一致，否则
  场景/相机/物体对不上。
- state / action / image 的键与顺序必须和训练配置 (configs/config.yaml) 完全一致，
  见下方 STATE_KEYS / IMAGE_KEYS / 动作切分。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from auto_atom import (
    ExecutionContext,
    PolicyEvaluator,
    TaskUpdate,
    load_task_file_hydra,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTDecoderLayer, ACTEncoderLayer, ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


# --- 必须与训练配置 configs/config.yaml 的 mcap.states / mcap.images 一致 ---
# observation.state = concat 这些键 -> 维度 3+6+1 = 10
STATE_KEYS = [
    "arm/pose/position",            # 3
    "arm/pose/rotation_6d",         # 6
    "gripper/joint_state/position", # 1
]
# observation.images.<lerobot_key> <- 仿真观测里的 color 键
# 左边是仿真观测 key，右边是模型里 image feature 名（训练时由目录名推断）
IMAGE_KEYS = {
    "wrist_cam/color/image_raw": "observation.images.wrist_cam",
    "env0_cam/color/image_raw": "observation.images.env0_cam",
}
# action(8) 的切分，必须与 mcap.actions 顺序一致：position(3)+orientation/quat(4)+gripper(1)
ACT_POS = slice(0, 3)
ACT_QUAT = slice(3, 7)
ACT_GRIP = slice(7, 8)
VERTICAL_QUAT_XYZW = np.asarray([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=np.float32)


def _patch_lerobot_act_attention_no_weights() -> None:
    """ACT does not consume attention weights; skip that fragile CUDA path."""

    if getattr(ACTEncoderLayer, "_aao_no_weights_patch", False):
        return

    def encoder_forward(self, x, pos_embed=None, key_padding_mask=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(
            q,
            k,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

    def decoder_forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x, need_weights=False)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
            need_weights=False,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x

    ACTEncoderLayer.forward = encoder_forward
    ACTDecoderLayer.forward = decoder_forward
    ACTEncoderLayer._aao_no_weights_patch = True
    ACTDecoderLayer._aao_no_weights_patch = True


_patch_lerobot_act_attention_no_weights()


def _normalize_quat_xyzw(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_xyzw_to_matrix(quat: Any) -> np.ndarray:
    x, y, z, w = _normalize_quat_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_angle_deg(q1: Any, q2: Any) -> float:
    a = _normalize_quat_xyzw(q1)
    b = _normalize_quat_xyzw(q2)
    dot = abs(float(np.dot(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _upright_tilt_deg(quat: Any) -> float:
    rot = _quat_xyzw_to_matrix(quat)
    local_z_in_world = rot[:, 2]
    cos_angle = float(np.dot(local_z_in_world, np.asarray([0.0, 0.0, 1.0])))
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _tensor_summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        data = value.detach()
        finite = torch.isfinite(data)
        if finite.any():
            finite_data = data[finite]
            min_val = float(finite_data.min().cpu())
            max_val = float(finite_data.max().cpu())
            mean_val = float(finite_data.float().mean().cpu())
            return (
                f"shape={tuple(data.shape)} dtype={data.dtype} device={data.device} "
                f"min={min_val:.5f} max={max_val:.5f} mean={mean_val:.5f}"
            )
        return f"shape={tuple(data.shape)} dtype={data.dtype} device={data.device} no_finite_values"
    arr = np.asarray(value)
    if arr.size:
        return (
            f"shape={arr.shape} dtype={arr.dtype} "
            f"min={float(np.nanmin(arr)):.5f} max={float(np.nanmax(arr)):.5f} "
            f"mean={float(np.nanmean(arr)):.5f}"
        )
    return f"shape={arr.shape} dtype={arr.dtype} empty"


def _print_section(title: str) -> None:
    print(f"\n== {title} ==")


def _image_keys_from_checkpoint(checkpoint: str) -> dict[str, str]:
    """Infer simulator camera keys from the checkpoint image feature names."""
    config_path = Path(checkpoint) / "config.json"
    with config_path.open(encoding="utf-8") as f:
        checkpoint_config = json.load(f)

    image_keys: dict[str, str] = {}
    for feat_key in checkpoint_config.get("input_features", {}):
        prefix = "observation.images."
        if feat_key.startswith(prefix):
            camera_name = feat_key.removeprefix(prefix)
            image_keys[f"{camera_name}/color/image_raw"] = feat_key
    return image_keys or dict(IMAGE_KEYS)


# ---------------------------------------------------------------------------
# ACT 适配器 policy
# ---------------------------------------------------------------------------


class ACTPolicyAdapter:
    """把 lerobot ACTPolicy 包成 aao PolicyEvaluator 需要的 policy 接口。"""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        debug: bool = False,
        action_debug_every: int = 0,
        replan_every_step: bool = False,
        delta_position_action: bool = False,
        lock_vertical_orientation: bool = False,
        place_xy_from_target: bool = False,
        place_xy_offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint = str(Path(checkpoint).expanduser())
        self.debug = debug
        self.action_debug_every = max(int(action_debug_every), 0)
        self.replan_every_step = replan_every_step
        self.delta_position_action = delta_position_action
        self.lock_vertical_orientation = lock_vertical_orientation
        self.place_xy_from_target = place_xy_from_target
        self.place_xy_offset = np.asarray(place_xy_offset, dtype=np.float32).reshape(2)
        self._printed_debug = False
        self._step = 0
        self._delta_anchor_pos: Optional[np.ndarray] = None
        self.image_keys = _image_keys_from_checkpoint(self.checkpoint)
        policy_cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        policy_cfg.device = self.device
        self.policy = ACTPolicy.from_pretrained(self.checkpoint, config=policy_cfg)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )
        self.policy.eval()
        self.policy.to(self.device)

    def reset(self) -> None:
        # 清空 ACT 内部的动作分块队列，开始新一轮 rollout
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self._delta_anchor_pos = None
        self._step = 0

    def _build_obs(self, observation: dict) -> dict:
        """aao 批量观测 dict -> lerobot 输入 dict。

        observation[key]["data"] 形状：低维 (B, dim)，图像 (B, H, W, 3) uint8。
        """
        # 1) 低维状态拼成 observation.state -> (B, 10)
        state_parts = [
            np.asarray(observation[k]["data"], dtype=np.float32) for k in STATE_KEYS
        ]
        state = np.concatenate(state_parts, axis=-1)
        obs: dict[str, torch.Tensor] = {
            "observation.state": torch.from_numpy(state).to(self.device),
        }
        # 2) 彩色图 (B,H,W,3) uint8 -> (B,3,H,W) float[0,1]
        for sim_key, feat_key in self.image_keys.items():
            img = np.asarray(observation[sim_key]["data"])  # (B,H,W,3) uint8
            t = torch.from_numpy(img).to(self.device).float() / 255.0
            obs[feat_key] = t.permute(0, 3, 1, 2).contiguous()
        return obs

    def check_io(self, observation: dict) -> None:
        """Print a one-frame train/inference IO consistency report."""
        config_path = Path(self.checkpoint) / "config.json"
        with config_path.open(encoding="utf-8") as f:
            checkpoint_config = json.load(f)

        _print_section("Checkpoint Features")
        print("input_features:")
        for key, feature in checkpoint_config["input_features"].items():
            print(f"  {key}: type={feature['type']} shape={tuple(feature['shape'])}")
        print("output_features:")
        for key, feature in checkpoint_config["output_features"].items():
            print(f"  {key}: type={feature['type']} shape={tuple(feature['shape'])}")
        print(f"normalization_mapping: {checkpoint_config.get('normalization_mapping')}")
        print(
            f"chunk_size={checkpoint_config.get('chunk_size')} "
            f"n_action_steps={checkpoint_config.get('n_action_steps')}"
        )

        _print_section("Expected Script Mapping")
        print(f"STATE_KEYS: {STATE_KEYS}")
        print(f"IMAGE_KEYS: {self.image_keys}")
        print("ACTION layout: position[0:3] quat_xyzw[3:7] gripper[7:8]")
        print(f"delta_position_action: {self.delta_position_action}")
        print(f"lock_vertical_orientation: {self.lock_vertical_orientation}")

        _print_section("Raw AAO Observation")
        for key in STATE_KEYS:
            payload = observation[key]["data"]
            print(f"  {key}: {_tensor_summary(payload)}")
        for sim_key in self.image_keys:
            payload = observation[sim_key]["data"]
            print(f"  {sim_key}: {_tensor_summary(payload)}")

        built_obs = self._build_obs(observation)
        _print_section("Model Input Before Preprocessor")
        for key, value in built_obs.items():
            print(f"  {key}: {_tensor_summary(value)}")

        processed_obs = self.preprocessor(dict(built_obs))
        _print_section("Model Input After Preprocessor")
        for key, value in processed_obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {_tensor_summary(value)}")

        missing = [
            key
            for key in checkpoint_config["input_features"]
            if key not in processed_obs
        ]
        extra = [
            key
            for key in processed_obs
            if key.startswith("observation.") and key not in checkpoint_config["input_features"]
        ]
        shape_mismatches = []
        for key, feature in checkpoint_config["input_features"].items():
            value = processed_obs.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            expected = tuple(feature["shape"])
            actual = tuple(value.shape[1:])
            if actual != expected:
                shape_mismatches.append((key, actual, expected))

        _print_section("Feature Check")
        if missing:
            print(f"missing inputs: {missing}")
        else:
            print("missing inputs: none")
        if extra:
            print(f"extra observation inputs: {extra}")
        else:
            print("extra observation inputs: none")
        if shape_mismatches:
            for key, actual, expected in shape_mismatches:
                print(f"shape mismatch: {key}: actual={actual} expected={expected}")
        else:
            print("shape mismatches: none")

        self.policy.reset()
        raw_action = self.policy.select_action(processed_obs)
        _print_section("Raw Policy Action Before Postprocessor")
        print(f"  action: {_tensor_summary(raw_action)}")
        action = self.postprocessor(raw_action)
        _print_section("Action After Postprocessor")
        print(f"  action: {_tensor_summary(action)}")
        action_np = action.detach().cpu().numpy().astype(np.float32)
        pos = action_np[:, ACT_POS]
        quat = action_np[:, ACT_QUAT]
        grip = action_np[:, ACT_GRIP]
        if self.lock_vertical_orientation:
            quat = np.broadcast_to(VERTICAL_QUAT_XYZW, quat.shape).copy()
        quat_norm = np.linalg.norm(quat, axis=-1)
        print(f"  position_or_delta[0]: {pos[0].round(5).tolist()}")
        if self.delta_position_action:
            anchor = np.asarray(
                observation[STATE_KEYS[0]]["data"], dtype=np.float32
            )[:, : pos.shape[-1]]
            abs_pos = anchor + pos
            print(f"  anchor_position[0]: {anchor[0].round(5).tolist()}")
            print(f"  absolute_position[0]: {abs_pos[0].round(5).tolist()}")
        print(f"  quat_xyzw[0]: {quat[0].round(5).tolist()} norm={quat_norm[0]:.5f}")
        print(f"  gripper[0]: {grip[0].round(5).tolist()}")

    @torch.inference_mode()
    def act(
        self, observation: Any, update: TaskUpdate, evaluator: PolicyEvaluator
    ) -> Optional[dict]:
        obs = self._build_obs(observation)
        obs = self.preprocessor(obs)
        current_pos = np.asarray(
            observation[STATE_KEYS[0]]["data"], dtype=np.float32
        )
        queue_empty = True
        if hasattr(self.policy, "_action_queue"):
            queue_empty = len(self.policy._action_queue) == 0
        if self.replan_every_step:
            # ACT 默认会把 n_action_steps 个动作放进队列开环执行。对 waypoint
            # 型任务，逐步重规划更容易从“到上方”切到“下压/闭合”。
            self.policy.reset()
            queue_empty = True
        if self.delta_position_action and (
            queue_empty or self._delta_anchor_pos is None
        ):
            self._delta_anchor_pos = current_pos[:, :3].copy()
        action = self.policy.select_action(obs)        # normalized or raw depending on processor chain
        action = self.postprocessor(action)            # (B, 8), unnormalized policy action
        action = action.detach().cpu().numpy().astype(np.float32)

        pos = action[:, ACT_POS]                        # (B, 3)
        quat = action[:, ACT_QUAT]                      # (B, 4) xyzw
        grip = action[:, ACT_GRIP]                      # (B, 1)
        if self.lock_vertical_orientation:
            quat = np.broadcast_to(VERTICAL_QUAT_XYZW, quat.shape).copy()
        if self.delta_position_action:
            anchor = (
                current_pos[:, :3]
                if self._delta_anchor_pos is None
                else self._delta_anchor_pos
            )
            delta_pos = pos
            pos = anchor + delta_pos
        if self.place_xy_from_target:
            stage_names = list(getattr(update, "stage_name", []))
            try:
                context = evaluator._require_context()
                target = context.backend.get_object_handler("target_pedestal")
                target_pose = target.get_pose() if target is not None else None
            except Exception:  # noqa: BLE001 - optional diagnostic/control path
                target_pose = None
            if target_pose is not None:
                target_xy = np.asarray(target_pose.position, dtype=np.float32)[:, :2]
                for env_index in range(min(pos.shape[0], len(stage_names))):
                    if "place" in stage_names[env_index]:
                        pos[env_index, :2] = target_xy[env_index] + self.place_xy_offset
        # 模型输出的四元数不是单位长度，apply_pose_action 需要单位四元数
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        quat = quat / np.clip(norm, 1e-8, None)
        should_print = self.debug and (
            not self._printed_debug
            or (self.action_debug_every > 0 and self._step % self.action_debug_every == 0)
        )
        if should_print:
            self._printed_debug = True
            print(
                f"[debug] step={self._step} "
                f"pos={pos[0].round(4).tolist()} "
                f"quat={quat[0].round(4).tolist()} "
                f"grip={grip[0].round(4).tolist()}"
                + (
                    f" delta={delta_pos[0].round(4).tolist()} anchor={anchor[0].round(4).tolist()}"
                    if self.delta_position_action
                    else ""
                )
            )
        self._step += 1
        return {"position": pos, "orientation": quat, "gripper": grip}


# ---------------------------------------------------------------------------
# action_applier / observation_getter（与 policy_eval_example.py 相同）
# ---------------------------------------------------------------------------


def action_applier(
    context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
) -> None:
    if action is not None:
        context.backend.env.apply_pose_action(
            "arm",
            action["position"],
            action["orientation"],
            action["gripper"],
            kinematic=False,
        )


def observation_getter(context: ExecutionContext) -> dict:
    return context.backend.env.capture_observation()


def _object_pose(context: ExecutionContext, name: str, env_index: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    handler = context.backend.get_object_handler(name)
    if handler is None:
        return None
    pose = handler.get_pose().select(env_index)
    return (
        np.asarray(pose.position[0], dtype=np.float64),
        np.asarray(pose.orientation[0], dtype=np.float64),
    )


def _trace_rollout_state(
    evaluator: PolicyEvaluator,
    update: TaskUpdate,
    *,
    rollout: int,
    step: int,
    observation: Optional[dict] = None,
    last_action: Optional[dict],
) -> None:
    context = evaluator._require_context()
    env_index = 0
    operator = context.backend.get_operator_handler("arm")
    eef_pose = operator.get_end_effector_pose().select(env_index)
    eef_pos = np.asarray(eef_pose.position[0], dtype=np.float64)
    eef_quat = np.asarray(eef_pose.orientation[0], dtype=np.float64)

    source = _object_pose(context, "source_block", env_index)
    target = _object_pose(context, "target_pedestal", env_index)
    grip_text = ""
    if observation is not None and "gripper/joint_state/position" in observation:
        grip_obs = np.asarray(
            observation["gripper/joint_state/position"]["data"], dtype=np.float64
        )
        if grip_obs.ndim >= 2 and grip_obs.shape[0] > env_index:
            grip_text = f" grip_obs={grip_obs[env_index].round(4).tolist()}"
    source_text = "source=missing"
    if source is not None:
        source_pos, source_quat = source
        world_delta = source_pos - eef_pos
        eef_delta = _quat_xyzw_to_matrix(eef_quat).T @ world_delta
        try:
            target_handler = context.backend.get_object_handler("source_block")
            grasp = operator._check_grasp_conditions(env_index, target_handler)
        except Exception as exc:  # noqa: BLE001 - debug path should not stop rollout
            grasp = {"error": type(exc).__name__}
        source_text = (
            f"source_pos={source_pos.round(4).tolist()} "
            f"source_tilt={_upright_tilt_deg(source_quat):.2f}deg "
            f"src-eef_world={world_delta.round(4).tolist()} "
            f"src_in_eef={eef_delta.round(4).tolist()} "
            f"grasp={grasp}"
        )

    target_text = "target=missing"
    if target is not None:
        target_pos, _ = target
        target_text = f"target_pos={target_pos.round(4).tolist()}"

    action_text = ""
    if last_action is not None:
        cmd_pos = np.asarray(last_action["position"][env_index], dtype=np.float64)
        cmd_quat = np.asarray(last_action["orientation"][env_index], dtype=np.float64)
        cmd_grip = np.asarray(last_action["gripper"][env_index], dtype=np.float64)
        action_text = (
            f" cmd_pos={cmd_pos.round(4).tolist()} "
            f"cmd_quat_err={_quat_angle_deg(cmd_quat, VERTICAL_QUAT_XYZW):.3f}deg"
            f" cmd_grip={cmd_grip.round(4).tolist()}"
        )

    print(
        f"[trace] rollout={rollout} step={step} "
        f"stage={int(update.stage_index[env_index])}:{update.stage_name[env_index]} "
        f"status={update.status[env_index]} "
        f"eef_pos={eef_pos.round(4).tolist()} "
        f"eef_quat_err={_quat_angle_deg(eef_quat, VERTICAL_QUAT_XYZW):.3f}deg "
        f"{source_text} {target_text}{grip_text}{action_text}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ACT policy 在 aao 仿真里闭环评估")
    parser.add_argument("--checkpoint", required=True, help="pretrained_model 目录路径")
    parser.add_argument("--config-name", required=True, help="aao task 配置名(与采集时一致)")
    parser.add_argument("--batch-size", type=int, default=1, help="并行环境数")
    parser.add_argument("--num-rollouts", type=int, default=10, help="跑多少轮评估")
    parser.add_argument("--max-steps", type=int, default=400, help="每轮最大步数")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="额外 Hydra override，可重复传，例如 --override +env=gl",
    )
    parser.add_argument("--debug-action", action="store_true", help="打印模型动作")
    parser.add_argument("--action-debug-every", type=int, default=0, help="每 N 步打印一次动作")
    parser.add_argument(
        "--check-io",
        action="store_true",
        help="检查训练/推理输入输出 key、shape、dtype、范围和归一化链",
    )
    parser.add_argument(
        "--check-io-only",
        action="store_true",
        help="只做 IO 检查，不执行 rollout",
    )
    parser.add_argument(
        "--replan-every-step",
        action="store_true",
        help="每步清空 ACT 动作队列并基于当前观测重新预测",
    )
    parser.add_argument(
        "--delta-position-action",
        action="store_true",
        help="把模型输出的 position[0:3] 当作相对查询时刻末端位置的 delta，再还原成绝对位置执行",
    )
    parser.add_argument(
        "--lock-vertical-orientation",
        action="store_true",
        help="忽略模型输出的 quat，强制使用训练数据里的竖直末端姿态 [0,sqrt(0.5),0,sqrt(0.5)]",
    )
    parser.add_argument(
        "--place-xy-from-target",
        action="store_true",
        help="诊断/仿真用：放置阶段把执行 XY 对准 target_pedestal，Z 和夹爪仍用模型输出",
    )
    parser.add_argument(
        "--place-xy-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DX", "DY"),
        help="配合 --place-xy-from-target 使用的目标台 XY 偏移，单位米",
    )
    parser.add_argument(
        "--trace-grasp",
        action="store_true",
        help="打印抓取/放置诊断：末端、方块、目标台位姿和抓取侧向误差",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=20,
        help="开启 --trace-grasp 时每 N 步打印一次诊断，阶段切换时总会打印",
    )
    parser.add_argument(
        "--sim-loop-frequency", type=float, default=10.0, help="from_config 的仿真频率"
    )
    args = parser.parse_args()

    overrides = [f"env.batch_size={args.batch_size}", *args.override]
    task_file = load_task_file_hydra(args.config_name, overrides=overrides)
    policy = ACTPolicyAdapter(
        args.checkpoint,
        debug=args.debug_action,
        action_debug_every=args.action_debug_every,
        replan_every_step=args.replan_every_step,
        delta_position_action=args.delta_position_action,
        lock_vertical_orientation=args.lock_vertical_orientation,
        place_xy_from_target=args.place_xy_from_target,
        place_xy_offset=tuple(args.place_xy_offset),
    )
    evaluator = PolicyEvaluator(
        action_applier=action_applier,
        observation_getter=observation_getter,
    ).from_config(task_file, args.sim_loop_frequency)

    successes = []
    try:
        if args.check_io or args.check_io_only:
            policy.reset()
            evaluator.reset()
            obs = evaluator.get_observation()
            policy.check_io(obs)
            if args.check_io_only:
                return

        for rollout in range(args.num_rollouts):
            policy.reset()
            update = evaluator.reset()
            step = -1
            prev_stage = None
            last_action = None
            for step in range(args.max_steps):
                obs = evaluator.get_observation()
                action = policy.act(obs, update=update, evaluator=evaluator)
                last_action = action
                if args.trace_grasp:
                    stage_key = (
                        int(update.stage_index[0]),
                        update.stage_name[0],
                        str(update.status[0]),
                    )
                    trace_every = max(int(args.trace_every), 1)
                    if prev_stage != stage_key or step % trace_every == 0:
                        _trace_rollout_state(
                            evaluator,
                            update,
                            rollout=rollout,
                            step=step,
                            observation=obs,
                            last_action=last_action,
                        )
                    prev_stage = stage_key
                update = evaluator.update(action)
                if update.done.all():
                    if args.trace_grasp:
                        _trace_rollout_state(
                            evaluator,
                            update,
                            rollout=rollout,
                            step=step,
                            observation=obs,
                            last_action=last_action,
                        )
                    break
            summary = evaluator.summarize(
                update, max_updates=args.max_steps, updates_used=step + 1
            )
            ok = list(summary.final_success)
            successes.extend(ok)
            print(
                f"[rollout {rollout}] steps={step + 1} "
                f"completed_stages={summary.completed_stage_count} success={ok}"
            )
        if successes:
            rate = sum(bool(s) for s in successes) / len(successes)
            print(f"\n成功率: {sum(bool(s) for s in successes)}/{len(successes)} = {rate:.1%}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
