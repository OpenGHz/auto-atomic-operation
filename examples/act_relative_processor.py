"""LeRobot processor steps for AIRDC ACT relative pose training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.utils.constants import ACTION, OBS_STATE


def _normalize_quat_xyzw(quat: Tensor, eps: float = 1e-8) -> Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate_xyzw(quat: Tensor) -> Tensor:
    out = quat.clone()
    out[..., :3] = -out[..., :3]
    return out


def quat_mul_xyzw(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product for xyzw quaternions."""
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def matrix_to_quat_xyzw(matrix: Tensor) -> Tensor:
    """Convert rotation matrices (..., 3, 3) to xyzw unit quaternions."""
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))

    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    quat = torch.stack((qx, qy, qz, qw), dim=-1)
    return _normalize_quat_xyzw(quat)


def rot6d_to_quat_xyzw(rot6d: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert Zhou-style 6D rotation columns to xyzw quaternions."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=eps)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    matrix = torch.stack((b1, b2, b3), dim=-1)
    return matrix_to_quat_xyzw(matrix)


@ProcessorStepRegistry.register("airdc_act_chunk_relative_pose")
@dataclass
class ACTChunkRelativePoseProcessorStep(ProcessorStep):
    """Convert ACT action chunks to the relative-pose target used by AAO eval.

    The LeRobot ACT dataset reader expands `action` into shape
    `(B, chunk_size, action_dim)`. This step keeps the chunk as a whole and
    anchors every future action to the current `observation.state`, matching
    `third_party/auto-atomic-operation/examples/act_policy_eval.py`:

    - `delta_position = action.position - state.position`
    - `delta_quat = inverse(state_quat) * action_quat`

    Gripper dimensions are left unchanged.
    """

    enabled: bool = True
    position_slice: tuple[int, int] = (0, 3)
    quat_slice: tuple[int, int] = (3, 7)
    state_position_slice: tuple[int, int] = (0, 3)
    state_rot6d_slice: tuple[int, int] = (3, 9)
    relative_position: bool = True
    relative_orientation: bool = True

    def __post_init__(self) -> None:
        # JSON configs deserialize tuples as lists. Keep runtime slicing stable.
        self.position_slice = tuple(self.position_slice)
        self.quat_slice = tuple(self.quat_slice)
        self.state_position_slice = tuple(self.state_position_slice)
        self.state_rot6d_slice = tuple(self.state_rot6d_slice)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        action = transition.get(TransitionKey.ACTION)
        if state is None or action is None:
            return transition

        if action.ndim not in (2, 3):
            raise ValueError(
                f"ACTChunkRelativePoseProcessorStep expects action rank 2 or 3, got {tuple(action.shape)}"
            )
        if state.ndim != 2:
            raise ValueError(
                f"ACTChunkRelativePoseProcessorStep expects state rank 2, got {tuple(state.shape)}"
            )

        action = action.clone()
        ps0, ps1 = self.position_slice
        ss0, ss1 = self.state_position_slice
        if self.relative_position:
            anchor_pos = state[..., ss0:ss1].to(device=action.device, dtype=action.dtype)
            if action.ndim == 3:
                anchor_pos = anchor_pos.unsqueeze(1)
            action[..., ps0:ps1] = action[..., ps0:ps1] - anchor_pos

        if self.relative_orientation:
            qs0, qs1 = self.quat_slice
            rs0, rs1 = self.state_rot6d_slice
            anchor_quat = rot6d_to_quat_xyzw(
                state[..., rs0:rs1].to(device=action.device, dtype=action.dtype)
            )
            action_quat = _normalize_quat_xyzw(action[..., qs0:qs1])
            inv_anchor = quat_conjugate_xyzw(anchor_quat)
            if action.ndim == 3:
                inv_anchor = inv_anchor.unsqueeze(1)
            action[..., qs0:qs1] = quat_mul_xyzw(inv_anchor, action_quat)

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "position_slice": self.position_slice,
            "quat_slice": self.quat_slice,
            "state_position_slice": self.state_position_slice,
            "state_rot6d_slice": self.state_rot6d_slice,
            "relative_position": self.relative_position,
            "relative_orientation": self.relative_orientation,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _get_valid_chunk_starts(episode_indices: np.ndarray, chunk_size: int) -> np.ndarray:
    total = len(episode_indices)
    if total < chunk_size:
        return np.array([], dtype=np.int64)
    max_start = total - chunk_size
    starts = np.arange(max_start + 1)
    return starts[episode_indices[starts] == episode_indices[starts + chunk_size - 1]]


def _relative_pose_chunk_batch(
    start_indices: np.ndarray,
    all_actions: np.ndarray,
    all_states: np.ndarray,
    chunk_size: int,
    processor: ACTChunkRelativePoseProcessorStep,
) -> np.ndarray:
    if len(start_indices) == 0:
        return np.empty((0, all_actions.shape[1]), dtype=np.float32)

    offsets = np.arange(chunk_size)
    frame_idx = start_indices[:, None] + offsets[None, :]
    action = torch.from_numpy(all_actions[frame_idx].astype(np.float32, copy=True))
    state = torch.from_numpy(all_states[start_indices].astype(np.float32, copy=False))
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
    }
    out = processor(transition)[TransitionKey.ACTION]
    return out.reshape(-1, all_actions.shape[1]).detach().cpu().numpy().astype(np.float32)


def compute_act_relative_pose_action_stats(
    hf_dataset: Any,
    *,
    chunk_size: int,
    relative_position: bool = True,
    relative_orientation: bool = True,
    num_workers: int = 0,
    processor: ACTChunkRelativePoseProcessorStep | None = None,
) -> dict[str, np.ndarray] | None:
    """Compute action normalizer stats for AIRDC ACT relative-pose training.

    Dataset actions remain absolute. This function mirrors the preprocessor's
    optional position/orientation delta conversion over every valid ACT chunk so
    the saved normalizer statistics match what the model actually sees.
    """
    if not relative_position and not relative_orientation:
        return None

    if processor is None:
        processor = ACTChunkRelativePoseProcessorStep(
            relative_position=relative_position,
            relative_orientation=relative_orientation,
        )

    all_actions = np.asarray(hf_dataset[ACTION], dtype=np.float32)
    all_states = np.asarray(hf_dataset[OBS_STATE], dtype=np.float32)
    episode_indices = np.asarray(hf_dataset["episode_index"])
    valid_starts = _get_valid_chunk_starts(episode_indices, chunk_size)
    if len(valid_starts) == 0:
        raise RuntimeError(
            f"No valid chunks found for relative pose stats \n"
            f"(total_frames={len(episode_indices)}, chunk_size={chunk_size})."
        )

    batch_size = 50_000
    batches = [valid_starts[i : i + batch_size] for i in range(0, len(valid_starts), batch_size)]
    running_stats = RunningQuantileStats()

    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(
                    _relative_pose_chunk_batch,
                    batch,
                    all_actions,
                    all_states,
                    chunk_size,
                    processor,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                running_stats.update(future.result())
    else:
        for batch in batches:
            running_stats.update(
                _relative_pose_chunk_batch(batch, all_actions, all_states, chunk_size, processor)
            )

    return running_stats.get_statistics()
