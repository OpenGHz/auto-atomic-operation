"""Episode-deterministic streams: an episode depends on (seed, episode_index).

On the mock backend: every episode is reset alone under its number, and a
slot's camera-noise frames count only the captures made for it. On MuJoCo:
scene, waypoints, and trajectory follow the episode index across batching and
sharding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pytest

import auto_atom.mock as mock
from auto_atom.data import EpisodeStream, StreamConfig
from auto_atom.policy_eval import ConfigDrivenDemoPolicy
from auto_atom.randomization import RandomizationFailureError
from auto_atom.utils.seed import ResetAddress

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "aao_configs")


def _mock_config(**fields: Any) -> StreamConfig:
    fields.setdefault("task", "policy_eval_mock")
    return StreamConfig(base_seed=0, config_dir=CONFIG_DIR, **fields)


def test_episode_stream_resets_each_episode_alone_under_its_number(
    monkeypatch,
) -> None:
    resets: List[tuple] = []
    original = mock.MockSceneBackend.reset

    def reset(self: Any, env_mask: Any = None) -> None:
        original(self, env_mask)
        resets.append((np.flatnonzero(env_mask).tolist(), self.reset_index))

    monkeypatch.setattr(mock.MockSceneBackend, "reset", reset)

    with EpisodeStream.from_config(
        _mock_config(num_episodes=5, batch_size=2)
    ) as episodes:
        collected = sorted(episodes, key=lambda episode: episode.episode_index)

    assert [e.metadata["reset_index"] for e in collected] == [1, 2, 3, 4, 5]
    assert all(len(slots) == 1 for slots, _ in resets)
    assert sorted(index for _, index in resets) == [1, 2, 3, 4, 5]


def test_a_retried_episode_keeps_its_number(monkeypatch) -> None:
    calls: List[Optional[ResetAddress]] = []
    original = mock.MockSceneBackend.reset

    def reset(self: Any, env_mask: Any = None) -> None:
        calls.append(self._next_reset_address)
        if len(calls) == 1:
            raise RandomizationFailureError(
                target="cup", attempts=2, violations=["collision"], minimum_clearance=0
            )
        original(self, env_mask)

    monkeypatch.setattr(mock.MockSceneBackend, "reset", reset)

    with EpisodeStream.from_config(_mock_config(num_episodes=1)) as episodes:
        (episode,) = list(episodes)

    assert calls == [ResetAddress(1, 0), ResetAddress(1, 1)]
    assert (episode.metadata["reset_index"], episode.metadata["retry"]) == (1, 1)


@pytest.fixture
def stall_slot_one(monkeypatch) -> None:
    """Withhold slot 1's demo action every other tick, so the slots drift apart."""
    act = ConfigDrivenDemoPolicy.act
    calls: List[None] = []

    def stalling_act(self: Any, observation: Any, update: Any, evaluator: Any) -> Any:
        action = act(self, observation, update, evaluator)
        calls.append(None)
        if len(calls) % 2 == 0 and len(action.env_actions) > 1:
            action.env_actions[1] = None
        return action

    monkeypatch.setattr(ConfigDrivenDemoPolicy, "act", stalling_act)


@pytest.mark.parametrize("cheap_commands", [True, False])
def test_a_slot_noise_frames_are_its_own_captures_only(
    monkeypatch, stall_slot_one, cheap_commands: bool
) -> None:
    """Each episode's noise advances once per capture made for it.

    That is its first observation, the observation before each further
    recorded row, and its final observation, however the other slot's resets
    and recorded ticks interleave with it, and whether reading a command takes
    a render-free capture or a full capture.
    """
    frames: Dict[int, int] = {}
    episode_of_slot: Dict[int, int] = {}
    held: Set[int] = set()
    original_reset = mock.MockSceneBackend.reset

    def reset(self: Any, env_mask: Any = None) -> None:
        original_reset(self, env_mask)
        for slot in np.flatnonzero(env_mask):
            episode_of_slot[int(slot)] = self.reset_index
            frames[self.reset_index] = 0

    def hold_camera_noise(self: Any, env_mask: Any) -> None:
        held.update(int(slot) for slot in np.flatnonzero(env_mask))

    original_capture = mock.MockEnv.capture_observation

    def capture_observation(self: Any) -> Dict[str, Any]:
        for slot, episode in episode_of_slot.items():
            if slot not in held:
                frames[episode] += 1
        held.clear()
        return original_capture(self)

    monkeypatch.setattr(mock.MockSceneBackend, "reset", reset)
    monkeypatch.setattr(
        mock.MockEnv, "hold_camera_noise", hold_camera_noise, raising=False
    )
    monkeypatch.setattr(mock.MockEnv, "capture_observation", capture_observation)
    if not cheap_commands:
        monkeypatch.delattr(mock.MockEnv, "capture_commands")

    with EpisodeStream.from_config(
        _mock_config(
            task="mock",
            num_episodes=6,
            batch_size=2,
            sample_stride=2,
            on_invalid="keep",
        ),
    ) as episodes:
        collected = list(episodes)

    assert len({len(episode) for episode in collected}) > 1
    for episode in collected:
        assert frames[episode.metadata["reset_index"]] == len(episode) + 1


# ---------------------------------------------------------------------------
#  End to end: scene, waypoint and trajectory follow the episode index
# ---------------------------------------------------------------------------

# Base and home-EEF randomization of the scene plus waypoint randomization in
# the first stage, drawn on the first tick.
OPEN_DOOR = dict(
    task="open_door",
    base_seed=7,
    config_dir=CONFIG_DIR,
    overrides=[
        "embodiment=airbot_play_g2p",
        "env.viewer=null",
        "observation=rgb_only",
    ],
    max_updates=25,
    on_invalid="keep",
    num_episodes=4,
)


def _open_door(worker: Optional[tuple] = None, **fields: Any) -> Dict[int, Any]:
    stream = EpisodeStream.from_config(StreamConfig(**OPEN_DOOR, **fields))
    if worker is not None:
        stream = stream.shard(*worker)
    with stream:
        return {episode.episode_index: episode for episode in stream}


def _assert_same_episode(left: Any, right: Any) -> None:
    for part in ("base", "eef"):
        np.testing.assert_array_equal(
            left.scene["operators"]["arm"][part]["position"],
            right.scene["operators"]["arm"][part]["position"],
        )
    assert left.randomization["initial_poses"] == right.randomization["initial_poses"]
    assert left.transitions.stage_name.tolist() == right.transitions.stage_name.tolist()
    for key, column in left.transitions.action.items():
        np.testing.assert_array_equal(column, right.transitions.action[key])
    for key, column in left.transitions.obs.items():
        other = right.transitions.obs[key]
        if column.dtype == np.uint8:
            # Native MuJoCo RGB can differ by one level on a few pixels with
            # the frames rendered before it; the state below it is exact.
            assert np.abs(column.astype(int) - other.astype(int)).max() <= 1, key
        elif column.dtype != object:
            np.testing.assert_array_equal(column, other, err_msg=key)


def test_episode_content_follows_its_index_across_batching_and_sharding() -> None:
    sequential = _open_door(determinism="sequential", batch_size=1)
    batched = _open_door(determinism="episode", batch_size=2)
    sharded = _open_door(worker=(1, 2), determinism="episode", batch_size=1)

    assert sorted(batched) == [0, 1, 2, 3]
    assert {episode.metadata["slot"] for episode in batched.values()} == {0, 1}
    assert sorted(sharded) == [1, 3]
    for index in range(4):
        _assert_same_episode(sequential[index], batched[index])
    for index in (1, 3):
        _assert_same_episode(sequential[index], sharded[index])
    bases = {
        tuple(episode.scene["operators"]["arm"]["base"]["position"])
        for episode in sequential.values()
    }
    assert len(bases) == 4
