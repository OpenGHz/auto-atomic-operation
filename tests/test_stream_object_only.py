"""Object-only streams record the carried object's pose and carry as actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

from auto_atom.data import Episode, stream

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "aao_configs")


def object_only_episodes(*extra: str, num_episodes: int = 1) -> List[Episode]:
    """rack_plate without a robot: a pick carries ``object`` to the rack."""
    with stream(
        "rack_plate",
        base_seed=0,
        config_dir=CONFIG_DIR,
        overrides=["env.viewer=null", "execution=object_only", *extra],
        num_episodes=num_episodes,
        max_updates=400,
        on_invalid="keep",
    ) as episodes:
        return sorted(episodes, key=lambda episode: episode.episode_index)


def assert_transport_channels(episode: Any) -> None:
    obs, actions = episode.transitions.obs, episode.transitions.action
    assert {"object/pose/position", "object/pose/orientation", "object/carried"} <= set(
        obs
    )
    assert set(actions) == {
        "action/object/pose/position",
        "action/object/pose/orientation",
        "action/object/carried",
    }
    # The pick acquires the object and the place releases it: the carried
    # command switches on once and off once.
    carried = actions["action/object/carried"][:, 0]
    switches = np.flatnonzero(np.diff(carried))
    assert carried[0] == 0.0 and carried[-1] == 0.0
    assert len(switches) == 2
    np.testing.assert_array_equal(obs["object/carried"][1:, 0], carried[:-1])
    # Transport is kinematic, so each commanded pose is where the object is
    # observed next.
    for axis in ("position", "orientation"):
        following = np.concatenate(
            [
                obs[f"object/pose/{axis}"][1:],
                episode.final_observation[f"object/pose/{axis}"][None],
            ]
        )
        np.testing.assert_allclose(
            actions[f"action/object/pose/{axis}"], following, atol=1e-6
        )
    assert (
        np.linalg.norm(
            episode.final_observation["object/pose/position"]
            - obs["object/pose/position"][0]
        )
        > 0.05
    )


def test_object_only_actions_are_the_carried_object_commands() -> None:
    (episode,) = object_only_episodes()

    assert episode.success
    assert_transport_channels(episode)


def test_an_uncommanded_object_holds_its_pose_after_a_reset() -> None:
    first, second = object_only_episodes(num_episodes=2)

    # Episode 1 left the object on the rack. Episode 2 starts with no command
    # yet, so its first commanded pose is where the reset put the object, not
    # where episode 1 placed it.
    obs, actions = second.transitions.obs, second.transitions.action
    np.testing.assert_array_equal(
        actions["action/object/pose/position"][0], obs["object/pose/position"][0]
    )
    assert not np.allclose(
        actions["action/object/pose/position"][0],
        first.final_observation["object/pose/position"],
    )
