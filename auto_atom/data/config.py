"""Configuration of an episode stream.

A pydantic model, like the other user-written configs: it is built once, so
validation costs nothing that matters, and it rejects an unset seed, unknown
fields, and out-of-range values before a simulator is built. Its JSON dump is
written into every ``Episode.metadata`` as the record of how it was made.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

# Overrides the config owns through its own fields.
_RESERVED_OVERRIDES = {
    "task.seed": "base_seed",
    "env.batch_size": "batch_size",
}


class StreamConfig(BaseModel):
    """What an :class:`~auto_atom.data.stream.EpisodeStream` produces."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", use_attribute_docstrings=True
    )

    task: str
    """Option of the ``task`` config group, as for ``load_task_file_hydra``."""
    base_seed: int
    """Run seed (``task.seed``). Required: ``None`` is rejected, ``0`` is a
    valid seed."""
    overrides: Tuple[str, ...] = ()
    """Further Hydra overrides. ``task.seed`` and ``env.batch_size`` come from
    ``base_seed`` and ``batch_size`` instead."""
    config_dir: Optional[str] = None
    """Hydra config directory; ``None`` is ``<cwd>/aao_configs``."""
    observation_keys: Optional[Tuple[str, ...]] = None
    """Measurement keys to record; ``None`` records every key. Command
    channels (``action/...``) are always recorded. Which keys exist is decided
    when the environment is built (enabled sensors and camera channels)."""
    policy: Literal["demo"] = "demo"
    """Policy that drives the rollouts: ``"demo"`` is the config-driven demo
    policy, the actions ``aao-demo`` takes. Other policies (scripted or
    trained) are not supported yet."""
    max_updates: PositiveInt = 600
    """Control ticks an episode may take before it is truncated."""
    sample_stride: PositiveInt = 1
    """Record every ``sample_stride``-th control tick (and always the last).
    Commands are still issued every tick."""
    batch_size: PositiveInt = 1
    """Environment slots run in one process (``env.batch_size``)."""
    num_episodes: Optional[PositiveInt] = None
    """Episode indices ``0 .. num_episodes - 1`` before sharding; ``None`` runs
    without end."""
    on_invalid: Literal["resample", "keep", "raise"] = "resample"
    """What a failed or truncated episode does: retry its index, be yielded
    with ``success=False``, or raise. A failed reset randomization has no
    episode to keep, so ``keep`` retries it too."""
    retry_budget: NonNegativeInt = 3
    """Retries of one episode index before the stream raises."""
    max_consecutive_invalid: PositiveInt = 20
    """Invalid attempts in a row after which the stream is unhealthy."""
    queue_size: PositiveInt = 16
    """Finished episodes buffered ahead of the consumer."""
    determinism: Literal["episode", "sequential"] = "episode"
    """``episode``: an episode depends only on ``(base_seed, episode_index)``.
    ``sequential``: episode content follows the reset order, as in
    ``aao-eval``."""

    @field_validator("overrides")
    @classmethod
    def _reject_reserved_overrides(cls, overrides: Tuple[str, ...]) -> Tuple[str, ...]:
        for override in overrides:
            key = override.split("=", 1)[0].lstrip("+~")
            field = _RESERVED_OVERRIDES.get(key)
            if field is not None:
                raise ValueError(
                    f"Override {override!r} sets {key}; use StreamConfig.{field}."
                )
        return overrides
