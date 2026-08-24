"""Abstract base class for all task runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from auto_atom.framework import TaskFileConfig
from auto_atom.runtime import EnvProtocol, TaskUpdate


class RunnerBase(ABC):
    """Abstract interface shared by all task runners.

    Every runner must support the lifecycle:
    ``__init__() -> from_config(cfg) -> reset()/update() loop -> close()``
    """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Return the number of logical environments managed by the runner."""
        ...

    @abstractmethod
    def from_config(self, config: TaskFileConfig) -> RunnerBase:
        """Initialize the runner from a task-file configuration."""
        ...

    @abstractmethod
    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        """Reset selected environments (all if *env_mask* is ``None``)."""
        ...

    @abstractmethod
    def update(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        """Advance execution by one step for selected environments."""
        ...

    @abstractmethod
    def get_env(self) -> EnvProtocol:
        """Return the underlying environment object managed by this runner."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources held by this runner."""
        ...
