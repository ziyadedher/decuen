"""Q-value based critics.

Contains abstractions for Q-value based critics.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import MutableSequence

import numpy as np  # type: ignore

from decuen.critics._critic import ActionCritic, ActionCriticSettings
from decuen.memories._memory import Transition


@dataclass
class QCriticSettings(ActionCriticSettings):
    """Q-value based critic settings."""

    target_update: int = 1
    double: bool = False


class QCritic(ActionCritic):
    """Abstract Q-value based critic to be used for building implementations of Q-value based critics."""

    def __init__(self, settings: QCriticSettings = QCriticSettings()) -> None:
        """Initialize an abstract Q-value based critic."""
        super().__init__(settings)

    @abstractmethod
    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Update internal critic representation based on past transitions."""
        ...

    @abstractmethod
    def crit(self, state: np.ndarray, action: np.ndarray) -> float:
        """Return a metric of 'goodness' of taking an action in a state."""
        ...

    @abstractmethod
    def values(self, state: np.ndarray) -> np.ndarray:
        """Return an array of Q-values of all actions in a specific state."""
        ...
