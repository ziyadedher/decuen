"""Q-value based critics.

Contains abstractions for Q-value based critics.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import MutableSequence

from decuen.critics._critic import ActionCritic, ActionCriticSettings
from decuen.structs import Action, State, Tensor, Transition


@dataclass
class QCriticSettings(ActionCriticSettings):
    """Q-value based critic settings."""

    target_update: int
    double: bool


class QCritic(ActionCritic):
    """Abstract Q-value based critic to be used for building implementations of Q-value based critics."""

    def __init__(self, settings: QCriticSettings) -> None:
        """Initialize an abstract Q-value based critic."""
        super().__init__(settings)

    @abstractmethod
    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Update internal critic representation based on past transitions."""
        ...

    @abstractmethod
    def crit(self, state: State, action: Action) -> Tensor:
        """Return the quality of taking an action or tensor of actions in a state."""
        ...
