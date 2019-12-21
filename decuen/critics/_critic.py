"""Interfaces for arbitrary critics and respective settings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from decuen.structs import Experience
from decuen.utils.context import Contextful


@dataclass
class CriticSettings:
    """Basic common settings for all critics."""

    discount_factor: float


# pylint: disable=too-few-public-methods
class Critic(ABC, Contextful):
    """Generic abstract critic interface.

    Note that this abstraction is more general that what might normally be viewed as a critic in modern literature: we
    generalize the critic to be able to either generate a state-value or an action-value; as such the interface for
    criticising is delegated to the more specialized interfaces of `StateCritic` and `ActionCritic` and those interfaces
    should be used for subclassing or most other purposes.

    This abstraction provides an interface for one general critic functionality common to both state- and action-value
    based critics: the ability to learn based on past transitions and trajectories to improve critical accuracy.
    """

    settings: CriticSettings
    _learn_step: int

    @abstractmethod
    def __init__(self, settings: CriticSettings) -> None:
        """Initialize this generic critic interface."""
        super().__init__()
        self.settings = settings
        self._learn_step = 0

    # TODO: support learning from trajectories
    # XXX: possibly return loss or some other metric?
    @abstractmethod
    def learn(self, experience: Experience) -> None:
        """Update internal critic representation based on a past experience."""
        ...

    @abstractmethod
    def advantage(self, experience: Experience) -> List[float]:
        """Estimate the advantage of every step in an experience."""
        ...
