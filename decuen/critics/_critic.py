"""Interfaces for arbitrary critics and respective settings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import MutableSequence

import numpy as np  # type: ignore
from gym import Space  # type: ignore

from decuen.memories._memory import Transition
from decuen.utils.context import get_context


@dataclass
class CriticSettings:
    """Basic common settings for all critics."""

    discount_factor: float


@dataclass
class StateCriticSettings(CriticSettings):
    """Basic common settings for all state critics."""


@dataclass
class ActionCriticSettings(CriticSettings):
    """Basic common settings for all action critics."""


# pylint: disable=too-few-public-methods
class Critic(ABC):
    """Generic abstract critic interface.

    Note that this abstraction is more general that what might normally be viewed as a critic in modern literature: we
    generalize the critic to be able to either generate a state-value or an action-value; as such the interface for
    criticising is delegated to the more specialized interfaces of `StateCritic` and `ActionCritic` and those interfaces
    should be used for subclassing or most other purposes.

    This abstraction provides an interface for one general critic functionality common to both state- and action-value
    based critics: the ability to learn based on past transitions and trajectories to improve critical accuracy.
    """

    state_space: Space
    action_space: Space
    settings: CriticSettings
    _learn_step: int

    def __init__(self, settings: CriticSettings) -> None:
        """Initialize this generic critic interface."""
        self.state_space = get_context().state_space
        self.action_space = get_context().action_space
        self.settings = settings
        self._learn_step = 0

    # TODO: support learning from trajectories
    # XXX: possibly return loss or some other metric?
    @abstractmethod
    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Update internal critic representation based on past transitions."""
        ...


class StateCritic(Critic):
    """Generic abstract state-value critic interface.

    This abstraction slightly specializes the very general `_Critic` interface to output a state-value and as such
    provides an interface for crticising purely based on a state saying how good or bad a given state is to be in.
    """

    settings: StateCriticSettings

    def __init__(self, settings: StateCriticSettings) -> None:
        """Initialize this generic state critic interface."""
        super().__init__(settings)

    @abstractmethod
    def crit(self, state: np.ndarray) -> float:
        """Return a metric of 'goodness' of a state."""
        ...


class ActionCritic(Critic):
    """Generic abstract action-value critic interface.

    This abstraction slightly specializes the very general `_Critic` interface to output an action-value and as such
    provides an interface for criticising based on an action commited in a particular state saying how good or bad
    taking that action in that state is.
    """

    settings: ActionCriticSettings

    def __init__(self, settings: ActionCriticSettings) -> None:
        """Initialize this generic actor critic interface."""
        super().__init__(settings)

    @abstractmethod
    def crit(self, state: np.ndarray, action: np.ndarray) -> float:
        """Return a metric of 'goodness' of taking an action in a state."""
        ...
