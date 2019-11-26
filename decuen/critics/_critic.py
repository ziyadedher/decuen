"""Interfaces for arbitrary critics and respective settings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import MutableSequence, Union

from decuen.structs import Action, State, Tensor, Trajectory, Transition
from decuen.utils.context import Contextful


@dataclass
class CriticSettings:
    """Basic common settings for all critics."""

    discount_factor: float


# pylint: disable=too-few-public-methods
class _Critic(ABC, Contextful):
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
    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Update internal critic representation based on past transitions."""
        ...


class AdvantageCritic(_Critic):
    """Generic abstract advantage critic interface.

    This abstraction slightly specializes the very general `_Critic` interface to output an advantage and as such
    provides an interface for generating the advantage of transitions.
    """

    @abstractmethod
    def advantage(self, trajectory: Union[Transition, Trajectory]) -> Tensor:
        """Estimate the advantage of every transition in a trajectory."""
        ...


class StateValueCritic(_Critic):
    """Generic abstract state-value critic interface.

    This abstraction slightly specializes the very general `_Critic` interface to output a state-value and as such
    provides an interface for crticising purely based on a state saying how good or bad a given state is to be in.
    """

    @abstractmethod
    def crit(self, state: State) -> Tensor:
        """Return a metric of 'goodness' of a state or tensor of states."""
        ...


class ActionValueCritic(_Critic):
    """Generic abstract action-value critic interface.

    This abstraction slightly specializes the very general `_Critic` interface to output an action-value and as such
    provides an interface for criticising based on an action commited in a particular state saying how good or bad
    taking that action in that state is.
    """

    @abstractmethod
    def crit(self, state: State, action: Action) -> Tensor:
        """Return a metric of 'goodness' of taking an action or tensor of actions in a state."""
        ...
