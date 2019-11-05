"""Interface for arbitrary memory mechanisms for reinforcement learning agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Generic, Iterable, MutableSequence, Optional, Sequence,
                    TypeVar)

import numpy as np  # type: ignore
from scipy.stats._distn_infrastructure import rv_frozen  # type: ignore


@dataclass
class Transition:
    """Simple data structure representing a transition from one state to another with associated information."""

    state: np.ndarray
    action: np.ndarray
    new_state: np.ndarray
    reward: float
    terminal: bool

    behavior: Optional[rv_frozen] = None
    state_value: Optional[float] = None
    action_value: Optional[float] = None


Trajectory = Sequence[Transition]

TransitionBufferType = TypeVar("TransitionBufferType", bound=MutableSequence[Transition])
TrajectoryBufferType = TypeVar("TrajectoryBufferType", bound=MutableSequence[Trajectory])


class Memory(Generic[TransitionBufferType, TrajectoryBufferType], ABC):
    """Generic abstract memory storage and management unit for an agent.

    This abstraction provides interfaces for the main functionalities of a memory mechanism:
        1. the ability to store a state transition with associated information in memory,
        2. the ability to replay experiences based on some internally implemented mechanism,
        3. the ability to store a trajectory of transitions, and
        4. the ability to replay a trajectory of experiences based on some internally implemented mechanism.
    """

    transition: Optional[Transition]
    trajectory: Optional[Trajectory]

    _transition_buffer: TransitionBufferType
    _trajectory_buffer: TrajectoryBufferType

    def __init__(self, transition_buffer: TransitionBufferType, trajectory_buffer: TrajectoryBufferType) -> None:
        """Initialize a generic memory mechanism."""
        self.transition = None
        self.trajectory = None
        self._transition_buffer = transition_buffer
        self._trajectory_buffer = trajectory_buffer

    @abstractmethod
    def store_transition(self, transition: Transition) -> None:
        """Store a transition in this memory mechanism's buffer with any needed associated information."""
        self.transition = transition

    @abstractmethod
    def replay_transitions(self, num: int) -> Iterable[Transition]:
        """Replay experiences from our memory buffer based on some mechanism."""
        ...

    @abstractmethod
    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Store a trajectory in this memory mechanism's buffer consisting of a sequence of transitions."""
        self.trajectory = trajectory

    @abstractmethod
    def replay_trajectories(self, num: int) -> Iterable[Trajectory]:
        """Replay trajectories from our memory buffer based on some mechanism."""
        ...
