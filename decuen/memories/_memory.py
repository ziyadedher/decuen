"""Interface for arbitrary memory mechanisms for reinforcement learning agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import MutableSequence, Optional, Sequence

import numpy as np  # type: ignore

from decuen.dists._distribution import Distribution


@dataclass
class Transition:
    """Simple data structure representing a transition from one state to another with associated information."""

    state: np.ndarray
    action: np.ndarray
    new_state: np.ndarray
    reward: float
    terminal: bool

    behavior: Optional[Distribution] = None
    state_value: Optional[float] = None
    action_value: Optional[float] = None


Trajectory = Sequence[Transition]


class Memory(ABC):
    """Generic abstract memory storage and management unit for an agent.

    This abstraction provides interfaces for the main functionalities of a memory mechanism:
        1. the ability to store a state transition with associated information in memory,
        2. the ability to replay experiences based on some internally implemented mechanism,
        3. the ability to store a trajectory of transitions, and
        4. the ability to replay a trajectory of experiences based on some internally implemented mechanism.
    """

    transition: Optional[Transition]
    trajectory: Optional[Trajectory]
    transition_replay_num: int
    trajectory_replay_num: int

    _transition_buffer: MutableSequence[Transition]
    _trajectory_buffer: MutableSequence[Trajectory]

    def __init__(self,
                 transition_buffer: MutableSequence[Transition], trajectory_buffer: MutableSequence[Trajectory],
                 transition_replay_num: int = 1, trajectory_replay_num: int = 1) -> None:
        """Initialize a generic memory mechanism."""
        self.transition = None
        self.trajectory = None
        self.transition_replay_num = transition_replay_num
        self.trajectory_replay_num = trajectory_replay_num
        self._transition_buffer = transition_buffer
        self._trajectory_buffer = trajectory_buffer

    @abstractmethod
    def store_transition(self, transition: Transition) -> None:
        """Store a transition in this memory mechanism's buffer with any needed associated information."""
        self.transition = transition

    def replay_transitions(self, num: Optional[int] = None) -> MutableSequence[Transition]:
        """Replay experiences from our memory buffer based on some mechanism."""
        if num is None:
            return self._replay_transitions(self.transition_replay_num)
        return self._replay_transitions(num)

    @abstractmethod
    def _replay_transitions(self, num: int) -> MutableSequence[Transition]:
        ...

    @abstractmethod
    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Store a trajectory in this memory mechanism's buffer consisting of a sequence of transitions."""
        self.trajectory = trajectory

    def replay_trajectories(self, num: Optional[int] = None) -> MutableSequence[Trajectory]:
        """Replay trajectories from our memory buffer based on some mechanism."""
        if num is None:
            return self._replay_trajectories(self.trajectory_replay_num)
        return self._replay_trajectories(num)

    @abstractmethod
    def _replay_trajectories(self, num: int) -> MutableSequence[Trajectory]:
        ...
