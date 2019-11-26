"""Interface for arbitrary memory mechanisms for reinforcement learning agents."""

from abc import ABC, abstractmethod
from typing import MutableSequence, Optional

from decuen.structs import Trajectory, Transition


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

    transitions: MutableSequence[Transition]
    trajectories: MutableSequence[Trajectory]

    def __init__(self,
                 transition_buffer: MutableSequence[Transition], trajectory_buffer: MutableSequence[Trajectory],
                 transition_replay_num: int = 1, trajectory_replay_num: int = 1) -> None:
        """Initialize a generic memory mechanism."""
        self.transition = None
        self.trajectory = None
        self.transition_replay_num = transition_replay_num
        self.trajectory_replay_num = trajectory_replay_num
        self.transitions = transition_buffer
        self.trajectories = trajectory_buffer

    @abstractmethod
    def store_transition(self, transition: Transition) -> None:
        """Store a transition in this memory mechanism's buffer with any needed associated information."""
        self.transition = transition

    def replay_transitions(self, num: Optional[int] = None) -> MutableSequence[Transition]:
        """Replay experiences from our memory buffer based on some mechanism."""
        return self._replay_transitions(min(len(self.transitions), num or self.transition_replay_num))

    @abstractmethod
    def _replay_transitions(self, num: int) -> MutableSequence[Transition]:
        ...

    @abstractmethod
    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Store a trajectory in this memory mechanism's buffer consisting of a sequence of transitions."""
        self.trajectory = trajectory

    def replay_trajectories(self, num: Optional[int] = None) -> MutableSequence[Trajectory]:
        """Replay trajectories from our memory buffer based on some mechanism."""
        return self._replay_trajectories(min(len(self.trajectories), num or self.trajectory_replay_num))

    @abstractmethod
    def _replay_trajectories(self, num: int) -> MutableSequence[Trajectory]:
        ...

    def clear(self) -> None:
        """Clear this memory and reset it to its state at initialization.

        Stored transitions and trajectories are forgotten and memory is reinitialized.
        """
        self.transition = None
        self.trajectory = None
        self.transitions.clear()
        self.trajectories.clear()
