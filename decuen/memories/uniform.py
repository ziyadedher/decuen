"""Implementation of a possibly bounded uniform experience replay manager."""

import random
from typing import Iterable, Optional

from decuen.memories._memory import Memory, Trajectory, Transition


class UniformMemory(Memory):
    """Sized uniform memory mechanism, stores memories up to a maximum amount if specified."""

    _transitions_cap: Optional[int]
    _trajectories_cap: Optional[int]

    def __init__(self, transition_replay_num: int = 1, trajectory_replay_num: int = 1,
                 transitions_cap: Optional[int] = None, trajectories_cap: Optional[int] = None) -> None:
        """Initialize a uniform memory mechanism."""
        super().__init__([], [], transition_replay_num, trajectory_replay_num)
        self._transitions_cap = transitions_cap
        self._trajectories_cap = trajectories_cap

    def store_transition(self, transition: Transition) -> None:
        """Store a transition in this memory mechanism's buffer with any needed associated information."""
        self.transition = transition
        if self._transitions_cap is not None and len(self._transition_buffer) == self._transitions_cap:
            self._transition_buffer.pop(0)
        self._transition_buffer.append(transition)

    def _replay_transitions(self, num: int) -> Iterable[Transition]:
        return random.choices(self._transition_buffer, k=num)

    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Store a trajectory in this memory mechanism's buffer consisting of a sequence of transitions."""
        self.trajectory = trajectory
        if self._trajectories_cap is not None and len(self._trajectory_buffer) == self._trajectories_cap:
            self._trajectory_buffer.pop(0)
        self._trajectory_buffer.append(trajectory)

    def _replay_trajectories(self, num: int) -> Iterable[Trajectory]:
        return random.choices(self._trajectory_buffer, k=num)
