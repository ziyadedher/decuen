"""Implementation of a "void" memory manager, a memory manager that does not actually have any memory.

Simulates the capacity for on-policy training.
"""

from typing import List

from decuen.memories._memory import Memory
from decuen.structs import Trajectory, Transition


class ShortTermMemory(Memory):
    """Short term memory manager, only stores the most recent events and forgets everything else.

    Can be used for agents that do not require the use of any past memories like most on-policy algorithms.
    """

    def __init__(self) -> None:
        """Initialize a short-term memory mechanism."""
        super().__init__([], [])

    # pylint: disable=useless-super-delegation
    def store_transition(self, transition: Transition) -> None:
        """Store nothing in long-term transition memory."""
        super().store_transition(transition)

    def _replay_transitions(self, num: int) -> List[Transition]:
        return [self.transition] if self.transition else []

    # pylint: disable=useless-super-delegation
    def store_trajectory(self, trajectory: Trajectory) -> None:
        """Store nothing in long-term trajectory memory."""
        super().store_trajectory(trajectory)

    def _replay_trajectories(self, num: int = None) -> List[Trajectory]:
        return [self.trajectory] if self.trajectory else []
