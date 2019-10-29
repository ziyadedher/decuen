"""Implementation of a "void" memory manager, a memory manager that does not actually have any memory."""

from typing import List

from decuen.memories.memory import Memory, Transition


class VoidMemory(Memory[List[Transition]]):
    """Void memory manager, does not store or replay any experiences.

    Can be used for agents that do not require the use of any past memories like most on-policy algorithms.
    """

    def __init__(self) -> None:
        """Initialize a void memory mechanism."""
        super().__init__([])

    def store(self, transition: Transition) -> None:
        """Store a transition in this memory mechanism's buffer with any needed associated information."""

    def replay(self, num: int) -> List[Transition]:
        """Replay experiences from our memory buffer based on some mechanism."""
        return []
