"""Interface for arbitrary memory mechanisms for reinforcement learning agents."""

from abc import ABC, abstractmethod
from typing import Collection, Container, Generic, NamedTuple, TypeVar

import numpy as np  # type: ignore


class Transition(NamedTuple):
    """Simple data structure representing a transition from one state to another with associated information."""

    state: np.ndarray
    action: np.ndarray
    new_state: np.ndarray
    reward: float
    end: bool


BufferType = TypeVar("BufferType", bound=Container[Transition])


class Memory(Generic[BufferType], ABC):
    """Generic abstract memory storage and management unit for an agent.

    This abstraction provides interfaces for the two main functionalities of a memory mechanism:
        1. the ability to store a state transition with associated information in memory, and
        2. the ability to replay experiences based on some internally implemented mechanism.
    """

    _buffer: BufferType

    def __init__(self, buffer: BufferType) -> None:
        """Initialize a generic memory mechanism."""
        self._buffer = buffer

    @abstractmethod
    def store(self, transition: Transition) -> None:
        """Store a transition in this memory mechanism's buffer with any needed associated information."""
        ...

    @abstractmethod
    def replay(self, num: int) -> Collection[Transition]:
        """Replay experiences from our memory buffer based on some mechanism."""
        ...
