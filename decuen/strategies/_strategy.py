"""Interfaces for arbitrary action selection strategies."""

from abc import ABC, abstractmethod

from decuen.structs import Action, Tensor


# pylint: disable=too-few-public-methods
class Strategy(ABC):
    """Action selection strategy interface.

    Note that this interface currently only support discrete action selection.
    """

    def __init__(self) -> None:
        """Initialize an action selection strategy."""
        ...

    @abstractmethod
    def choose(self, action_values: Tensor) -> Action:
        """Choose an action to perform given the respective action values."""
        ...
