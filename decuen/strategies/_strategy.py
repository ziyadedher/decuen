"""Interfaces for arbitrary action selection strategies."""

from abc import ABC, abstractmethod
from typing import Collection

import numpy as np  # type: ignore


# pylint: disable=too-few-public-methods
class Strategy(ABC):
    """Action selection strategy interface.

    Note that this interface currently only support discrete action selection.
    """

    def __init__(self) -> None:
        """Initialize an action selection strategy."""
        ...

    @abstractmethod
    def choose(self, action_values: Collection[float]) -> np.ndarray:
        """Choose an action to perform given the respective action values.

        Assumes that each value is indexed respectively by its action.
        """
        ...
