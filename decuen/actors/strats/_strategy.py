"""Interfaces for arbitrary action selection strategies."""

from abc import ABC, abstractmethod
from typing import List, Type, Union

from decuen.dists import Distribution
from decuen.structs import Tensor
from decuen.utils.context import Contextful


# pylint: disable=too-few-public-methods
class Strategy(ABC, Contextful):
    """Action selection strategy interface.

    Note that this interface currently only support discrete action selection.
    """

    distribution_type: Type[Distribution]

    def __init__(self, distribution_type: Type[Distribution]) -> None:
        """Initialize an action selection strategy."""
        super().__init__()
        self.distribution_type = distribution_type

    @abstractmethod
    def params(self, values: Union[List[float], List[List[float]]]) -> Tensor:
        """Generate the parameters for the strategy action distribution based on the values of actions."""
        ...
