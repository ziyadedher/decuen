"""Implementation of a random action selection strategy."""

import random

from decuen._structs import Action, Tensor, tensor
from decuen.strategies._strategy import Strategy


# pylint: disable=too-few-public-methods
class RandomStrategy(Strategy):
    """Random action selection strategy."""

    def choose(self, action_values: Tensor) -> Action:
        """Choose an action to perform uniformly at random."""
        return tensor(random.randint(0, action_values.size()[0] - 1))
