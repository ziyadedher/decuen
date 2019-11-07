"""Implementation of a random action selection strategy."""

from typing import Collection

import numpy as np  # type: ignore

from decuen.strategies._strategy import Strategy


# pylint: disable=too-few-public-methods
class RandomStrategy(Strategy):
    """Random action selection strategy."""

    def choose(self, action_values: Collection[float]) -> np.ndarray:
        """Choose an action to perform uniformly at random."""
        return np.array(np.random.randint(0, len(action_values)))
