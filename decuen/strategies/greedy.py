"""Implementation of a greedy action selection strategy."""

from typing import Collection

import numpy as np  # type: ignore

from decuen.strategies._strategy import Strategy


# pylint: disable=too-few-public-methods
class GreedyStrategy(Strategy):
    """Greedy action selection strategy."""

    def choose(self, action_values: Collection[float]) -> np.ndarray:
        """Chooses the highest-valued action greedily."""
        return np.array(np.argmax(action_values))
