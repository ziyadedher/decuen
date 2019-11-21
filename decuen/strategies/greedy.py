"""Implementation of a greedy action selection strategy."""

from decuen._structs import Action, Tensor
from decuen.strategies._strategy import Strategy


# pylint: disable=too-few-public-methods
class GreedyStrategy(Strategy):
    """Greedy action selection strategy."""

    def choose(self, action_values: Tensor) -> Action:
        """Chooses the highest-valued action greedily."""
        return action_values.argmax()
