"""Implementation of a greedy action selection strategy."""

from torch import zeros

from decuen.actors.strats._strategy import Strategy
from decuen.dists import Categorical
from decuen.structs import Tensor


# pylint: disable=too-few-public-methods
class GreedyStrategy(Strategy):
    """Greedy action selection strategy."""

    def __init__(self) -> None:
        """Initialize a greedy strategy."""
        super().__init__(Categorical)

    def act(self, action_values: Tensor) -> Tensor:
        """Generate parameters for a categorical distribution that assigns full probability to one action greedily."""
        probs = zeros(action_values.size())
        probs[action_values.argmax()] = 1
        return probs
