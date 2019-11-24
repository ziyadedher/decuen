"""Implementation of a Boltzmann (softmax) action selection strategy.

[1] http://incompleteideas.net/book/ebook/node17.html
"""

from torch.distributions import Categorical

from decuen.actors.strats._strategy import Strategy
from decuen.structs import Tensor


# pylint: disable=too-few-public-methods
class BoltzmannStrategy(Strategy):
    """Boltzmann action selection strategy."""

    def __init__(self, temperature: float = 1) -> None:
        """Initialize a Boltzmann strategy."""
        super().__init__(Categorical)
        self.temperature = temperature

    def act(self, action_values: Tensor) -> Tensor:
        """Generate the parameters for a categorical action distribution based on the action-value logits.

        Computes a softmax over the action-values and returns those as parameters to a categorical distribution.
        """
        action_values /= self.temperature
        return action_values.softmax(0)
