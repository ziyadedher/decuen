"""Implementation of a Boltzmann (softmax) action selection strategy.

[1] http://incompleteideas.net/book/ebook/node17.html
"""

from torch.distributions import Categorical

from decuen.strategies._strategy import Strategy
from decuen.structs import Action, Tensor


# pylint: disable=too-few-public-methods
class BoltzmannStrategy(Strategy):
    """Boltzmann action selection strategy."""

    def __init__(self, temperature: float = 1) -> None:
        """Initialize a Boltzmann strategy."""
        super().__init__()
        self.temperature = temperature

    def choose(self, action_values: Tensor) -> Action:
        """Choose an action to perform based on the Boltzmann distribution with action-value states.

        Essentially computes a softmax over the action-values and stochastically chooses an action by interpreting the
        outputs of the softmax of the probability of sampling each action from the distribution.
        """
        action_values /= self.temperature
        return Categorical(action_values.softmax(0)).sample()
