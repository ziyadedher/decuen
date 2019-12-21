"""Implementation of a Boltzmann (softmax) action selection strategy.

[1] http://incompleteideas.net/book/ebook/node17.html
"""

from typing import List, Union, cast

from torch import stack
from torch.distributions import Categorical

from decuen.actors.strats._strategy import Strategy
from decuen.structs import Tensor, tensor


# pylint: disable=too-few-public-methods
class BoltzmannStrategy(Strategy):
    """Boltzmann action selection strategy."""

    def __init__(self, temperature: float = 1) -> None:
        """Initialize a Boltzmann strategy."""
        super().__init__(Categorical)
        self.temperature = temperature

    def params(self, values: Union[List[float], List[List[float]]]) -> Tensor:
        """Generate the parameters for a categorical action distribution based on the action-value logits.

        Computes a softmax over the action-values and returns those as parameters to a categorical distribution.
        """
        if isinstance(values, list) and isinstance(values[0], float):
            values = [cast(List[float], values)]

        values_tensor = tensor(values)
        values_tensor /= self.temperature
        return stack([vals.softmax(0) for vals in values_tensor.unbind()])
# pylint: enable=too-few-public-methods
