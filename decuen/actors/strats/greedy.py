"""Implementation of a greedy action selection strategy."""

from typing import List, Union, cast

from torch import arange, zeros_like

from decuen.actors.strats._strategy import Strategy
from decuen.dists import Categorical
from decuen.structs import Tensor, tensor


# pylint: disable=too-few-public-methods
class GreedyStrategy(Strategy):
    """Greedy action selection strategy."""

    def __init__(self) -> None:
        """Initialize a greedy strategy."""
        super().__init__(Categorical)

    def params(self, values: Union[List[float], List[List[float]]]) -> Tensor:
        """Generate parameters for a categorical distribution that assigns full probability to one action greedily."""
        if isinstance(values, list) and isinstance(values[0], float):
            values = [cast(List[float], values)]
        values_tensor = tensor(values)

        probs = zeros_like(values_tensor)
        probs[arange(probs.size()[0]), values_tensor.argmax(dim=1)] = 1
        return probs
# pylint: enable=too-few-public-methods
