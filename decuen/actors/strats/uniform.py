"""Implementation of a random action selection strategy."""

from typing import List, Union

from torch import ones_like

from decuen.actors.strats._strategy import Strategy
from decuen.dists import Categorical
from decuen.structs import Tensor, tensor


# pylint: disable=too-few-public-methods
class UniformStrategy(Strategy):
    """Uniform action selection strategy."""

    def __init__(self) -> None:
        """Initialize a uniformly random strategy."""
        super().__init__(Categorical)

    def params(self, values: Union[List[float], List[List[float]]]) -> Tensor:
        """Generate parameters for a uniform categorical distribution."""
        values_tensor = tensor(values)
        return ones_like(values_tensor) / values_tensor.numel()
# pylint: enable=too-few-public-methods
