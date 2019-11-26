"""Implementation of a random action selection strategy."""

from torch import ones

from decuen.actors.strats._strategy import Strategy
from decuen.dists import Categorical
from decuen.structs import Tensor


# pylint: disable=too-few-public-methods
class UniformStrategy(Strategy):
    """Uniform action selection strategy."""

    def __init__(self) -> None:
        """Initialize a uniformly random strategy."""
        super().__init__(Categorical)

    def act(self, action_values: Tensor) -> Tensor:
        """Generate parameters for a uniform categorical distribution."""
        return ones(action_values.size()) / action_values.numel()
