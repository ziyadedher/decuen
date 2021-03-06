"""Implementation of an epsilon-greedy action selection strategy."""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional

from decuen.actors.strats._strategy import Strategy
from decuen.actors.strats.greedy import GreedyStrategy
from decuen.actors.strats.uniform import UniformStrategy
from decuen.dists import Categorical
from decuen.structs import Action, Tensor
from decuen.utils.function_property import FunctionProperty


# pylint: disable=too-few-public-methods
class EpsilonDecay(ABC):
    """Technique used to decay epsilon in `EpsilonGreedyStrategy`."""

    @abstractmethod
    def decay(self, value: float) -> float:
        """Decay an epsilon value and return the decayed value."""
        ...


class NoEpsilonDecay(EpsilonDecay):
    """No-op epsilon "decay"."""

    def decay(self, value: float) -> float:
        """Return the inputted value directly with no decay."""
        return value


class FunctionEpsilonDecay(EpsilonDecay):
    """Functional epsilon decay.

    Decays based on a custom decay function.
    """

    func: FunctionProperty[Callable[[float], float]]

    def __init__(self, func: Callable[[float], float]) -> None:
        """Initialize a functional epsilon decay technique."""
        self.func = func

    def decay(self, value: float) -> float:
        """Return the decayed value according to the decaying function."""
        return self.func(value)


class LinearEpsilonDecay(EpsilonDecay):
    """Linear epsilon decay.

    Decays linearly based on a linear decay factor.
    """

    factor: float

    def __init__(self, factor: float) -> None:
        """Initialize a linear epsilon decay technique."""
        self.factor = factor

    def decay(self, value: float) -> float:
        """Return the value reduced by the linear decay factor."""
        return value - self.factor


class ExponentialEpsilonDecay(EpsilonDecay):
    """Exponential epsilon decay.

    Decays geometrically based on a exponential decay factor.
    """

    factor: float

    def __init__(self, factor: float) -> None:
        """Initialize an exponential epsilon decay technique."""
        # TODO: warn against weird factors (i.e. <= 0 or >= 1)
        self.factor = factor

    def decay(self, value: float) -> float:
        """Return the value multiplied by the exponential decay factor."""
        return value * self.factor


# pylint: disable=too-few-public-methods
class EpsilonGreedyStrategy(Strategy):
    """Epsilon-greedy action selection strategy."""

    greedy: ClassVar[GreedyStrategy] = GreedyStrategy()
    random: ClassVar[UniformStrategy] = UniformStrategy()
    epsilon: float
    min_epsilon: float
    max_epsilon: float
    _decay: EpsilonDecay

    def __init__(self, epsilon: float, max_epsilon: float = 1, min_epsilon: float = 0,
                 decay: Optional[EpsilonDecay] = None) -> None:
        """Initialize an epsilon-greedy strategy."""
        super().__init__(Categorical)
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self._decay = decay if decay else NoEpsilonDecay()

    def act(self, action_values: Tensor) -> Action:
        """Generate parameters for a categorical action distribution based on a epsilon-greedy strategy.

        Decays epsilon according to the decay mechanism after choosing an action.
        """
        probs = (1 - self.epsilon) * self.greedy.act(action_values) + self.epsilon * self.random.act(action_values)
        self.decay()
        return probs

    def decay(self) -> None:
        """Decay the epsilon according to the decaying technique."""
        self.epsilon = max(self._decay.decay(self.epsilon), self.min_epsilon)

    def reset(self) -> None:
        """Reset the epsilon to be the maximum epsilon."""
        self.epsilon = self.max_epsilon
