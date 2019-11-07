"""Implementation of an epsilon-greedy action selection strategy."""

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Collection, Optional

import numpy as np  # type: ignore

from decuen.strategies._strategy import Strategy
from decuen.strategies.greedy import GreedyStrategy
from decuen.strategies.rand import RandomStrategy
from decuen.utils.function_property import _FunctionProperty


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

    func: _FunctionProperty[Callable[[float], float]]

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
    random: ClassVar[RandomStrategy] = RandomStrategy()
    epsilon: float
    min_epsilon: float
    max_epsilon: float
    _decay: EpsilonDecay

    def __init__(self, epsilon: float, max_epsilon: float = 1, min_epsilon: float = 0,
                 decay: Optional[EpsilonDecay] = None) -> None:
        """Initialize an epsilon greedy strategy."""
        super().__init__()
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self._decay = decay if decay else NoEpsilonDecay()

    def choose(self, action_values: Collection[float]) -> np.ndarray:
        """Choose an action to perform greedily with probability epsilon, otherwise randomly.

        Decays epsilon according to the decay mechanism after choosing an action.
        """
        action = (self.greedy.choose(action_values) if np.random.rand() > self.epsilon
                  else self.random.choose(action_values))
        self.decay()
        return action

    def decay(self) -> None:
        """Decay the epsilon according to the decaying technique."""
        self.epsilon = max(self._decay.decay(self.epsilon), self.min_epsilon)

    def reset(self) -> None:
        """Reset the epsilon to be the maximum epsilon."""
        self.epsilon = self.max_epsilon
