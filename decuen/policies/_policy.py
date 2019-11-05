"""Interface for arbitrary policies for reinforcement learning agents."""

from typing import Any, Callable, Generic, Type, TypeVar

import numpy as np  # type: ignore

from decuen.dists._distribution import Distribution

CallableType = TypeVar("CallableType")


class _FunctionProperty(Generic[CallableType]):
    """Wrapper class to provide ability to store functions as class properties."""

    def __get__(self, oself: Any, owner: Any) -> CallableType:
        ...

    def __set__(self, oself: Any, value: CallableType) -> None:
        ...


class Policy:
    """Generic policy usually associated with an agent.

    This abstraction provides an interface for the definition of a policy: returning a distribution over actions given
    a state of the environment. Note that for deterministic policies we still output a distribution over actions but
    hat distribution is spiked at the one specific action we would like to output so that only that action is chosen.
    This simulates the deterministic policy in our stochastic policy setting.
    """

    _parameters_factory: _FunctionProperty[Callable[[np.ndarray], np.ndarray]]
    _distribution_factory: Type[Distribution]

    def __init__(self, parameters_factory: Callable[[np.ndarray], np.ndarray],
                 distribution_factory: Type[Distribution]) -> None:
        """Initialize a generic policy."""
        self._parameters_factory = parameters_factory
        self._distribution_factory = distribution_factory

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Alias calling an instance of this policy as just calling `act`.

        This induces the effect of the policy seeming like a function as it appears in the literature.
        """
        return self.act(state)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate a distribution over action to perform given a state."""
        return self._distribution_factory(self._parameters_factory(state))
