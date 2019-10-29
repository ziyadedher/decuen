"""Interface for arbitrary policies for reinforcement learning agents."""

from abc import ABC, abstractmethod

import numpy as np  # type: ignore


class Policy(ABC):  # pylint: disable=too-few-public-methods
    """Generic abstract policy usually associated with an agent.

    These policies are usually dynamic and heavily mutable to improve efficiency and ease-of-use; agents tend to
    iteratively update these policies in-place and as such care must be taken when attempting to freeze a policy for
    any reason.

    This abstraction provides an interface for the definition of a policy: choosing an action given some state. Notice
    that this choice is possibly stochastic but can also be deterministic.
    """

    def __init__(self) -> None:
        """Initialize a generic policy."""

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """Choose an action to perform given a state."""
        ...
