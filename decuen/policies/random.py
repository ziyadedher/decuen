"""Implementation of a random policy, a policy that outputs random valid actions."""

import numpy as np  # type: ignore
from gym.spaces.space import Space  # type: ignore

from decuen.policies.policy import Policy


class RandomPolicy(Policy):  # pylint: disable=too-few-public-methods
    """Random policy.

    Samples randomly and independently from the from action space to choose actions.
    """

    action_space: Space

    def __init__(self, action_space: Space) -> None:
        """Initialize a random policy."""
        super().__init__()
        self.action_space = action_space

    def act(self, state: np.ndarray) -> np.ndarray:
        """Choose a random action from the action space."""
        return self.action_space.sample()
