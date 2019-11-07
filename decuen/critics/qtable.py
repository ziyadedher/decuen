"""Q-table action critics.

Implements both Q-learning [1] and Double Q-learning [2] algorithms to train action critics.

[1] Watkins, Christopher (1989), Learning from Delayed Rewards (Ph.D. thesis)
    http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf
[2] van Hasselt, Hado (2011), Double Q-learning
    http://papers.nips.cc/paper/3964-double-q-learning.pdf
[3] Wikipedia, Q-learning
    https://en.wikipedia.org/wiki/Q-learning
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np  # type: ignore
from gym.spaces import Discrete  # type: ignore

from decuen.critics._critic import ActionCritic, ActionCriticSettings
from decuen.memories._memory import Transition
from decuen.utils import checks


@dataclass
class QTableCriticSettings(ActionCriticSettings):
    """Settings for Q-table critics."""


class QTableCritic(ActionCritic):
    """Q-table critic based on [1]."""

    table: np.ndarray

    def __init__(self, state_space: Discrete, action_space: Discrete, settings: QTableCriticSettings) -> None:
        """Initialize this generic actor critic interface."""
        super().__init__(state_space, action_space, settings)

        # TODO: possibly generalize to multi-discrete spaces
        if not isinstance(state_space, Discrete):
            raise TypeError("state space for Q-table critic must be discrete")
        if not isinstance(action_space, Discrete):
            raise TypeError("action space for Q-table critic must be discrete")

        # XXX: possibly experiment with different intiialization techniques
        self.table = np.zeros((state_space.n, action_space.n))

    def learn(self, transitions: Iterable[Transition]) -> None:
        """Update internal critic representation based on past transitions.

        Utilizes a simple value iteration update using a weighted average of the old value and the new information.
        Many of the concepts used here are elaborated upon in [3].
        """
        learn_rate = self.settings.learning_rate
        discount = self.settings.discount_factor

        for transition in transitions:
            checks.check_transition(self.state_space, self.action_space, transition)

            prev_value = self.crit(transition.state, transition.action)
            new_values = self.values(transition.new_state)
            new_value = transition.reward + (0 if transition.terminal else discount * np.max(new_values))

            self.table[transition.state][transition.action] = (1 - learn_rate) * prev_value + learn_rate * new_value

    def crit(self, state: np.ndarray, action: np.ndarray) -> float:
        """Return the Q-value of taking a specific action in a specific state."""
        checks.check_state(self.state_space, state)
        checks.check_action(self.action_space, action)

        return self.table[state][action]

    def values(self, state: np.ndarray) -> np.ndarray:
        """Return an array of Q-values of all actions in a specific state."""
        checks.check_state(self.state_space, state)

        return self.table[state]