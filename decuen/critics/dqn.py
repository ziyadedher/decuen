"""Deep Q-network action critics.

Implements both deep Q-learning [1, 2] and double deep Q-learning [3] algorithms to train action critics.

[1] Mnih, Volodymyr; et al. (2013). Playing Atari with Deep Reinforcement Learning
    https://arxiv.org/pdf/1312.5602.pdf
[2] Mnih, Volodymyr; et al. (2015). Human-level control through deep reinforcement learning
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
[3] van Hasselt, Hado; et al. (2015). Deep Reinforcement Learning with Double Q-learning
    https://arxiv.org/pdf/1509.06461.pdf
"""

from dataclasses import dataclass
from typing import MutableSequence

import numpy as np  # type: ignore
from gym.spaces import Discrete  # type: ignore
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.models import clone_model  # type: ignore

from decuen.critics._q import QCritic, QCriticSettings
from decuen.memories._memory import Transition
from decuen.utils import checks


@dataclass
class DQNCriticSettings(QCriticSettings):
    """Settings for Q-table critics."""


class DQNCritic(QCritic):
    """Deep Q-network critic.

    See [1], [2], [3] for more details.
    """

    action_space: Discrete
    settings: DQNCriticSettings
    network: Sequential
    _target_network: Sequential

    def __init__(self, model: Sequential, settings: DQNCriticSettings) -> None:
        """Initialize this generic actor critic interface."""
        super().__init__(settings)

        # TODO: possibly generalize to multi-discrete spaces
        if not isinstance(self.action_space, Discrete):
            raise TypeError("action space for Q-table critic must be discrete")

        self.network = model
        self._target_network = clone_model(model)
        self._target_network.set_weights(self.network.get_weights())

    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Update internal critic representation based on past transitions."""
        self._learn_step += 1

        if not transitions:
            return
        for transition in transitions:
            checks.check_transition(self.state_space, self.action_space, transition)

        states = np.array([transition.state for transition in transitions])
        new_states = np.array([transition.new_state for transition in transitions])

        values = self.network.predict_on_batch(states).numpy()
        new_values = self.network.predict_on_batch(new_states).numpy()
        target_new_values = self._target_network.predict_on_batch(new_states)

        for i, transition in enumerate(transitions):
            target = transition.reward
            if not transition.terminal:
                if self.settings.double:
                    target += self.settings.discount_factor * new_values[i][np.argmax(target_new_values[i])]
                else:
                    target += self.settings.discount_factor * np.max(target_new_values[i])
            values[i][transition.action] = target

        self.network.train_on_batch(states, values)

        if self._learn_step % self.settings.target_update == 0:
            self._target_network.set_weights(self.network.get_weights())

    def crit(self, state: np.ndarray, action: np.ndarray) -> float:
        """Return the Q-value of taking a specific action in a specific state."""
        checks.check_state(self.state_space, state)
        checks.check_action(self.action_space, action)

        return self.values(state)[action]

    def values(self, state: np.ndarray) -> np.ndarray:
        """Return an array of Q-values of all actions in a specific state."""
        checks.check_state(self.state_space, state)

        return self.network.predict_on_batch(np.array([state]))[0].numpy()
