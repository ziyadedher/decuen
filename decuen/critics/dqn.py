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
from gym.spaces import Discrete, Space  # type: ignore
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.activations import linear  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.losses import mean_squared_error  # type: ignore
from tensorflow.keras.optimizers import Optimizer  # type: ignore

from decuen.critics._critic import ActionCritic, ActionCriticSettings
from decuen.memories._memory import Transition
from decuen.utils import checks


@dataclass
class DQNCriticSettings(ActionCriticSettings):
    """Settings for Q-table critics."""

    optimizer: Optimizer


class DQNCritic(ActionCritic):
    """Q-table critic based on [1, 2].

    Implements Q-learning with or without target networks.
    """

    settings: DQNCriticSettings
    network: Sequential

    def __init__(self, state_space: Space, action_space: Discrete, settings: DQNCriticSettings,
                 model: Sequential) -> None:
        """Initialize this generic actor critic interface."""
        super().__init__(state_space, action_space, settings)

        # TODO: possibly generalize to multi-discrete spaces
        if not isinstance(action_space, Discrete):
            raise TypeError("action space for Q-table critic must be discrete")

        self._finalize_model(model)
        self.network = model

    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Update internal critic representation based on past transitions."""
        if not transitions:
            return
        for transition in transitions:
            checks.check_transition(self.state_space, self.action_space, transition)

        states = np.array([transition.state for transition in transitions])
        new_states = np.array([transition.new_state for transition in transitions])

        values = self.network.predict_on_batch(states).numpy()
        new_values = self.network.predict_on_batch(new_states).numpy()

        for i, transition in enumerate(transitions):
            target = transition.reward
            if not transition.terminal:
                target += self.settings.discount_factor * np.max(new_values[i])
            values[i][transition.action] = target

        self.network.train_on_batch(states, values)

    def crit(self, state: np.ndarray, action: np.ndarray) -> float:
        """Return the Q-value of taking a specific action in a specific state."""
        checks.check_state(self.state_space, state)
        checks.check_action(self.action_space, action)

        return self.values(state)[action]

    def values(self, state: np.ndarray) -> np.ndarray:
        """Return an array of Q-values of all actions in a specific state."""
        checks.check_state(self.state_space, state)

        return self.network.predict_on_batch(np.array([state]))[0].numpy()

    def _finalize_model(self, model: Sequential) -> Sequential:
        """Finalize a model for use as a Q-network and compile it.

        Creates a new layer to serve as the final output Q-values layer and then compiles the entire keras model.
        """
        model.add(Dense(self.action_space.n, activation=linear))
        # TODO: do something cooler with compile
        model.compile(self.settings.optimizer, loss=mean_squared_error)
