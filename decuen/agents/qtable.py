"""Q-table-critic based agent."""

from dataclasses import dataclass

import numpy as np  # type: ignore
from gym.spaces import Discrete  # type: ignore

from decuen.agents._agent import AgentSettings, CriticAgent
from decuen.critics.qtable import QTableCritic, QTableCriticSettings
from decuen.memories._memory import Memory
from decuen.strategies._strategy import Strategy


@dataclass
class QTableAgentSettings(AgentSettings, QTableCriticSettings):
    """Settings for Q-table-critic based agents."""


class QTableAgent(CriticAgent):
    """Critic agent with only a Q-table based critic guiding action selection."""

    critic: QTableCritic

    # pylint: disable=too-many-arguments
    def __init__(self, state_space: Discrete, action_space: Discrete, settings: QTableAgentSettings,
                 memory: Memory, strategy: Strategy) -> None:
        """Initialize a Q-table critic agent."""
        critic = QTableCritic(state_space, action_space, settings)
        super().__init__(state_space, action_space, settings, memory, critic, strategy)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate an action to perform based on the Q-values of different actions in a state."""
        return self.strategy.choose(self.critic.values(state))
