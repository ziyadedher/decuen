"""Deep Q-network critic based agent."""


from dataclasses import dataclass

import numpy as np  # type: ignore
from gym.spaces import Discrete  # type: ignore
from tensorflow.keras import Sequential  # type: ignore

from decuen.agents._agent import AgentSettings, CriticAgent
from decuen.critics.dqn import DQNCritic, DQNCriticSettings
from decuen.memories._memory import Memory
from decuen.strategies._strategy import Strategy


@dataclass
class DQNAgentSettings(AgentSettings, DQNCriticSettings):
    """Settings for deep Q-network critic based agents."""


class DQNAgent(CriticAgent):
    """Critic agent with only a deep Q-network based critic guiding action selection."""

    critic: DQNCritic

    # pylint: disable=too-many-arguments
    def __init__(self, state_space: Discrete, action_space: Discrete, settings: DQNAgentSettings,
                 memory: Memory, strategy: Strategy, model: Sequential) -> None:
        """Initialize a deep Q-network critic agent."""
        critic = DQNCritic(state_space, action_space, settings, model)
        super().__init__(state_space, action_space, settings, memory, critic, strategy)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate an action to perform based on the Q-values of different actions in a state."""
        return self.strategy.choose(self.critic.values(state))
