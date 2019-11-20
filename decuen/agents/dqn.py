"""Deep Q-network critic based agent."""


from dataclasses import dataclass

import numpy as np  # type: ignore

from decuen.agents._agent import AgentSettings, CriticAgent
from decuen.critics.dqn import _DQNCritic
from decuen.memories._memory import Memory
from decuen.strategies._strategy import Strategy


@dataclass
class DQNAgentSettings(AgentSettings):
    """Settings for deep Q-network critic based agents."""


class DQNAgent(CriticAgent):
    """Critic agent with only a deep Q-network based critic guiding action selection."""

    critic: _DQNCritic

    # pylint: disable=too-many-arguments
    def __init__(self, memory: Memory, critic: _DQNCritic, strategy: Strategy,
                 settings: DQNAgentSettings = DQNAgentSettings()) -> None:
        """Initialize a deep Q-network critic agent."""
        super().__init__(memory, critic, strategy, settings)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate an action to perform based on the Q-values of different actions in a state."""
        return self.strategy.choose(self.critic.values(state))
