"""Q-value based agent."""

from dataclasses import dataclass

import numpy as np  # type: ignore

from decuen.agents._agent import AgentSettings, CriticAgent
from decuen.critics._q import QCritic
from decuen.memories._memory import Memory
from decuen.strategies._strategy import Strategy


@dataclass
class QAgentSettings(AgentSettings):
    """Settings for Q-value critic based agents."""


class QAgent(CriticAgent):
    """Critic agent with only a Q-value based critic guiding action selection."""

    critic: QCritic

    # pylint: disable=too-many-arguments
    def __init__(self, memory: Memory, critic: QCritic, strategy: Strategy,
                 settings: QAgentSettings = QAgentSettings()) -> None:
        """Initialize a Q-value based critic agent."""
        super().__init__(memory, critic, strategy, settings)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate an action to perform based on the Q-values of different actions in a state."""
        return self.strategy.choose(self.critic.values(state))
