"""Q-value based agent."""

from dataclasses import dataclass

from decuen._structs import Action, State
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
    def __init__(self, memory: Memory, critic: QCritic, strategy: Strategy, settings: QAgentSettings) -> None:
        """Initialize a Q-value based critic agent."""
        super().__init__(memory, critic, strategy, settings)

    def _act(self, state: State) -> Action:
        return self.strategy.choose(self.critic.values(state))


# Finish PyTorch stuff
# TD learning, implement state and value critic stuff
# Clean up agent representation
# Think about how to pass data from critic to actor
# Maybe make agent generate policy and replace actors with agents
