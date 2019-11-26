from dataclasses import dataclass

from decuen.actors import StrategyActor, StrategyActorSettings
from decuen.actors.strats import Strategy
from decuen.agents._agent import Agent, AgentSettings
from decuen.critics import ActionValueCritic
from decuen.memories import Memory


@dataclass
class CriticAgentSettings(AgentSettings, StrategyActorSettings):
    """Settings for critic agents."""


class CriticAgent(Agent):
    """High-level reinforcement learning agent based on just a critic and strategy for action selection."""

    def __init__(self, memory: Memory, strategy: Strategy, critic: ActionValueCritic,
                 settings: CriticAgentSettings) -> None:
        """Initialize a generic critic agent."""
        super().__init__(memory, StrategyActor(strategy, critic, settings), critic, settings)
