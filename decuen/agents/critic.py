"""Interface for agents based soley on critics for guided action selection."""

from dataclasses import dataclass

from decuen.actors import StrategyActor, StrategyActorSettings
from decuen.actors.strats import Strategy
from decuen.agents._agent import Agent, AgentSettings
from decuen.critics import QValueCritic
from decuen.memories import Memory


@dataclass(frozen=True)
class CriticAgentSettings(AgentSettings, StrategyActorSettings):
    """Settings for critic agents."""


class CriticAgent(Agent):
    """High-level reinforcement learning agent based on just a critic and strategy for action selection."""

    def __init__(self, memory: Memory, strategy: Strategy, critic: QValueCritic,
                 settings: CriticAgentSettings) -> None:
        """Initialize a generic critic agent."""
        super().__init__(memory, StrategyActor(strategy, settings), critic, settings)
