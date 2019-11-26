from dataclasses import dataclass

from decuen.actors import Actor
from decuen.agents._agent import Agent, AgentSettings
from decuen.critics import MonteCarloCritic, MonteCarloCriticSettings
from decuen.memories import Memory


@dataclass
class ActorAgentSettings(AgentSettings, MonteCarloCriticSettings):
    """Settings for critic agents."""


class ActorAgent(Agent):
    """High-level reinforcement learning agent based on just an actor and a method for advantage computation."""

    def __init__(self, memory: Memory, actor: Actor, settings: ActorAgentSettings) -> None:
        """Initialize a generic actor agent."""
        super().__init__(memory, actor, MonteCarloCritic(settings), settings)
