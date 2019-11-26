"""Interface for agents based on an actor-critic schema."""

from dataclasses import dataclass

from decuen.actors import Actor
from decuen.agents._agent import Agent, AgentSettings
from decuen.critics import Critic
from decuen.memories import Memory


@dataclass
class ActorCriticAgentSettings(AgentSettings):
    """Settings for actor-critic agents."""


class ActorCriticAgent(Agent):
    """High-level reinforcement learning agent based on an actor-critic formulation."""

    def __init__(self, memory: Memory, actor: Actor, critic: Critic, settings: ActorCriticAgentSettings) -> None:
        """Initialize a generic actor-critic agent."""
        super().__init__(memory, actor, critic, settings)
