from dataclasses import dataclass

from decuen.agents._agent import Agent, AgentSettings


@dataclass
class ActorCriticAgentSettings(AgentSettings):
    """Settings for actor-critic agents."""


class ActorCriticAgent(Agent):
    """High-level reinforcement learning agent based on an actor-critic formulation."""
