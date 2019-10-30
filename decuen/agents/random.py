"""Implementation of a random agent with nothing special."""

from gym.spaces.space import Space  # type: ignore

from decuen.agents.agent import Agent, Settings
from decuen.memories.memory import Transition
from decuen.memories.void import VoidMemory
from decuen.policies.random import RandomPolicy


class RandomAgent(Agent):
    """Random reinforcement learning agent.

    Uses a `RandomPolicy` to generate its actions and does not learn or have memory.
    """

    def __init__(self, state_space: Space, action_space: Space) -> None:
        """Initialize a random agent."""
        memory = VoidMemory()
        policy = RandomPolicy(action_space)
        settings = Settings(0, 0, 0)
        super().__init__(state_space, action_space, memory, policy, settings)

    def learn(self, transition: Transition) -> None:
        """Learn nothing as the random agent will remain random."""
