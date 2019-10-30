"""Interface for arbitrary reinforcement learning agents."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np  # type: ignore
from gym.spaces.space import Space  # type: ignore

from decuen.memories.memory import Memory, Transition
from decuen.policies.policy import Policy


class Settings(NamedTuple):
    """Basic common hyperparameter settings for all agents."""

    learning_rate: float
    discount_factor: float

    replay_size: int


class Agent(ABC):
    """Generic abstract reinforcement learning agent interface.

    This abstraction provides interfaces for the two main functionalities associated with an agent in an environment:
        1. the ability to experience a transition possibly learning from it based some internal mechanism, and
        2. the ability to act based on a given state.
    These functionalities are delegated respectively to the memory container and policy associated with this agent and
    as such the main implementation burden falls on specifying the learning mechanism for policy updates, if any.
    """

    state_space: Space
    action_space: Space
    memory: Memory
    policy: Policy
    settings: Settings

    @abstractmethod
    # pylint: disable=too-many-arguments
    def __init__(self, state_space: Space, action_space: Space,
                 memory: Memory, policy: Policy, settings: Settings) -> None:
        """Initialize a generic agent."""
        self.state_space = state_space
        self.action_space = action_space
        self.memory = memory
        self.policy = policy
        self.settings = settings

    def experience(self, transition: Transition) -> None:
        """Experience a transition and potentially learn from it."""
        self._check_transition(transition)
        self.memory.store(transition)
        self.learn(transition)

    @abstractmethod
    def learn(self, transition: Transition) -> None:
        """Learn or improve policy from memory with the most recent transition passed in."""
        ...

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate an action to perform based on a state."""
        self._check_state(state)
        action = self.policy.act(state)
        self._check_action(action)
        return action

    def _check_state(self, state: np.ndarray) -> None:
        """Check that a state is an appropriate input to this agent.

        Raises a `ValueError` if the state is malformed, i.e not part of the state space.
        """
        if state not in self.state_space:
            raise ValueError(f"state `{state}` is not in the agent state space `{self.state_space}`")

    def _check_action(self, action: np.ndarray) -> None:
        """Check that an action is an appropriate output from this agent.

        Raises a `ValueError` if the action is malformed, i.e not part of the action space.
        """
        if action not in self.action_space:
            raise ValueError(f"action `{action}` is not in the agent state space `{self.action_space}`")

    def _check_transition(self, transition: Transition) -> None:
        """Check that a transition is an appropriate experience for this agent.

        Raises a `ValueError` if the transition is not valid, i.e. some state or action in it is malformed.
        """
        self._check_state(transition.state)
        self._check_action(transition.action)
        self._check_state(transition.new_state)
