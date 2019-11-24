"""Interfaces for arbitrary reinforcement learning agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np  # type: ignore
from gym.spaces import Discrete, Space  # type: ignore

from decuen.actors import Actor
from decuen.critics import ActionCritic
from decuen.memories import Memory
from decuen.strategies import RandomStrategy, Strategy
from decuen.structs import Action, State, Transition, tensor
from decuen.utils.context import get_context


@dataclass
class AgentSettings:
    """Basic common hyperparameter settings for all agents."""


class Agent(ABC):
    """High-level reinforcement learning agent abstraction.

    This abstraction implements essentially no logic instead delegating all processing to the respective subclasses
    and acts only as a wrapper and liason to the internal structures. There is some high-level functionality provided
    by this interface in order to streamline the use of agents:
        1. the ability to step based on an environment and update internal states,
        2. the ability to act based on a state, and
        3. the ability to learn from memory.
    """

    state_space: Space
    action_space: Space
    memory: Memory
    settings: AgentSettings

    # Current state of the agent
    _state: Optional[State]
    # Action taken at that state
    _action: Optional[State]
    # Current agent trajectory
    _trajectory: List[Transition]

    def __init__(self, memory: Memory, settings: AgentSettings) -> None:
        """Initialize a generic agent."""
        self.state_space = get_context().state_space
        self.action_space = get_context().action_space
        self.memory = memory
        self.settings = settings

        self._state = None
        self._action = None
        self._trajectory = []

    def init(self, state: np.ndarray) -> np.ndarray:
        """Initialize an agent at the start of a new episode."""
        return self._step(tensor(state.astype(np.float32)), None, None).numpy()

    def step(self, state: np.ndarray, reward: float, terminal: bool) -> np.ndarray:
        """Step based on a new state, a terminal state signal, and a reward signal.

        Moves this agent forward in time and calculates transitions based on the state of the agent the last time step
        was called. The reward signal corresponds to the transition that caused the migration to this state and the
        terminal signal corresponds to the currently inputted state.
        """
        return self._step(tensor(state.astype(np.float32)), reward, terminal).numpy()

    def _step(self, state: State, reward: Optional[float], terminal: Optional[bool]) -> Action:
        action = self._act(state)

        # If we have no history in this episode, we still don't have anything to store
        if self._state is None or self._action is None or reward is None or terminal is None:
            self._state = state
            self._action = action
            return action

        # Generate the transition and append it to the trajectory
        transition = Transition(state=self._state, action=self._action,
                                new_state=state, reward=reward, terminal=terminal)
        self._trajectory.append(transition)

        # Store the transition in memory and either reset the state of this agent while storing the trajectory
        # if we reached the end of the episode or just continue normally otherwise
        self.memory.store_transition(transition)
        if terminal:
            self.memory.store_trajectory(self._trajectory)
            self._state = None
            self._action = None
            self._trajectory = []
        else:
            self._state = state
            self._action = action

        return action

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate an action to perform based on a state."""
        return self._act(tensor(state.astype(np.float32))).numpy()

    @abstractmethod
    def _act(self, state: State) -> Action:
        """Act internally based on a state.

        Override this instead of `act` in order to preserve compatibility layers.
        """
        ...

    @abstractmethod
    def learn(self) -> None:
        """Learn or improve this agent from memory."""
        ...


class ActorAgent(Agent):
    """Generic actor agent interface.

    Provides functionality to use different modular actors in an agent instead of coding basics of the actor framework
    from scratch as would be the case using a regular agent.
    """

    actor: Actor

    def __init__(self, memory: Memory, actor: Actor, settings: AgentSettings) -> None:
        """Initialize a generic actor agent."""
        super().__init__(memory, settings)
        self.actor = actor

    def _act(self, state: State) -> Action:
        return self.actor.act(state)

    def learn(self) -> None:
        """Learn or improve this agent from memory.

        Delegates learning to the actor which learns to generate better actions.
        """
        self.actor.learn(self.memory.replay_trajectories())


class CriticAgent(Agent):
    """Generic critic agent interface.

    Provides functionality to use different modular critics in an agent instead of coding basics of the critic framework
    from scratch as would be the case using a regular agent.
    """

    critic: ActionCritic
    strategy: Strategy

    # TODO: support state critic and action critic
    def __init__(self, memory: Memory, critic: ActionCritic, strategy: Strategy, settings: AgentSettings) -> None:
        """Initialize a generic critic agent."""
        super().__init__(memory, settings)
        self.critic = critic
        self.strategy = strategy

    def _act(self, state: State) -> Action:
        # Queries the critic to figure out which action would be the best to perform given the state.
        # This method should usually be overriden because the implementation runs through all possible actions and
        # produces the one with the highest critic score; however, with more information about the interal
        # representation of the critic, very large optimizations could be made.

        # TODO: support sampling mechanism to get around needing the action space to be discrete
        if not isinstance(self.action_space, Discrete):
            raise TypeError("critic agent acting is only supported for discrete action spaces")
        return self.strategy.choose(tensor([self.critic.crit(state, action) for action in self.action_space.n]))

    def learn(self) -> None:
        """Learn or improve this agent from memory.

        Delegates learning to the critic which learns more accurate state or action values.
        """
        self.critic.learn(self.memory.replay_transitions())


class ActorCriticAgent(ActorAgent, CriticAgent):
    """Generic actor-critic agent interface.

    Provides functionality to use different modular actors and critic in an agent instead of coding basics of the
    actor-critic framework from scratch as would be the case using a regular agent.
    """

    def __init__(self, memory: Memory, actor: Actor, critic: ActionCritic, settings: AgentSettings) -> None:
        """Initialize a generic actor-critic agent."""
        ActorAgent.__init__(self, memory, actor, settings)
        # Note that strategy does nothing in this case since we are never calling the `act` of the `CriticAgent`
        CriticAgent.__init__(self, memory, critic, RandomStrategy(), settings)
        self.actor = actor
        self.critic = critic

    def _act(self, state: State) -> Action:
        return ActorAgent.act(self, state)

    def learn(self) -> None:
        """Learn or improve this agent from memory.

        Calls learning procedures on both the actor and the critic.
        """
        ActorAgent.learn(self)
        CriticAgent.learn(self)
