"""Interfaces for arbitrary reinforcement learning agents."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np  # type: ignore

from decuen.actors import Actor
from decuen.critics import Critic
from decuen.memories import Memory
from decuen.structs import Action, State, Trajectory, Transition, tensor
from decuen.utils.context import Contextful


@dataclass
class AgentSettings:
    """Basic common hyperparameter settings for all agents."""


class Agent(Contextful):
    """High-level reinforcement learning agent abstraction.

    This abstraction implements essentially no logic instead delegating all processing to the respective subclasses
    and acts only as a wrapper and liason to the internal structures. There is some high-level functionality provided
    by this interface in order to streamline the use of agents:
        1. the ability to step based on an environment and update internal states,
        2. the ability to act based on a state, and
        3. the ability to learn from memory.
    """

    memory: Memory
    settings: AgentSettings
    actor: Actor
    critic: Critic

    # Current state of the agent
    _state: Optional[State]
    # Action taken at that state
    _action: Optional[State]
    # Current agent trajectory
    _trajectory: List[Transition]

    def __init__(self, memory: Memory, actor: Actor, critic: Critic, settings: AgentSettings) -> None:
        """Initialize a generic agent."""
        super().__init__()
        self.memory = memory
        self.settings = settings
        self.actor = actor
        self.critic = critic

        self._state = None
        self._action = None
        self._trajectory = []

        self.actor.critic = critic

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
        action = self._act(state).squeeze(0)

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
            self.memory.store_trajectory(Trajectory(self._trajectory))
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

    def _act(self, state: State) -> Action:
        """Act internally based on a state.

        Override this instead of `act` in order to preserve compatibility layers.
        """
        return self.actor.act(state).sample()

    def learn(self) -> None:
        """Learn or improve this agent from memory."""
        for trajectory in self.memory.replay_trajectories():
            self.actor.learn(trajectory)
        self.critic.learn(self.memory.replay_transitions())
