"""Interface for arbitrary reinforcement learning agents."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np  # type: ignore
from gym import Space  # type: ignore

from decuen.actors._actor import Actor
from decuen.critics._critic import ActionCritic
from decuen.memories._memory import Memory, Transition
from decuen.utils import checks


@dataclass
class AgentSettings:
    """Basic common hyperparameter settings for all agents."""

    num_replay_transitions: int
    num_replay_trajectories: int


# pylint: disable=too-many-instance-attributes
class Agent:
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
    actor: Actor
    critic: ActionCritic  # TODO: support state critic and action critic
    settings: AgentSettings

    # Current state of the agent
    _state: Optional[np.ndarray]
    # Action taken at that state
    _action: Optional[np.ndarray]
    # Current agent trajectory
    _trajectory: List[Transition]

    # TODO: support state critic and action critic
    # pylint: disable=too-many-arguments
    def __init__(self, state_space: Space, action_space: Space,
                 memory: Memory, actor: Actor, critic: ActionCritic,
                 settings: AgentSettings) -> None:
        """Initialize a generic agent."""
        self.state_space = state_space
        self.action_space = action_space
        self.memory = memory
        self.actor = actor
        self.critic = critic
        self.settings = settings
        self._trajectory = []

    def step(self, state: np.ndarray, reward: Optional[float] = None, terminal: Optional[bool] = None, *,
             learn: bool = True) -> np.ndarray:
        """Step based on a new state and a previous reward and end if they exist."""
        checks.check_state(self.state_space, state)

        action = self.act(state)

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

        if learn:
            self.learn()

        return action

    def act(self, state: np.ndarray) -> np.ndarray:
        """Generate an action to perform based on a state."""
        checks.check_state(self.state_space, state)

        action = self.actor.act(state)

        checks.check_action(self.action_space, action)
        return action

    def learn(self) -> None:
        """Learn or improve this agent from memory."""
        transitions = self.memory.replay_transitions(self.settings.num_replay_transitions)
        trajectories = self.memory.replay_trajectories(self.settings.num_replay_trajectories)

        for transition in transitions:
            self._populate_values(transition)
            checks.check_transition(self.state_space, self.action_space, transition)
        for trajectory in trajectories:
            for transition in trajectory:
                self._populate_values(transition)
                checks.check_transition(self.state_space, self.action_space, transition)

        if transitions:
            self.critic.learn(transitions)
        if trajectories:
            self.actor.learn(trajectories)

    def _populate_values(self, transition: Transition) -> None:
        """Populate the critic fields in a transition based on the current critic."""
        # TODO: support state critic and action critic
        transition.state_value = self.critic.crit(transition.state, transition.action)
