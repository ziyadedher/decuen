"""Collection of simple checks and exceptions for use around the library."""

import numpy as np  # type: ignore
from gym.spaces import Discrete, Space  # type: ignore

from decuen.memories._memory import Transition


class DecuenError(Exception):
    """General error class used to encapsulate all errors produced by the library."""


class MalformedStateError(DecuenError):
    """Error raised when a state is found to not belong to an appropriate state space."""


class MalformedActionError(DecuenError):
    """Error raised when an action is found to not belong to an appropriate action space."""


def check_state(state_space: Space, state: np.ndarray) -> None:
    """Check that a state is an appropriately part of a state space.

    Raises a `MalformedStateError` if the state is malformed, i.e not part of the state space.
    """
    # TODO: possibly support mutli-discrete state spaces
    _state = state
    if isinstance(state_space, Discrete):
        check_discrete_state(state)
        _state = _state[0]

    if _state not in state_space:
        raise MalformedStateError(f"state `{state}` is not in the agent state space `{state_space}`")


def check_action(action_space: Space, action: np.ndarray) -> None:
    """Check that an action is an appropriate part of an action space.

    Raises a `MalformedActionError` if the action is malformed, i.e not part of the action space.
    """
    # TODO: possibly support multi-discrete action spaces
    _action = action
    if isinstance(action_space, Discrete):
        check_discrete_action(action)
        _action = _action[0]

    if _action not in action_space:
        raise MalformedActionError(f"action `{action}` is not in the agent state space `{action_space}`")


def check_discrete_state(state: np.ndarray) -> None:
    """Check that an state is possibly part of some discrete state space.

    Raise a `MalformedStateError` if the state cannot be part of a discrete state space.
    """
    if state.shape != (1,):
        raise MalformedStateError(f"discrete state space element must be a (1,)-shaped array")


def check_discrete_action(action: np.ndarray) -> None:
    """Check that an action is possibly part of some discrete action space.

    Raise a `MalformedActionError` if the action cannot be part of a discrete action space.
    """
    if action.shape != (1,):
        raise MalformedStateError(f"discrete action space element must be a (1,)-shaped array")


def check_transition(state_space: Space, action_space: Space, transition: Transition) -> None:
    """Check that a transition is appropriately formed according to given state and action spaces.

    Raises a `MalformedStateError` or `MalformedActionError` if any state or action in the transition is malformed.
    """
    check_state(state_space, transition.state)
    check_action(action_space, transition.action)
    check_state(state_space, transition.new_state)
