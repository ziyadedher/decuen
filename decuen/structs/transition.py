"""Constructs representing and helpers for transitions in the framework."""

from dataclasses import dataclass
from typing import Optional

from decuen.dists import Distribution
from decuen.structs.action import Action
from decuen.structs.state import State


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class Transition:
    """Representation of a transition.

    A transition is minimally defined by an initial state, an action taken on that state, and the resultant state. In
    the context of finite-horizon reinforcement learning, we also require two additional pieces of data: a reward
    attained due to taking that action in that state and whether or not this transition ends in a terminal state, i.e.
    one in which the simulation ends.
    """

    state: State
    action: Action
    new_state: State
    reward: float
    terminal: bool

    behavior: Optional[Distribution] = None
    state_value: Optional[float] = None
    action_value: Optional[float] = None
# pylint: enable=too-many-instance-attributes
