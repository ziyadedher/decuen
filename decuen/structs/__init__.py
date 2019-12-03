"""Common structures and representations to be passed within the framework."""

from decuen.structs.action import Action
from decuen.structs.experience import (Experience, actions, new_states,
                                       rewards, states, terminals)
from decuen.structs.state import State
from decuen.structs.tensor import tensor
from decuen.structs.trajectory import Trajectory
from decuen.structs.transition import Transition

__all__ = [
    "Action", "State", "Transition", "Trajectory", "Experience",
    "tensor", "actions", "new_states", "rewards", "states", "terminals"
]
