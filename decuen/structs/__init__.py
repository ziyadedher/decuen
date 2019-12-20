"""Common structures and representations to be passed within the framework."""

from decuen.structs.action import Action
from decuen.structs.experience import (Experience, gather_actions,
                                       gather_new_states, gather_rewards,
                                       gather_states, gather_terminals)
from decuen.structs.state import State
from decuen.structs.tensor import Tensor, tensor
from decuen.structs.trajectory import Trajectory
from decuen.structs.transition import Transition

__all__ = [
    "Tensor", "Action", "State", "Transition", "Trajectory", "Experience",
    "tensor", "gather_actions", "gather_new_states", "gather_rewards", "gather_states", "gather_terminals"
]
