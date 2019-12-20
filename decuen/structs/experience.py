"""Constructs representing and helpers for experiences in the framework.

An experience is simply either a trajectory (sequential transitions) or an arbitrary sequence of transitions.
"""

from typing import Sequence, Union

from torch import Tensor, stack

from decuen.structs.tensor import tensor
from decuen.structs.trajectory import Trajectory
from decuen.structs.transition import Transition

Experience = Union[Trajectory, Sequence[Transition]]


# TODO: create a function constructor to reduce code duplication.
# TODO: create some function container to reduce import complexity

def gather_states(experience: Experience) -> Tensor:
    """Extract and stack all the states from an experience in the same order they appear in.

    This helper function, and the rest of the experience helper functions, are very useful when trying to pass
    underlying experience data through optimization frameworks that work faster on batched data, e.g. most deep learning
    frameworks and symbolic computation engines.
    """
    return stack([transition.state.tensor for transition in experience])


def gather_actions(experience: Experience) -> Tensor:
    """Extract and stack all the actions from an experience in the same order they appear in.

    This helper function, and the rest of the experience helper functions, are very useful when trying to pass
    underlying experience data through optimization frameworks that work faster on batched data, e.g. most deep learning
    frameworks and symbolic computation engines.
    """
    return stack([transition.action.tensor for transition in experience])


def gather_new_states(experience: Experience) -> Tensor:
    """Extract and stack all the new states from an experience in the same order they appear in.

    This helper function, and the rest of the experience helper functions, are very useful when trying to pass
    underlying experience data through optimization frameworks that work faster on batched data, e.g. most deep learning
    frameworks and symbolic computation engines.
    """
    return stack([transition.new_state.tensor for transition in experience])


def gather_rewards(experience: Experience) -> Tensor:
    """Extract and stack all the rewards from an experience in the same order they appear in.

    This helper function, and the rest of the experience helper functions, are very useful when trying to pass
    underlying experience data through optimization frameworks that work faster on batched data, e.g. most deep learning
    frameworks and symbolic computation engines.
    """
    return tensor([transition.reward for transition in experience])


def gather_terminals(experience: Experience) -> Tensor:
    """Extract and stack all the terminal statuses from an experience in the same order they appear in.

    This helper function, and the rest of the experience helper functions, are very useful when trying to pass
    underlying experience data through optimization frameworks that work faster on batched data, e.g. most deep learning
    frameworks and symbolic computation engines.
    """
    return tensor([transition.terminal for transition in experience])
