"""Collection of common structures subject to expansion or modification.

These structures should only be used internally within the framework and their APIs should never be considered stable.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from decuen.dists import Distribution

State = torch.Tensor
Action = torch.Tensor
Tensor = torch.Tensor


def tensor(*args, **kwargs) -> torch.Tensor:  # noqa
    return torch.tensor(*args, **kwargs)  # noqa


tensor.__doc__ = torch.tensor.__doc__


@dataclass
class BatchedTransitions:
    """Represents a batch of transitions packaged in the format expected by our training procedures."""

    states: torch.Tensor
    actions: torch.Tensor
    new_states: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor


@dataclass
class Transition:
    """Simple data structure representing a transition from one state to another with associated information."""

    state: State
    action: Action
    new_state: State
    reward: float
    terminal: bool

    behavior: Optional[Distribution] = None
    state_value: Optional[float] = None
    action_value: Optional[float] = None


@dataclass
class Trajectory:
    """Simple data structure representing a trajectory."""

    transitions: Sequence[Transition]

    @property
    def batched(self) -> BatchedTransitions:
        """Return the transitions in this trajectories in batched format."""
        return batch_experience(self.transitions)


Experience = Union[Trajectory, Sequence[Transition]]


def batch_experience(experience: Experience) -> BatchedTransitions:
    """Batch an experience into the format expected by our training procedures."""
    if isinstance(experience, Trajectory):
        return experience.batched

    states = torch.stack([transition.state for transition in experience])
    actions = torch.stack([transition.action for transition in experience])
    new_states = torch.stack([transition.new_state for transition in experience])
    rewards = tensor([transition.reward for transition in experience])
    terminals = tensor([transition.terminal for transition in experience])
    return BatchedTransitions(states, actions, new_states, rewards, terminals)
