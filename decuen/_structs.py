from dataclasses import dataclass
from typing import MutableSequence, Optional, Sequence

import torch

from decuen.dists._distribution import Distribution

State = torch.Tensor
Action = torch.Tensor
Tensor = torch.Tensor


def tensor(*args, **kwargs) -> torch.Tensor:
    return torch.tensor(*args, **kwargs)  # noqa


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

    @staticmethod
    def batch(transitions: MutableSequence['Transition']) -> BatchedTransitions:
        """Batch a sequence of transitions into the format expected by our training procedures."""
        states = torch.stack([transition.state for transition in transitions])
        actions = torch.stack([transition.action for transition in transitions])
        new_states = torch.stack([transition.new_state for transition in transitions])
        rewards = tensor([transition.reward for transition in transitions])
        terminals = tensor([transition.terminal for transition in transitions])

        return BatchedTransitions(states, actions, new_states, rewards, terminals)


Trajectory = Sequence[Transition]
