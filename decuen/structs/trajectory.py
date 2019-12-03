"""Constructs representing and helpers for trajectories in the framework."""

from dataclasses import dataclass
from typing import Generator, Sequence

from decuen.structs.transition import Transition


# TODO: hook in checks for continuity here?
@dataclass(frozen=True)
class Trajectory:
    """Representation of a trajectory of transitions.

    A trajectory is simply a time-sequential and continuous collection of transitions, i.e. a sequence of transitions
    such that each transition's new state is the state of the initial state of the next transition in the sequence.
    """

    _transitions: Sequence[Transition]

    def __iter__(self) -> Generator[Transition, None, None]:
        """Iterate through the underlying transitions in this trajectory."""
        yield from self._transitions
