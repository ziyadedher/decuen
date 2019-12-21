"""Constructs representing and helpers for states (which we conflate with observations) in the framework."""

from typing import Union

import numpy as np  # type: ignore
from gym.spaces import Discrete  # type: ignore
from torch import Tensor, from_numpy

from decuen.utils.context import Contextful


# TODO: hook in state checking here?
class State(Contextful):
    """State (observation) representation.

    Essentially a thin wrapper around some data and provides hooks to access either the array data or the tensor data.
    """

    _data: Tensor

    def __init__(self, data: Union[np.ndarray, float, int]) -> None:
        """Initialize a state."""
        super().__init__()
        self._data = from_numpy(np.array(data))
        if not isinstance(self.state_space, Discrete):
            self._data = self._data.float()

    @property
    def numpy(self) -> np.ndarray:
        """Return a numpy array representation of this state."""
        return self._data.numpy()

    @property
    def tensor(self) -> np.ndarray:
        """Return a torch tensor representation of this state."""
        return self._data
