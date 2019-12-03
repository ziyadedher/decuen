"""Constructs representing states (which we conflate with observations) in the framework."""

import numpy as np  # type: ignore
from torch import Tensor, from_numpy


# TODO: hook in state checking here?
class State:
    """State (observation) representation.

    Essentially a thin wrapper around some data and provides hooks to access either the array data or the tensor data.
    """

    _data: Tensor

    def __init__(self, data: np.ndarray) -> None:
        """Initialize a state."""
        self._data = from_numpy(data)

    @property
    def numpy(self) -> np.ndarray:
        """Return a numpy array representation of this state."""
        return self._data.numpy()

    @property
    def tensor(self) -> np.ndarray:
        """Return a torch tensor representation of this state."""
        return self._data
