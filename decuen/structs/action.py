"""Constructs representing and helpers for actions in the framework."""

from typing import Union

import numpy as np  # type: ignore
from gym.spaces import Discrete  # type: ignore
from torch import Tensor, from_numpy

from decuen.utils.context import Contextful


# TODO: hook in action checking here?
class Action(Contextful):
    """Action representation.

    Essentially a thin wrapper around some data and provides hooks to access either the array data or the tensor data.
    """

    _data: Tensor

    def __init__(self, data: Union[np.ndarray, float, int]) -> None:
        """Initialize an action."""
        super().__init__()
        self._data = from_numpy(np.array(data))
        if not isinstance(self.action_space, Discrete):
            self._data = self._data.float()

    @property
    def numpy(self) -> np.ndarray:
        """Return a numpy array representation of this action."""
        return self._data.numpy()

    @property
    def tensor(self) -> np.ndarray:
        """Return a torch tensor representation of this action."""
        return self._data
