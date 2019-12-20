"""Workaround for the 'tensor is not callable' pylint error.

Simply contains a forwarder of the torch tensor function that forces the linter to agree that the function is, in fact,
a function.
"""

import torch

Tensor = torch.Tensor

def tensor(*args, **kwargs) -> torch.Tensor:  # noqa
    return torch.tensor(*args, **kwargs)  # noqa


tensor.__doc__ = torch.tensor.__doc__
