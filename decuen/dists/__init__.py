"""Interfaces and implementations of different distributions for use in policies.

Most of the "implementations" are direct forwardings of pytorch distributions.
"""

import torch
from torch.distributions import *  # noqa

__all__ = torch.distributions.__all__
