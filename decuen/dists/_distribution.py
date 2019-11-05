"""Interfaces for arbitrary distributions, single- or multi-variate."""

from abc import ABC, abstractmethod

import numpy as np  # type: ignore


class Distribution(ABC):
    """Arbitrary parameterized distribution abstraction belonging to a particular family of distributions.

    This abstraction provides interfaces for high-level functionality common to all distributions and frequently used
    within reinforcement learning literature and consequently the library.
    """

    parameters: np.ndarray

    def __init__(self, parameters: np.ndarray) -> None:
        """Initialize a parameterized distribution."""
        self.parameters = parameters

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Sample at random from the distribution."""
        ...

    @abstractmethod
    def pdf(self, sample: np.ndarray) -> float:
        """Return the probability of observing a sample."""
        ...

    @abstractmethod
    def log_pdf(self, sample: np.ndarray) -> float:
        """Return the log of the probability of observing a sample."""
        ...

    @abstractmethod
    def entropy(self) -> float:
        """Return the differential entropy of this distribution."""
        ...

    @abstractmethod
    def kl_divergence(self, other: 'Distribution') -> float:
        """Return the Kullback-Leibler divergence between this distribution and another one."""
        ...
