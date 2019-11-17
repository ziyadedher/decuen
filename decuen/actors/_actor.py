"""Interface for arbitrary actor-learners and respective settings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import MutableSequence, Optional

import numpy as np  # type: ignore

from decuen.dists._distribution import Distribution
from decuen.memories._memory import Trajectory
from decuen.policies._policy import Policy


@dataclass
class ActorSettings:
    """Basic common settings for all actor-learners."""


class Actor(ABC):
    """Generic abstract actor-learner interface.

    This abstraction provides interfaces for the two main functionalities of an actor-learner:
        1. the ability to choose an action to perform given a state, and
        2. the ability to learn based on past transitions and trajectories.
    """

    policy: Policy

    @abstractmethod
    def __init__(self, distribution: Optional[Distribution] = None,
                 settings: ActorSettings = ActorSettings()) -> None:
        """Initialize a generic actor-learner."""
        self.policy = Policy(self._generate_policy_parameters,
                             distribution if distribution else self._choose_action_distribution())

    def act(self, state: np.ndarray) -> np.ndarray:
        """Choose an action to perform based on an environment state."""
        return self.policy.act(state).rvs()

    # TODO: support learning from transitions
    # XXX: possibly return loss or some other metric?
    @abstractmethod
    def learn(self, trajectories: MutableSequence[Trajectory]) -> None:
        """Update policy based on past trajectories."""
        ...

    @abstractmethod
    def _generate_policy_parameters(self, state: np.ndarray) -> np.ndarray:
        """Generate policy parameters on-the-fly based on an environment state."""
        ...

    def _choose_action_distribution(self) -> np.ndarray:
        """Choose an action distribution for the policy depending on the action space."""
        # TODO: implement
