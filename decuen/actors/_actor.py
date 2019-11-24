"""Interface for arbitrary actor-learners and respective settings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import MutableSequence, Optional, Type

from decuen.critics import ActionCritic
from decuen.dists import Distribution
from decuen.policy import Policy
from decuen.structs import State, Tensor, Trajectory
from decuen.utils.context import Contextful


@dataclass
class ActorSettings:
    """Basic common settings for all actor-learners."""

    dist: Type[Distribution]


class Actor(ABC, Contextful):
    """Generic abstract actor-learner interface.

    This abstraction provides interfaces for the two main functionalities of an actor-learner:
        1. the ability to choose an action to perform given a state, and
        2. the ability to learn based on past transitions and trajectories.
    """

    settings: ActorSettings
    policy: Policy
    critic: Optional[ActionCritic]

    @abstractmethod
    def __init__(self, settings: ActorSettings) -> None:
        """Initialize a generic actor-learner."""
        super().__init__()
        self.settings = settings
        self.policy = Policy(self._generate_policy_parameters, self.settings.dist)
        self.critic = None

    def act(self, state: State) -> Distribution:
        """Construct a parameterized policy and return the generated distribution."""
        return self.policy.act(state)

    # TODO: support learning from transitions
    # XXX: possibly return loss or some other metric?
    @abstractmethod
    def learn(self, trajectories: MutableSequence[Trajectory]) -> None:
        """Update policy based on past trajectories."""
        ...

    @abstractmethod
    def _generate_policy_parameters(self, state: State) -> Tensor:
        """Generate policy parameters on-the-fly based on an environment state."""
        ...
