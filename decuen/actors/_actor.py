"""Interface for arbitrary actor-learners and respective settings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import MutableSequence, Optional, Type

from gym.spaces import Box, Discrete  # type: ignore

from decuen.critics import ActionCritic
from decuen.dists import Categorical, Distribution, MultivariateNormal, Normal
from decuen.structs import State, Tensor, Trajectory, eye
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
    critic: Optional[ActionCritic]

    @abstractmethod
    def __init__(self, settings: ActorSettings) -> None:
        """Initialize a generic actor-learner."""
        super().__init__()
        self.settings = settings
        self.critic = None

    def act(self, state: State) -> Distribution:
        """Construct a parameterized policy and return the generated distribution."""
        return self._gen_behaviour(self._gen_policy_params(state))

    # TODO: support learning from transitions
    # XXX: possibly return loss or some other metric?
    @abstractmethod
    def learn(self, trajectories: MutableSequence[Trajectory]) -> None:
        """Update policy based on past trajectories."""
        ...

    @abstractmethod
    def _gen_policy_params(self, state: State) -> Tensor:
        """Generate policy parameters on-the-fly based on an environment state."""
        ...

    @property
    def _num_policy_params(self) -> int:
        """Calculate the number of parameters needed for the policy."""
        if not any(isinstance(self.action_space, space_type) for space_type in (Discrete, Box)):
            raise TypeError("actors only support Discrete, Box action spaces")

        if self.settings.dist is Categorical:
            if not isinstance(self.action_space, Discrete):
                raise TypeError("categorical distributions for actions can only be used for a Discrete action space")
            return self.action_space.n

        if self.settings.dist is Normal:
            if isinstance(self.action_space, Discrete):
                return 2
            if isinstance(self.action_space, Box):
                if self.action_space.shape != (1,):
                    raise TypeError("univariate normal distribution can only be used with unidimensional action spaces")
                return 2

        if self.settings.dist is MultivariateNormal:
            if isinstance(self.action_space, Discrete):
                raise TypeError("mutivariate normal distribution cannot be used with Discrete action spaces")
            if isinstance(self.action_space, Box):
                return 2 * reduce((lambda x, y: x * y), self.action_space.shape)

        raise NotImplementedError("actors do not support this action distribution yet")

    def _gen_behaviour(self, params: Tensor) -> Distribution:
        """Generate the behavioural policy based on the given parameters and the distribution family of this actor."""
        # TODO: check for parameter size mismatches

        if self.settings.dist is Categorical:
            return Categorical(params)

        if self.settings.dist is Normal:
            return Normal(params[0], params[1])

        if self.settings.dist is MultivariateNormal:
            half = params.size()[0] / 2
            return MultivariateNormal(params[:half], eye(params[:half]))

        raise NotImplementedError("actors do not support this action distribution yet")
