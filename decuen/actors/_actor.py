"""Interface for arbitrary actor-learners and respective settings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import (Generic, List, Literal, Optional, Type, TypeVar, Union,
                    overload)

from gym.spaces import Box, Discrete  # type: ignore
from torch import diag_embed
from torch.nn.functional import softplus

from decuen.critics import Critic
from decuen.dists import Categorical, Distribution, MultivariateNormal, Normal
from decuen.structs import Action, Experience, State, Tensor
from decuen.utils.context import Contextful


@dataclass(frozen=True)
class ActorSettings:
    """Basic common settings for all actor-learners."""

    dist: Type[Distribution]
    discount_factor: float


CriticType = TypeVar("CriticType", bound=Critic)


class Actor(Generic[CriticType], ABC, Contextful):
    # TODO: update docstring
    """Generic abstract actor-learner interface.

    This abstraction provides interfaces for the two main functionalities of an actor-learner:
        1. the ability to choose an action to perform given a state, and
        2. the ability to learn based on past transitions and trajectories.
    """

    settings: ActorSettings

    _critic: Optional[CriticType]

    @abstractmethod
    def __init__(self, settings: ActorSettings) -> None:
        """Initialize a generic actor-learner."""
        super().__init__()
        self.settings = settings

        self._critic = None

    @property
    def critic(self) -> CriticType:
        """Get the critic associated with this actor."""
        if not self._critic:
            raise ValueError("no critic associated with this actor")
        return self._critic

    @critic.setter
    def critic(self, critic: CriticType) -> None:
        """Set the critic of this actor.

        You probably do not want to do this manually.
        """
        self._critic = critic

    @overload
    def policy(self, states: State, *, stack: bool = False) -> Distribution: ...  # noqa

    @overload
    def policy(self, states: List[State], *, stack: Literal[True]) -> Distribution: ...  # noqa

    @overload
    def policy(self, states: List[State], *, stack: Literal[False]) -> List[Distribution]: ...  # noqa

    @overload
    def policy(self, states: List[State], *, stack: bool = False) -> Union[Distribution, List[Distribution]]: ...  # noqa

    def policy(self, states: Union[State, List[State]], *,  # noqa
               stack: bool = False) -> Union[Distribution, List[Distribution]]:
        """Construct parameterized policies that can be sampled to choose actions based on the given states.

        This method provides an interface to generating actions from this agent by sampling from an action distribution.
        To perform a more streamlined action selection in cases where the underlying behavioural distribution is not
        needed, use `Agent.act` instead.

        When given a single state, this method produces a single distribution over the agent's action space representing
        the behavioural policy under that state.

        When given a list of states, this method, by default, produces a respective list of distributions each over the
        agent's action space representing the behavioural policies under the different states. If <stack> is set to
        `True`, then the different distributions are combined into a single distribution that is more efficient to
        sample from than from each distribution in sequence. This can be used in cases where multiple actions across
        different states are needed at the same time as is the case with most multi-worker trainers.
        """
        if isinstance(states, State):
            if stack is not False:
                print("setting <stack> has no effect when not given a list of states")
            return self._behaviour(self._params([states]))

        params = self._params(states)
        return self._behaviour(params) if stack else [self._behaviour(subparams) for subparams in params.unbind()]

    @overload
    def act(self, states: State) -> Action: ...  # noqa

    @overload
    def act(self, states: List[State]) -> List[Action]: ...  # noqa

    def act(self, states: Union[State, List[State]]) -> Union[Action, List[Action]]:  # noqa
        """Choose actions to perform based on the given states.

        When given a single state, this method produces a single action sampled from the underlying generated policy.

        When given a list of states, this method produces a respective list of actions sampled from the generated
        policies.
        """
        if isinstance(states, State):
            policy = self.policy(states)
            return policy.sample().unbind()
        policy = self.policy(states, stack=True)
        return list(policy.sample().unbind())

    # XXX: possibly return loss or some other metric?
    @abstractmethod
    def learn(self, experience: Experience) -> None:
        """Update policy based on an experience."""
        ...

    @abstractmethod
    def _params(self, states: List[State]) -> Tensor:
        """Generate policy parameters on-the-fly based on the given states."""
        ...

    @cached_property
    def _num_params(self) -> int:
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

    def _behaviour(self, params: Tensor) -> Distribution:
        """Generate a behavioural policy based on the given parameters and the distribution family of this actor."""
        # TODO: check for parameter size mismatches

        if len(params.size()) != 2:
            # FIXME: better error message
            raise ValueError("unknown dimensionality")

        if self.settings.dist is Categorical:
            return Categorical(logits=params)

        if self.settings.dist is Normal:
            return Normal(params[:, 0], params[:, 1])

        if self.settings.dist is MultivariateNormal:
            half = params.size()[1] // 2
            return MultivariateNormal(params[:, :half], diag_embed(softplus(params[:, half:])))

        raise NotImplementedError("actors do not support this action distribution yet")
