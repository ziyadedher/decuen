"""Implementation of a strategy-based actor."""

from dataclasses import dataclass

from gym.spaces import Discrete  # type: ignore
from torch import arange

from decuen.actors._actor import Actor, ActorSettings
from decuen.actors.strats import Strategy
from decuen.critics import QValueCritic
from decuen.structs import Experience, State, Tensor


@dataclass
class StrategyActorSettings(ActorSettings):
    """Settings for strategy-based actors."""


class StrategyActor(Actor[QValueCritic]):
    """Strategy-based actor.

    Generates a policy purely based on critic values and a mechanism of policy extraction called a strategy.
    """

    settings: StrategyActorSettings

    def __init__(self, strategy: Strategy, settings: StrategyActorSettings) -> None:
        """Initialize a strategy actor."""
        super().__init__(settings)
        self.strategy = strategy

    def learn(self, experience: Experience) -> None:
        """Do nothing. Learning is not supported for strategy-based actors."""

    def _gen_policy_params(self, state: State) -> Tensor:
        """Generate policy parameters on-the-fly based on an environment state."""
        if not self.critic:
            raise ValueError("strategy actor must be assigned a critic")
        if not isinstance(self.action_space, Discrete):
            raise NotImplementedError("strategy actor does not support non-discrete action spaces")

        return self.strategy.act(self.critic.crit(state, arange(self.action_space.n)))
