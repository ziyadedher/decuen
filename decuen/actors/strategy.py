"""Implementation of a strategy-based actor."""

from typing import MutableSequence

from gym.spaces import Discrete  # type: ignore

from decuen.actors._actor import Actor, ActorSettings
from decuen.actors.strats import Strategy
from decuen.structs import State, Tensor, Trajectory, arange


class StrategyActor(Actor):
    """Strategy-based actor.

    Generates a policy purely based on critic values and a mechanism of policy extraction called a strategy.
    """

    def __init__(self, strategy: Strategy) -> None:
        """Initialize a strategy actor."""
        super().__init__(ActorSettings(dist=strategy.distribution_type))
        self.strategy = strategy

    def learn(self, trajectories: MutableSequence[Trajectory]) -> None:
        """Do nothing. Learning is not supported for strategy-based actors."""

    def _generate_policy_parameters(self, state: State) -> Tensor:
        """Generate policy parameters on-the-fly based on an environment state."""
        if not self.critic:
            raise ValueError("strategy actor must be assigned a critic")
        if not isinstance(self.action_space, Discrete):
            raise NotImplementedError("strategy actor does not support non-discrete action spaces")

        return self.strategy.act(self.critic.crit(state, arange(self.action_space.n)))
