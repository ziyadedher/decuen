"""Implementation of a policy-gradient actor-learner.

Based on REINFORCE algorithm with causality and baselines.
"""

from dataclasses import dataclass
from typing import List

from torch import stack
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore

from decuen.actors._actor import Actor, ActorSettings
from decuen.critics import Critic
from decuen.structs import Experience, State, Tensor, gather_actions, tensor
from decuen.utils.module_construction import finalize_module


@dataclass(frozen=True)
class PGActorSettings(ActorSettings):
    """Basic common settings for all actor-learners."""

    normalize: bool
    optimizer: Optimizer


class PGActor(Actor[Critic]):
    """Policy-gradient actor-learner.

    Uses a function approximator to generate the parameters for a policy and improves that estimator.
    """

    settings: PGActorSettings

    def __init__(self, model: Module, settings: PGActorSettings) -> None:
        """Initialize a policy-gradient actor-learner."""
        super().__init__(settings)

        final_layer, self.network = finalize_module(model, State(self.state_space.sample()), self._num_params)
        self.settings.optimizer.add_param_group({"params": final_layer.parameters()})

    def learn(self, experience: Experience) -> None:
        """Update policy based on an experience."""
        if not experience:
            return

        actions = gather_actions(experience)

        policies = self.policy([transition.state for transition in experience], stack=True)
        neglogs = -policies.log_prob(actions)

        advantages = tensor(self.critic.advantage(experience))
        if self.settings.normalize:
            advantages -= advantages.mean()
            advantages /= advantages.std()

        loss = (neglogs * advantages).sum()

        self.settings.optimizer.zero_grad()
        loss.backward()
        self.settings.optimizer.step()

    def _params(self, states: List[State]) -> Tensor:
        return self.network(stack([state.tensor for state in states]))
