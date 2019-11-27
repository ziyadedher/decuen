"""Implementation of a policy-gradient actor-learner.

Based on REINFORCE algorithm with causality and baselines.
"""

from dataclasses import dataclass

from torch import from_numpy
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore

from decuen.actors._actor import Actor, ActorSettings
from decuen.critics import Critic
from decuen.structs import (Experience, State, Tensor, Trajectory,
                            batch_experience)
from decuen.utils.module_construction import finalize_module


@dataclass
class PGActorSettings(ActorSettings):
    """Basic common settings for all actor-learners."""

    optimizer: Optimizer


class PGActor(Actor[Critic]):
    """Policy-gradient actor-learner.

    Uses a function approximator to generate the parameters for a policy and improves that estimator.
    """

    settings: PGActorSettings

    def __init__(self, model: Module, settings: PGActorSettings) -> None:
        """Initialize a policy-gradient actor-learner."""
        super().__init__(settings)

        final_layer, self.network = finalize_module(model, from_numpy(self.state_space.sample()).float(),
                                                    self._num_policy_params)
        self.settings.optimizer.add_param_group({"params": final_layer.parameters()})

    def learn(self, experience: Experience) -> None:
        """Update policy based on an experience."""
        if not experience:
            return

        if isinstance(experience, Trajectory):
            batch = experience.batched
        else:
            batch = batch_experience(experience)

        policies = self.act(batch.states)
        neglogs = -policies.log_prob(batch.actions)

        advantages = self.critic.advantage(experience)
        advantages -= advantages.mean()
        advantages /= advantages.std()
        loss = (neglogs * advantages).sum()

        self.settings.optimizer.zero_grad()
        loss.backward()
        self.settings.optimizer.step()

    def _gen_policy_params(self, state: State) -> Tensor:
        """Generate policy parameters on-the-fly based on an environment state."""
        return self.network(state)
