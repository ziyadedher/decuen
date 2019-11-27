"""Deep state value action critics."""

from dataclasses import dataclass

from torch import from_numpy, zeros_like  # pylint: disable=no-name-in-module
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore

from decuen.critics._critic import Critic, CriticSettings
from decuen.structs import Experience, State, Tensor, batch_experience
from decuen.utils.module_construction import finalize_module


@dataclass
class StateValueCriticSettings(CriticSettings):
    """Settings for Q-network critics."""

    optimizer: Optimizer
    loss: Module


class StateValueCritic(Critic):
    """State-value critic."""

    settings: StateValueCriticSettings
    network: Module

    def __init__(self, model: Module, settings: StateValueCriticSettings) -> None:
        """Initialize this generic actor critic interface."""
        super().__init__(settings)

        final_layer, self.network = finalize_module(model, from_numpy(self.state_space.sample()).float(), 1)

        self.settings.optimizer.add_param_group({"params": final_layer.parameters()})

    def learn(self, experience: Experience) -> None:
        """Update internal critic representation based on past transitions."""
        self._learn_step += 1
        if not experience:
            return

        batch = batch_experience(experience)

        new_states_not_terminal = batch.new_states[~batch.terminals]
        future_values = zeros_like(batch.rewards)
        future_values[~batch.terminals] = self.crit(new_states_not_terminal)
        target_values = batch.rewards + self.settings.discount_factor * future_values

        values = self.network(batch.states).squeeze(1)

        loss = self.settings.loss(values, target_values)
        self.settings.optimizer.zero_grad()
        loss.backward()
        self.settings.optimizer.step()

    def crit(self, state: State) -> Tensor:
        """Estimate the quality of a state or tensor of states."""
        return self.network(state).detach().squeeze(1)

    def advantage(self, experience: Experience) -> Tensor:
        """Estimate the advantage of every step in an experience."""
        batch = batch_experience(experience)
        new_values = self.network(batch.new_states).detach().squeeze(1)
        values = self.network(batch.states).detach().squeeze(1)
        advantages = batch.rewards + (self.settings.discount_factor * new_values) - values
        return advantages
