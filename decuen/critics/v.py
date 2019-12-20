"""Deep state value action critics."""

from dataclasses import dataclass

from torch import zeros_like
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore

from decuen.critics._critic import Critic, CriticSettings
from decuen.structs import (Experience, State, Tensor, gather_new_states,
                            gather_rewards, gather_states, gather_terminals)
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

        final_layer, self.network = finalize_module(model, State(self.state_space.sample()), 1)
        self.settings.optimizer.add_param_group({"params": final_layer.parameters()})

    def learn(self, experience: Experience) -> None:
        """Update internal critic representation based on past transitions."""
        self._learn_step += 1
        if not experience:
            return

        states = gather_states(experience)
        new_states = gather_new_states(experience)
        rewards = gather_rewards(experience)
        terminals = gather_terminals(experience)

        next_values = zeros_like(rewards)
        if (~terminals).any():
            new_states_not_terminal = new_states[~terminals]
            next_values[~terminals] = self.network(new_states_not_terminal).detach().squeeze(1)
        target_values = rewards + self.settings.discount_factor * next_values

        values = self.network(states).squeeze(1)

        loss = self.settings.loss(values, target_values)
        self.settings.optimizer.zero_grad()
        loss.backward()
        self.settings.optimizer.step()

    def crit(self, state: State) -> Tensor:
        """Estimate the quality of a state or tensor of states."""
        return self.network(state).detach().squeeze(1)

    def advantage(self, experience: Experience) -> Tensor:
        """Estimate the advantage of every step in an experience."""
        states = gather_states(experience)
        new_states = gather_new_states(experience)
        rewards = gather_rewards(experience)

        values = self.network(states).detach()
        new_values = self.network(new_states).detach()
        return rewards + (self.settings.discount_factor * new_values) - values
