"""Interfaces for deep Q critics.

Implements both deep Q-learning [1, 2] and double deep Q-learning [3] algorithms to train action critics.

[1] Mnih, Volodymyr; et al. (2013). Playing Atari with Deep Reinforcement Learning
    https://arxiv.org/pdf/1312.5602.pdf
[2] Mnih, Volodymyr; et al. (2015). Human-level control through deep reinforcement learning
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
[3] van Hasselt, Hado; et al. (2015). Deep Reinforcement Learning with Double Q-learning
    https://arxiv.org/pdf/1509.06461.pdf
"""

import copy
from dataclasses import dataclass

from gym.spaces import Discrete  # type: ignore
from torch import zeros_like
from torch.nn import Module
from torch.optim import Optimizer  # type: ignore

from decuen.critics._critic import Critic, CriticSettings
from decuen.structs import (Action, Experience, State, Tensor, gather_actions,
                            gather_new_states, gather_rewards, gather_states,
                            gather_terminals)
from decuen.utils.module_construction import finalize_module


@dataclass
class QValueCriticSettings(CriticSettings):
    """Settings for Q-network critics."""

    target_update: int
    double: bool
    clipped: bool
    optimizer: Optimizer
    loss: Module


class QValueCritic(Critic):
    """Deep Q critic, or action-value critic.

    See [1], [2], [3] for more details.
    """

    action_space: Discrete
    settings: QValueCriticSettings
    network: Module
    _target_network: Module

    def __init__(self, model: Module, settings: QValueCriticSettings) -> None:
        """Initialize this generic actor critic interface."""
        super().__init__(settings)

        # TODO: possibly generalize to multi-discrete spaces, maybe continuous as well with a separate formulation
        if not isinstance(self.action_space, Discrete):
            raise TypeError("action space for Q-network critic must be discrete")

        final_layer, self.network = finalize_module(model, State(self.state_space.sample()), self.action_space.n)
        self._target_network = copy.deepcopy(self.network)
        self._target_network.eval()

        self.settings.optimizer.add_param_group({"params": final_layer.parameters()})

    def learn(self, experience: Experience) -> None:
        """Update internal critic representation based on an experience."""
        self._learn_step += 1
        if not experience:
            return

        states = gather_states(experience)
        actions = gather_actions(experience)
        new_states = gather_new_states(experience)
        rewards = gather_rewards(experience)
        terminals = gather_terminals(experience)

        values = self.network(states).gather(1, actions.unsqueeze(1))
        new_states_not_terminal = new_states[~terminals]

        next_values = zeros_like(rewards)
        if self.settings.double:
            chosen_actions = self._target_network(new_states_not_terminal).argmax(1, keepdims=True)
            next_values[~terminals] = (self.network(new_states_not_terminal)
                                       .gather(1, chosen_actions).detach().squeeze(1))
        else:
            next_values[~terminals] = self._target_network(new_states_not_terminal).detach().max(1)[0]
        target_values = (rewards + (self.settings.discount_factor * next_values)).unsqueeze(1)

        loss = self.settings.loss(values, target_values)
        self.settings.optimizer.zero_grad()
        loss.backward()
        if self.settings.clipped:
            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
        self.settings.optimizer.step()

        if self._learn_step % self.settings.target_update == 0:
            self._target_network.load_state_dict(self.network.state_dict())

    def crit(self, state: State, action: Action) -> Tensor:
        """Estimate the quality of taking an action or tensor of actions in a state."""
        return self.network(state).detach()[action]

    def advantage(self, experience: Experience) -> Tensor:
        """Estimate the advantage of every step in an experience."""
        states = gather_states(experience)
        actions = gather_actions(experience)
        return self.network(states).detach().gather(1, actions.unsqueeze(1))
