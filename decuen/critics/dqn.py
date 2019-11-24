"""Deep Q-network action critics.

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
from typing import MutableSequence

from gym.spaces import Discrete  # type: ignore
from torch import from_numpy, zeros  # pylint: disable=no-name-in-module
from torch.nn import Linear, Module, Sequential
from torch.optim import Optimizer  # type: ignore

from decuen.critics._q import QCritic, QCriticSettings
from decuen.structs import Action, State, Transition


@dataclass
class DQNCriticSettings(QCriticSettings):
    """Settings for Q-table critics."""

    clipped: bool
    optimizer: Optimizer
    loss: Module


class DQNCritic(QCritic):
    """Deep Q-network critic.

    See [1], [2], [3] for more details.
    """

    action_space: Discrete
    settings: DQNCriticSettings
    network: Module
    _target_network: Module

    def __init__(self, model: Module, settings: DQNCriticSettings) -> None:
        """Initialize this generic actor critic interface."""
        super().__init__(settings)

        # TODO: possibly generalize to multi-discrete spaces, maybe continuous as well with a separate formulation
        if not isinstance(self.action_space, Discrete):
            raise TypeError("action space for Q-network critic must be discrete")

        # TODO: generalize this to other classes
        try:
            size = model(from_numpy(self.state_space.sample())).size()
        except RuntimeError:
            raise ValueError("given model is incompatible with the state space")
        if len(size) != 1:
            raise ValueError(f"given model must have one-dimensional output, instead got output shape {size}")

        final_layer = Linear(size[0], self.action_space.n)
        self.network = Sequential(model, final_layer)
        self._target_network = copy.deepcopy(self.network)
        self._target_network.eval()

        self.settings.optimizer.add_param_group({"params": final_layer.parameters()})

    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Update internal critic representation based on past transitions."""
        self._learn_step += 1
        if not transitions:
            return

        batch = Transition.batch(transitions)

        values = self.network(batch.states).gather(1, batch.actions.unsqueeze(1))
        new_states_not_terminal = batch.new_states[~batch.terminals]

        next_values = zeros(len(transitions))
        if self.settings.double:
            chosen_actions = self._target_network(new_states_not_terminal).argmax(1, keepdims=True)
            next_values[~batch.terminals] = (self.network(new_states_not_terminal)
                                             .gather(1, chosen_actions).squeeze(1).detach())
        else:
            next_values[~batch.terminals] = self._target_network(new_states_not_terminal).max(1)[0].detach()
        target_values = (batch.rewards + (self.settings.discount_factor * next_values)).unsqueeze(1)

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

    def crit(self, state: State, action: Action) -> float:
        """Return the Q-value of taking a specific action in a specific state."""
        return float(self.values(state)[action])

    def values(self, state: State) -> Action:
        """Return an array of Q-values of all actions in a specific state."""
        return self.network(state.unsqueeze(0)).detach()[0]
