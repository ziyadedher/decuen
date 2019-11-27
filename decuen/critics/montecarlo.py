"""Implementation of a rudimentary Monte Carlo "reward" critic.

Simply estimates the state values based on a Monte Carlo estimate of the expected reward.
"""

from dataclasses import dataclass

from decuen.critics._critic import Critic, CriticSettings
from decuen.structs import (Experience, Tensor, Trajectory, batch_experience,
                            tensor)


@dataclass
class MonteCarloCriticSettings(CriticSettings):
    """Settings for Monte Carlo critics."""


class MonteCarloCritic(Critic):
    """Monte Carlo critic."""

    def __init__(self, settings: MonteCarloCriticSettings) -> None:
        """Initialize a Monte Carlo critic."""
        super().__init__(settings)

    def learn(self, experience: Experience) -> None:
        """Do nothing. Monte Carlo critic does not learn."""

    def advantage(self, experience: Experience) -> Tensor:
        """Estimate the advantage of every step in an experience by using a Monte Carlo sampling."""
        if isinstance(experience, Trajectory):
            rewards = self._calculate_trajectory_rewards(experience)
            return rewards
        return batch_experience(experience).rewards[:]

    def _calculate_trajectory_rewards(self, trajectory: Trajectory):
        batch = trajectory.batched
        rewards = batch.rewards[:]

        running = tensor(0.)
        for i in reversed(range(rewards.size()[0])):
            if rewards[i] == 0:
                running = tensor(0.)
                continue
            running = running * self.settings.discount_factor + rewards[i]
            rewards[i] = running

        return rewards
