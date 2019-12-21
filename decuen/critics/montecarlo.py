"""Implementation of a rudimentary Monte Carlo "reward" critic.

Simply estimates the state values based on a Monte Carlo estimate of the expected reward.
"""

from dataclasses import dataclass
from typing import List

from decuen.critics._critic import Critic, CriticSettings
from decuen.structs import Experience, Trajectory, gather_rewards


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

    def advantage(self, experience: Experience) -> List[float]:
        """Estimate the advantage of every step in an experience by using a Monte Carlo sampling."""
        if isinstance(experience, Trajectory):
            return self._calculate_trajectory_rewards(experience)
        return gather_rewards(experience).tolist()

    def _calculate_trajectory_rewards(self, trajectory: Trajectory):
        rewards = gather_rewards(trajectory).tolist()

        running = 0
        for i, reward in reversed(list(enumerate(rewards))):
            if reward == 0:
                running = 0
                continue
            running = running * self.settings.discount_factor + reward
            rewards[i] = running

        return rewards
