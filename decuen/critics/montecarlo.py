"""Implementation of a rudimentary Monte Carlo "reward" critic.

Simply estimates the state values based on a Monte Carlo estimate of the expected reward.
"""

from dataclasses import dataclass
from typing import MutableSequence

from torch import arange

from decuen.critics._critic import Critic, CriticSettings
from decuen.structs import (Tensor, Trajectory, Transition, batch_transitions,
                            tensor)


@dataclass
class MonteCarloCriticSettings(CriticSettings):
    """Settings for Monte Carlo critics."""


class MonteCarloCritic(Critic):
    """Monte Carlo critic."""

    def __init__(self, settings: MonteCarloCriticSettings) -> None:
        """Initialize a Monte Carlo critic."""
        super().__init__(settings)

    def learn(self, transitions: MutableSequence[Transition]) -> None:
        """Do nothing. Monte Carlo critic does not learn."""

    def _advantage(self, trajectory: Trajectory) -> Tensor:
        batch = batch_transitions(trajectory)
        discounted_rewards = tensor([self.settings.discount_factor]).pow(arange(batch.rewards.size()[0]))
        advantages = discounted_rewards.flip(0).cumsum(0).flip(0)  # Reverse cumulative sum (causality)
        return advantages
