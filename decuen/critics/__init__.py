"""Critic interfaces and implementations for generating and learning critical analysis of actions and states."""

from typing import Union

from decuen.critics._critic import (ActionValueCritic, AdvantageCritic,
                                    CriticSettings, StateValueCritic)
from decuen.critics.dqn import DQNCritic, DQNCriticSettings
from decuen.critics.montecarlo import (MonteCarloCritic,
                                       MonteCarloCriticSettings)

Critic = Union[ActionValueCritic, StateValueCritic, AdvantageCritic]

__all__ = [
    "Critic", "CriticSettings",
    "StateValueCritic",
    "ActionValueCritic",
    "AdvantageCritic",
    "DQNCritic", "DQNCriticSettings",
    "MonteCarloCritic", "MonteCarloCriticSettings",
]
