"""Critic interfaces and implementations for generating and learning critical analysis of actions and states."""

from decuen.critics._critic import (ActionCritic, ActionCriticSettings, Critic,
                                    CriticSettings, StateCritic,
                                    StateCriticSettings)
from decuen.critics.dqn import DDQNCritic, DQNCritic, DQNCriticSettings
from decuen.critics.qtable import QTableCritic, QTableCriticSettings

__all__ = [
    "Critic", "CriticSettings", "StateCritic", "StateCriticSettings", "ActionCritic", "ActionCriticSettings",
    "DQNCritic", "DDQNCritic", "DQNCriticSettings",
    "QTableCritic", "QTableCriticSettings",
]
