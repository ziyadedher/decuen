"""Critic interfaces and implementations for generating and learning critical analysis of actions and states."""

from decuen.critics._critic import (ActionCritic, ActionCriticSettings, Critic,
                                    CriticSettings, StateCritic,
                                    StateCriticSettings)

__all__ = ["Critic", "CriticSettings", "StateCritic", "StateCriticSettings", "ActionCritic", "ActionCriticSettings"]
