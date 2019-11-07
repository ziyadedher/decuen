"""Strategy interfaces and implementations for selecting actions based on only critic evaluation."""

from decuen.strategies._strategy import Strategy
from decuen.strategies.epsilon import EpsilonGreedyStrategy
from decuen.strategies.greedy import GreedyStrategy
from decuen.strategies.rand import RandomStrategy

__all__ = ["Strategy", "EpsilonGreedyStrategy", "GreedyStrategy", "RandomStrategy"]
