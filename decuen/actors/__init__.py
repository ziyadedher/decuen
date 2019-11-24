"""Actor-learner interfaces and implementations for generating and learning behavioral policies."""

from decuen.actors._actor import Actor, ActorSettings
from decuen.actors.pg import PGActor, PGActorSettings
from decuen.actors.strategy import StrategyActor

__all__ = [
    "Actor", "ActorSettings",
    "StrategyActor",
    "PGActor", "PGActorSettings"
]
