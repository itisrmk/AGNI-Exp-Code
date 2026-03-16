# AGNI Framework - Adaptation Strategies
# Stage 2: Periodic Retraining
# Stage 3: Continual Learning (EWC + Experience Replay)

from .continual import ContinualAdapter, ContinualLearningExperiment
from .ewc import EWC, OnlineEWC
from .periodic import PeriodicAdapter, PeriodicRetrainingExperiment
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, StratifiedReplayBuffer

__all__ = [
    # Stage 2
    "PeriodicAdapter",
    "PeriodicRetrainingExperiment",
    # Stage 3
    "EWC",
    "OnlineEWC",
    "ReplayBuffer",
    "StratifiedReplayBuffer",
    "PrioritizedReplayBuffer",
    "ContinualAdapter",
    "ContinualLearningExperiment",
]
