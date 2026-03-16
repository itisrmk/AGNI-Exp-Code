"""
AGNI Framework - Stratified Experience Replay Buffer
Stage 3: Maintains memory of critical past experiences

The replay buffer stores past training examples with stratification
by glucose range (hypoglycemia, normal, hyperglycemia) to ensure
balanced representation of rare but critical events.

Reference: Rolnick et al., "Experience Replay for Continual Learning" (2019)
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Basic experience replay buffer with reservoir sampling.

    Uses reservoir sampling to maintain a fixed-size buffer that
    represents the full distribution of seen examples.
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.position = 0
        self.n_seen = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Add a sample to the buffer using reservoir sampling.

        Args:
            x: Input tensor
            y: Target tensor
        """
        self.n_seen += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((x.clone(), y.clone()))
        else:
            # Reservoir sampling: replace with probability capacity/n_seen
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (x.clone(), y.clone())

    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Add a batch of samples to the buffer.

        Args:
            x: Batch of inputs (batch_size, ...)
            y: Batch of targets (batch_size, ...)
        """
        for i in range(x.size(0)):
            self.add(x[i], y[i])

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n examples from the buffer.

        Args:
            n: Number of samples to retrieve

        Returns:
            Tuple of (inputs, targets) tensors
        """
        n = min(n, len(self.buffer))
        samples = random.sample(self.buffer, n)

        x = torch.stack([s[0] for s in samples])
        y = torch.stack([s[1] for s in samples])

        return x, y

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
        self.position = 0
        self.n_seen = 0


class StratifiedReplayBuffer:
    """
    Stratified experience replay buffer for glucose prediction.

    Maintains separate buffers for different glucose ranges:
    - Hypoglycemia: <70 mg/dL (critical - must remember!)
    - Normal: 70-180 mg/dL
    - Hyperglycemia: >180 mg/dL (important for safety)

    This ensures rare but critical events (hypoglycemia) are not
    forgotten even when they occur infrequently.
    """

    # Glucose thresholds (mg/dL)
    HYPO_THRESHOLD = 70
    HYPER_THRESHOLD = 180

    def __init__(
        self,
        capacity: int = 1000,
        hypo_ratio: float = 0.3,
        normal_ratio: float = 0.4,
        hyper_ratio: float = 0.3,
    ):
        """
        Initialize the stratified buffer.

        Args:
            capacity: Total capacity across all strata
            hypo_ratio: Fraction of capacity for hypoglycemia samples
            normal_ratio: Fraction of capacity for normal samples
            hyper_ratio: Fraction of capacity for hyperglycemia samples
        """
        assert abs(hypo_ratio + normal_ratio + hyper_ratio - 1.0) < 1e-6, (
            "Ratios must sum to 1"
        )

        self.capacity = capacity
        self.hypo_ratio = hypo_ratio
        self.normal_ratio = normal_ratio
        self.hyper_ratio = hyper_ratio

        # Create stratified buffers
        self.hypo_buffer = ReplayBuffer(int(capacity * hypo_ratio))
        self.normal_buffer = ReplayBuffer(int(capacity * normal_ratio))
        self.hyper_buffer = ReplayBuffer(int(capacity * hyper_ratio))

        # Statistics
        self.stats = {"hypo_added": 0, "normal_added": 0, "hyper_added": 0}

    def _get_stratum(self, glucose_value: float) -> str:
        """Determine which stratum a glucose value belongs to."""
        if glucose_value < self.HYPO_THRESHOLD:
            return "hypo"
        elif glucose_value > self.HYPER_THRESHOLD:
            return "hyper"
        else:
            return "normal"

    def add(self, x: torch.Tensor, y: torch.Tensor, glucose_value: float) -> None:
        """
        Add a sample to the appropriate stratum.

        Args:
            x: Input tensor (sequence of glucose values)
            y: Target tensor (prediction target)
            glucose_value: Raw glucose value for stratification
        """
        stratum = self._get_stratum(glucose_value)

        if stratum == "hypo":
            self.hypo_buffer.add(x, y)
            self.stats["hypo_added"] += 1
        elif stratum == "hyper":
            self.hyper_buffer.add(x, y)
            self.stats["hyper_added"] += 1
        else:
            self.normal_buffer.add(x, y)
            self.stats["normal_added"] += 1

    def add_batch(
        self, x: torch.Tensor, y: torch.Tensor, glucose_values: np.ndarray
    ) -> None:
        """
        Add a batch of samples to the buffer.

        Args:
            x: Batch of inputs
            y: Batch of targets
            glucose_values: Raw glucose values for stratification
        """
        for i in range(x.size(0)):
            self.add(x[i], y[i], glucose_values[i])

    def sample(
        self, n: int, balanced: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n examples from the buffer.

        Args:
            n: Number of samples to retrieve
            balanced: If True, sample proportionally from each stratum
                     If False, sample based on buffer sizes

        Returns:
            Tuple of (inputs, targets) tensors
        """
        if balanced:
            # Sample proportionally from each stratum
            n_hypo = int(n * self.hypo_ratio)
            n_hyper = int(n * self.hyper_ratio)
            n_normal = n - n_hypo - n_hyper
        else:
            # Sample based on actual buffer sizes
            total = len(self)
            if total == 0:
                return torch.tensor([]), torch.tensor([])

            n_hypo = int(n * len(self.hypo_buffer) / total)
            n_hyper = int(n * len(self.hyper_buffer) / total)
            n_normal = n - n_hypo - n_hyper

        samples_x = []
        samples_y = []

        # Sample from each stratum
        if len(self.hypo_buffer) > 0 and n_hypo > 0:
            x, y = self.hypo_buffer.sample(min(n_hypo, len(self.hypo_buffer)))
            samples_x.append(x)
            samples_y.append(y)

        if len(self.normal_buffer) > 0 and n_normal > 0:
            x, y = self.normal_buffer.sample(min(n_normal, len(self.normal_buffer)))
            samples_x.append(x)
            samples_y.append(y)

        if len(self.hyper_buffer) > 0 and n_hyper > 0:
            x, y = self.hyper_buffer.sample(min(n_hyper, len(self.hyper_buffer)))
            samples_x.append(x)
            samples_y.append(y)

        if not samples_x:
            return torch.tensor([]), torch.tensor([])

        # Concatenate and shuffle
        x = torch.cat(samples_x, dim=0)
        y = torch.cat(samples_y, dim=0)

        # Shuffle
        perm = torch.randperm(x.size(0))
        return x[perm], y[perm]

    def __len__(self) -> int:
        return len(self.hypo_buffer) + len(self.normal_buffer) + len(self.hyper_buffer)

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "total_size": len(self),
            "hypo_size": len(self.hypo_buffer),
            "normal_size": len(self.normal_buffer),
            "hyper_size": len(self.hyper_buffer),
            "hypo_added": self.stats["hypo_added"],
            "normal_added": self.stats["normal_added"],
            "hyper_added": self.stats["hyper_added"],
            "capacity": self.capacity,
        }

    def clear(self) -> None:
        """Clear all buffers."""
        self.hypo_buffer.clear()
        self.normal_buffer.clear()
        self.hyper_buffer.clear()
        self.stats = {"hypo_added": 0, "normal_added": 0, "hyper_added": 0}


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples are prioritized based on prediction error - examples
    where the model performed poorly are more likely to be sampled.
    """

    def __init__(
        self,
        capacity: int = 1000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """
        Initialize prioritized buffer.

        Args:
            capacity: Maximum capacity
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: How much to increase beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(
        self, x: torch.Tensor, y: torch.Tensor, priority: Optional[float] = None
    ) -> None:
        """
        Add a sample with priority.

        Args:
            x: Input tensor
            y: Target tensor
            priority: Sample priority (default: max_priority)
        """
        if priority is None:
            priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append((x.clone(), y.clone(), priority))
        else:
            self.buffer[self.position] = (x.clone(), y.clone(), priority)

        self.priorities[self.position] = priority**self.alpha
        self.position = (self.position + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)

    def sample(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[int]]:
        """
        Sample n examples with prioritization.

        Returns:
            Tuple of (inputs, targets, importance_weights, indices)
        """
        if len(self.buffer) == 0:
            return torch.tensor([]), torch.tensor([]), np.array([]), []

        n = min(n, len(self.buffer))

        # Compute sampling probabilities
        priorities = self.priorities[: len(self.buffer)]
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), size=n, p=probs, replace=False)

        # Compute importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Get samples
        samples = [self.buffer[i] for i in indices]
        x = torch.stack([s[0] for s in samples])
        y = torch.stack([s[1] for s in samples])

        return x, y, weights, indices.tolist()

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for sampled indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)
