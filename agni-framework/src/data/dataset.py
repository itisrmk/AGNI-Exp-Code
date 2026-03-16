"""
PyTorch Dataset classes for glucose prediction.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class GlucoseDataset(Dataset):
    """
    Dataset for glucose prediction with sliding windows.

    Creates input-output pairs where:
    - Input: sequence of `window_size` historical glucose values
    - Output: glucose value `horizon` steps into the future

    Args:
        glucose: Array of glucose values (normalized)
        timestamps: Array of timestamps
        window_size: Number of historical samples (default: 24 = 120 min)
        horizon: Prediction horizon in samples (3=15min, 6=30min, 12=60min)
        glucose_raw: Optional raw glucose values for stratification
    """

    def __init__(
        self,
        glucose: np.ndarray,
        timestamps: np.ndarray = None,
        window_size: int = 24,
        horizon: int = 6,
        glucose_raw: Optional[np.ndarray] = None,
    ):
        self.glucose = torch.FloatTensor(glucose)
        self.timestamps = timestamps
        self.window_size = window_size
        self.horizon = horizon
        self.glucose_raw = glucose_raw if glucose_raw is not None else glucose

        # Create valid indices (where we have full window + horizon without NaN)
        self.valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self) -> List[int]:
        """
        Find indices where we have complete window and target.

        An index i is valid if:
        - We have window_size samples before it: [i - window_size, i)
        - We have a target horizon samples ahead: at position i + horizon - 1
        - No NaN values in window or target
        """
        valid = []

        for i in range(self.window_size, len(self.glucose) - self.horizon + 1):
            # Get window and target
            window = self.glucose[i - self.window_size : i]
            target = self.glucose[i + self.horizon - 1]

            # Check for NaN
            if not (torch.isnan(window).any() or torch.isnan(target)):
                valid.append(i)

        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index into valid_indices

        Returns:
            Tuple of (input_window, target):
                - input_window: (window_size, 1) tensor
                - target: scalar tensor
        """
        i = self.valid_indices[idx]
        window = self.glucose[i - self.window_size : i]
        target = self.glucose[i + self.horizon - 1]

        # Add feature dimension: (window_size,) -> (window_size, 1)
        return window.unsqueeze(-1), target

    def get_glucose_category(self, idx: int) -> str:
        """
        Get glucose category for stratified sampling.

        Categories:
        - 'hypo': < 70 mg/dL (hypoglycemia)
        - 'hyper': > 180 mg/dL (hyperglycemia)
        - 'normal': 70-180 mg/dL (in range)

        Args:
            idx: Index into valid_indices

        Returns:
            Category string
        """
        i = self.valid_indices[idx]
        raw_value = self.glucose_raw[i + self.horizon - 1]

        if raw_value < 70:
            return "hypo"
        elif raw_value > 180:
            return "hyper"
        else:
            return "normal"

    def get_raw_target(self, idx: int) -> float:
        """Get raw (unnormalized) target glucose value."""
        i = self.valid_indices[idx]
        return float(self.glucose_raw[i + self.horizon - 1])

    def get_timestamp(self, idx: int) -> Optional[np.datetime64]:
        """Get timestamp for the target of this sample."""
        if self.timestamps is None:
            return None
        i = self.valid_indices[idx]
        return self.timestamps[i + self.horizon - 1]

    def get_category_distribution(self) -> Dict[str, int]:
        """Get count of samples in each glucose category."""
        dist = {"hypo": 0, "normal": 0, "hyper": 0}
        for idx in range(len(self)):
            cat = self.get_glucose_category(idx)
            dist[cat] += 1
        return dist


class MultiHorizonGlucoseDataset(Dataset):
    """
    Dataset that returns targets for multiple prediction horizons.

    Useful for training models that predict multiple horizons simultaneously.

    Args:
        glucose: Array of glucose values (normalized)
        timestamps: Array of timestamps
        window_size: Number of historical samples
        horizons: List of prediction horizons in samples
        glucose_raw: Optional raw glucose values
    """

    def __init__(
        self,
        glucose: np.ndarray,
        timestamps: np.ndarray = None,
        window_size: int = 24,
        horizons: List[int] = [3, 6, 12],
        glucose_raw: Optional[np.ndarray] = None,
    ):
        self.glucose = torch.FloatTensor(glucose)
        self.timestamps = timestamps
        self.window_size = window_size
        self.horizons = horizons
        self.max_horizon = max(horizons)
        self.glucose_raw = glucose_raw if glucose_raw is not None else glucose

        self.valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self) -> List[int]:
        """Find valid indices for all horizons."""
        valid = []

        for i in range(self.window_size, len(self.glucose) - self.max_horizon + 1):
            window = self.glucose[i - self.window_size : i]

            # Check window is valid
            if torch.isnan(window).any():
                continue

            # Check all horizon targets are valid
            all_valid = True
            for h in self.horizons:
                target = self.glucose[i + h - 1]
                if torch.isnan(target):
                    all_valid = False
                    break

            if all_valid:
                valid.append(i)

        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with multiple horizon targets.

        Returns:
            Tuple of (input_window, targets):
                - input_window: (window_size, 1) tensor
                - targets: (num_horizons,) tensor
        """
        i = self.valid_indices[idx]
        window = self.glucose[i - self.window_size : i]

        targets = torch.tensor([self.glucose[i + h - 1] for h in self.horizons])

        return window.unsqueeze(-1), targets


class TemporalSplitter:
    """
    Chronological train/val/test splitting.

    Ensures no data leakage by respecting temporal order.
    The first portion is for training, then validation, then test.

    Args:
        train_ratio: Fraction of data for training (default: 0.5)
        val_ratio: Fraction of training data for validation (default: 0.2)
    """

    def __init__(self, train_ratio: float = 0.5, val_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, dataset_length: int) -> Tuple[range, range, range]:
        """
        Return train, val, test index ranges.

        Args:
            dataset_length: Total number of samples in dataset

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        train_end = int(dataset_length * self.train_ratio)
        val_size = int(train_end * self.val_ratio)

        train_indices = range(0, train_end - val_size)
        val_indices = range(train_end - val_size, train_end)
        test_indices = range(train_end, dataset_length)

        return train_indices, val_indices, test_indices

    def split_dataset(self, dataset: Dataset) -> Tuple[Subset, Subset, Subset]:
        """
        Split dataset into train/val/test Subsets.

        Args:
            dataset: PyTorch Dataset to split

        Returns:
            Tuple of (train_subset, val_subset, test_subset)
        """
        train_idx, val_idx, test_idx = self.split(len(dataset))

        return (
            Subset(dataset, list(train_idx)),
            Subset(dataset, list(val_idx)),
            Subset(dataset, list(test_idx)),
        )


class DayBasedSplitter:
    """
    Split data by days for temporal evaluation.

    Useful for simulating real-world deployment where models
    are evaluated day-by-day.

    Args:
        samples_per_day: Number of samples per day (default: 288 for 5-min intervals)
    """

    def __init__(self, samples_per_day: int = 288):
        self.samples_per_day = samples_per_day

    def get_day_ranges(self, dataset_length: int) -> List[Tuple[int, int]]:
        """
        Get index ranges for each day.

        Returns:
            List of (start_idx, end_idx) tuples for each day
        """
        days = []
        start = 0

        while start < dataset_length:
            end = min(start + self.samples_per_day, dataset_length)
            days.append((start, end))
            start = end

        return days

    def split_by_days(
        self, dataset: Dataset, train_days: int, val_days: int = 0
    ) -> Tuple[Subset, Optional[Subset], Subset]:
        """
        Split dataset by number of days.

        Args:
            dataset: Dataset to split
            train_days: Number of days for training
            val_days: Number of days for validation (taken from end of training)

        Returns:
            Tuple of (train_subset, val_subset, test_subset)
        """
        day_ranges = self.get_day_ranges(len(dataset))

        train_end_day = min(train_days, len(day_ranges))
        val_start_day = train_end_day - val_days

        train_indices = []
        for day in range(val_start_day):
            start, end = day_ranges[day]
            train_indices.extend(range(start, end))

        val_indices = []
        if val_days > 0:
            for day in range(val_start_day, train_end_day):
                start, end = day_ranges[day]
                val_indices.extend(range(start, end))

        test_indices = []
        for day in range(train_end_day, len(day_ranges)):
            start, end = day_ranges[day]
            test_indices.extend(range(start, end))

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices) if val_indices else None
        test_subset = Subset(dataset, test_indices)

        return train_subset, val_subset, test_subset
