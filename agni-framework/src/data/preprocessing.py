"""
Data preprocessing for CGM time series.
Handles gap detection, outlier removal, and normalization.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing CGM data."""

    min_glucose: float = 20.0
    max_glucose: float = 600.0
    max_gap_minutes: int = 30
    sampling_interval: int = 5  # minutes


def detect_gaps(timestamps: pd.Series, max_gap_minutes: int = 30) -> np.ndarray:
    """
    Detect gaps in CGM data exceeding threshold.

    Args:
        timestamps: Series of datetime timestamps
        max_gap_minutes: Maximum allowed gap in minutes

    Returns:
        Boolean array where True indicates a gap exceeds threshold
    """
    time_diffs = timestamps.diff().dt.total_seconds() / 60
    return (time_diffs > max_gap_minutes).values


def get_gap_indices(timestamps: pd.Series, max_gap_minutes: int = 30) -> List[int]:
    """
    Get indices where large gaps occur.

    Args:
        timestamps: Series of datetime timestamps
        max_gap_minutes: Maximum allowed gap in minutes

    Returns:
        List of indices where gaps occur
    """
    gaps = detect_gaps(timestamps, max_gap_minutes)
    return list(np.where(gaps)[0])


def remove_outliers(
    glucose: np.ndarray, min_val: float = 20, max_val: float = 600
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flag and remove physiologically implausible glucose values.

    Args:
        glucose: Array of glucose values
        min_val: Minimum valid glucose value (mg/dL)
        max_val: Maximum valid glucose value (mg/dL)

    Returns:
        Tuple of (cleaned glucose array with NaN for outliers, boolean mask of valid values)
    """
    mask = (glucose >= min_val) & (glucose <= max_val)
    cleaned = glucose.copy()
    cleaned[~mask] = np.nan
    return cleaned, mask


def interpolate_gaps(
    glucose: np.ndarray,
    timestamps: pd.Series = None,
    max_gap_samples: int = 6,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate small gaps in glucose data.

    Args:
        glucose: Array of glucose values (may contain NaN)
        timestamps: Optional timestamps for time-aware interpolation
        max_gap_samples: Maximum consecutive NaN samples to interpolate
        method: Interpolation method ('linear', 'cubic', 'spline')

    Returns:
        Interpolated glucose array
    """
    if timestamps is not None:
        df = pd.DataFrame({"glucose": glucose}, index=timestamps)
    else:
        df = pd.DataFrame({"glucose": glucose})

    return df["glucose"].interpolate(method=method, limit=max_gap_samples).values


def segment_by_gaps(
    glucose: np.ndarray, timestamps: pd.Series, max_gap_minutes: int = 30
) -> List[Tuple[int, int]]:
    """
    Segment data into continuous chunks separated by large gaps.

    Args:
        glucose: Array of glucose values
        timestamps: Series of timestamps
        max_gap_minutes: Maximum gap to consider continuous

    Returns:
        List of (start_idx, end_idx) tuples for each segment
    """
    gap_indices = get_gap_indices(timestamps, max_gap_minutes)

    segments = []
    start_idx = 0

    for gap_idx in gap_indices:
        if gap_idx > start_idx:
            segments.append((start_idx, gap_idx))
        start_idx = gap_idx

    # Add final segment
    if start_idx < len(glucose):
        segments.append((start_idx, len(glucose)))

    return segments


class GlucoseNormalizer:
    """
    Per-patient glucose normalization using z-score standardization.

    Attributes:
        mean: Fitted mean value
        std: Fitted standard deviation
    """

    def __init__(self):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self._fitted = False

    def fit(self, glucose: np.ndarray) -> "GlucoseNormalizer":
        """
        Fit normalizer on glucose values.

        Args:
            glucose: Array of glucose values (may contain NaN)

        Returns:
            Self for chaining
        """
        self.mean = np.nanmean(glucose)
        self.std = np.nanstd(glucose)

        # Prevent division by zero
        if self.std < 1e-6:
            self.std = 1.0

        self._fitted = True
        return self

    def transform(self, glucose: np.ndarray) -> np.ndarray:
        """
        Transform glucose values to normalized scale.

        Args:
            glucose: Array of glucose values

        Returns:
            Normalized glucose values
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (glucose - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """
        Transform normalized values back to original glucose scale.

        Args:
            normalized: Array of normalized values

        Returns:
            Glucose values in mg/dL
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        return normalized * self.std + self.mean

    def fit_transform(self, glucose: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            glucose: Array of glucose values

        Returns:
            Normalized glucose values
        """
        return self.fit(glucose).transform(glucose)

    def get_params(self) -> dict:
        """Get normalizer parameters."""
        return {"mean": self.mean, "std": self.std}

    def set_params(self, mean: float, std: float) -> "GlucoseNormalizer":
        """Set normalizer parameters manually."""
        self.mean = mean
        self.std = std
        self._fitted = True
        return self


class MinMaxNormalizer:
    """
    Min-max normalization to [0, 1] range.

    Uses fixed clinical range by default (40-400 mg/dL).
    """

    def __init__(self, min_val: float = 40.0, max_val: float = 400.0):
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def transform(self, glucose: np.ndarray) -> np.ndarray:
        """Transform to [0, 1] range."""
        return (glucose - self.min_val) / self.range

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Transform back to glucose scale."""
        return normalized * self.range + self.min_val

    def fit_transform(self, glucose: np.ndarray) -> np.ndarray:
        """No fitting needed, just transform."""
        return self.transform(glucose)


def preprocess_glucose_series(
    glucose: np.ndarray,
    timestamps: pd.Series = None,
    config: PreprocessingConfig = None,
) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for glucose time series.

    Args:
        glucose: Raw glucose values
        timestamps: Optional timestamps
        config: Preprocessing configuration

    Returns:
        Tuple of (preprocessed glucose, preprocessing info dict)
    """
    if config is None:
        config = PreprocessingConfig()

    info = {
        "original_length": len(glucose),
        "outliers_removed": 0,
        "gaps_interpolated": 0,
    }

    # Step 1: Remove outliers
    cleaned, valid_mask = remove_outliers(
        glucose, config.min_glucose, config.max_glucose
    )
    info["outliers_removed"] = int((~valid_mask).sum())

    # Step 2: Interpolate small gaps
    max_gap_samples = config.max_gap_minutes // config.sampling_interval
    nan_before = np.isnan(cleaned).sum()

    interpolated = interpolate_gaps(
        cleaned, timestamps, max_gap_samples=max_gap_samples
    )

    nan_after = np.isnan(interpolated).sum()
    info["gaps_interpolated"] = int(nan_before - nan_after)
    info["remaining_nan"] = int(nan_after)

    return interpolated, info
