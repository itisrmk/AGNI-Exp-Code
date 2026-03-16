"""
Prediction accuracy metrics.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE in same units as input
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE in same units as input
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE as percentage (0-100)
    """
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Correlation coefficient (-1 to 1)
    """
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        R² value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MSE
    """
    return float(np.mean((y_true - y_pred) ** 2))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all standard prediction metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with all metrics
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_square_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
    }


def compute_glucose_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, normalizer=None
) -> Dict[str, float]:
    """
    Compute glucose-specific metrics.

    If normalizer is provided, converts normalized values back to mg/dL.

    Args:
        y_true: Ground truth glucose values
        y_pred: Predicted glucose values
        normalizer: Optional normalizer for inverse transform

    Returns:
        Dictionary with metrics in mg/dL units
    """
    # Convert back to mg/dL if normalized
    if normalizer is not None:
        y_true = normalizer.inverse_transform(y_true)
        y_pred = normalizer.inverse_transform(y_pred)

    metrics = compute_all_metrics(y_true, y_pred)

    # Add glucose-specific metrics
    metrics["mean_error"] = float(np.mean(y_pred - y_true))  # Bias
    metrics["std_error"] = float(np.std(y_pred - y_true))

    return metrics


def compute_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hypo_threshold: float = 70,
    hyper_threshold: float = 180,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by glucose range.

    Args:
        y_true: Ground truth glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        hypo_threshold: Threshold for hypoglycemia
        hyper_threshold: Threshold for hyperglycemia

    Returns:
        Dictionary with metrics for each range
    """
    results = {}

    # Hypoglycemia (<70)
    hypo_mask = y_true < hypo_threshold
    if np.sum(hypo_mask) > 0:
        results["hypo"] = compute_all_metrics(y_true[hypo_mask], y_pred[hypo_mask])
        results["hypo"]["n_samples"] = int(np.sum(hypo_mask))

    # Normal range (70-180)
    normal_mask = (y_true >= hypo_threshold) & (y_true <= hyper_threshold)
    if np.sum(normal_mask) > 0:
        results["normal"] = compute_all_metrics(
            y_true[normal_mask], y_pred[normal_mask]
        )
        results["normal"]["n_samples"] = int(np.sum(normal_mask))

    # Hyperglycemia (>180)
    hyper_mask = y_true > hyper_threshold
    if np.sum(hyper_mask) > 0:
        results["hyper"] = compute_all_metrics(y_true[hyper_mask], y_pred[hyper_mask])
        results["hyper"]["n_samples"] = int(np.sum(hyper_mask))

    return results


class MetricsTracker:
    """
    Track metrics over time for drift analysis.

    Useful for monitoring model performance degradation.
    """

    def __init__(self):
        self.history: List[Dict[str, float]] = []
        self.timestamps: List = []

    def add(self, metrics: Dict[str, float], timestamp=None):
        """Add metrics for a time point."""
        self.history.append(metrics)
        self.timestamps.append(timestamp)

    def get_metric_history(self, metric_name: str) -> np.ndarray:
        """Get history of a specific metric."""
        return np.array([m.get(metric_name, np.nan) for m in self.history])

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics."""
        if not self.history:
            return {}

        summary = {}
        metric_names = self.history[0].keys()

        for name in metric_names:
            values = self.get_metric_history(name)
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                summary[name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "final": float(valid_values[-1]),
                }

        return summary

    def compute_drift(self, window_size: int = 7) -> Dict[str, float]:
        """
        Compute metric drift comparing early vs late performance.

        Args:
            window_size: Number of samples to use for comparison

        Returns:
            Dictionary with drift values (positive = degradation)
        """
        if len(self.history) < 2 * window_size:
            return {}

        drift = {}
        metric_names = self.history[0].keys()

        for name in metric_names:
            values = self.get_metric_history(name)
            early = np.mean(values[:window_size])
            late = np.mean(values[-window_size:])
            drift[name] = float(late - early)

        return drift
