"""
Clinical safety metrics: Clarke Error Grid and hypoglycemia detection.
"""

from typing import Dict, Tuple

import numpy as np


def clarke_error_grid(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Clarke Error Grid zone distribution.

    The Clarke Error Grid is the clinical standard for evaluating
    glucose prediction accuracy. Zones:
    - Zone A: Clinically accurate (within 20% or both <70 mg/dL)
    - Zone B: Benign errors (would not lead to wrong treatment)
    - Zone C: Overcorrection (would cause unnecessary treatment)
    - Zone D: Dangerous failure to detect (missed hypo/hyperglycemia)
    - Zone E: Erroneous treatment (opposite treatment direction)

    Args:
        y_true: Reference glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        Dictionary with percentage in each zone
    """
    n = len(y_true)
    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for ref, pred in zip(y_true, y_pred):
        zone = _get_clarke_zone(ref, pred)
        zones[zone] += 1

    # Convert to percentages
    percentages = {k: (v / n) * 100 for k, v in zones.items()}

    # Add clinically acceptable percentage (A + B)
    percentages["A+B"] = percentages["A"] + percentages["B"]

    return percentages


def _get_clarke_zone(ref: float, pred: float) -> str:
    """
    Determine Clarke Error Grid zone for a single point.

    Args:
        ref: Reference glucose value (mg/dL)
        pred: Predicted glucose value (mg/dL)

    Returns:
        Zone letter ('A', 'B', 'C', 'D', or 'E')
    """
    # Zone A: Both values <= 70 or within 20% of reference
    if ref <= 70 and pred <= 70:
        return "A"
    if ref >= 70 and abs(pred - ref) <= 0.2 * ref:
        return "A"

    # Zone E: Dangerous - opposite hypoglycemia detection
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"

    # Zone D: Dangerous - failure to detect
    if ref >= 240 and pred <= 180:
        return "D"
    if ref <= 70 and pred >= 180:
        return "D"

    # Zone C: Overcorrection
    if ref >= 70 and ref <= 180:
        if pred >= 180 or pred <= 70:
            return "C"

    # Zone B: Everything else (benign errors)
    return "B"


def parkes_error_grid(
    y_true: np.ndarray, y_pred: np.ndarray, diabetes_type: int = 1
) -> Dict[str, float]:
    """
    Compute Parkes (Consensus) Error Grid zone distribution.

    The Parkes Error Grid is a more recent standard than Clarke,
    with zones A-E based on clinical risk assessment.

    Args:
        y_true: Reference glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        diabetes_type: Type 1 or Type 2 diabetes

    Returns:
        Dictionary with percentage in each zone
    """
    n = len(y_true)
    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for ref, pred in zip(y_true, y_pred):
        zone = _get_parkes_zone(ref, pred, diabetes_type)
        zones[zone] += 1

    percentages = {k: (v / n) * 100 for k, v in zones.items()}
    percentages["A+B"] = percentages["A"] + percentages["B"]

    return percentages


def _get_parkes_zone(ref: float, pred: float, diabetes_type: int = 1) -> str:
    """
    Determine Parkes Error Grid zone for a single point.

    Simplified implementation - actual boundaries are complex polygons.
    """
    abs_diff = abs(pred - ref)

    if ref <= 50:
        pct_diff = abs_diff
    else:
        pct_diff = abs_diff / ref * 100

    # Simplified zone boundaries
    if pct_diff <= 20 or (ref <= 70 and pred <= 70):
        return "A"
    elif pct_diff <= 40:
        return "B"
    elif pct_diff <= 60:
        return "C"
    elif pct_diff <= 80:
        return "D"
    else:
        return "E"


def hypoglycemia_detection(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 70.0
) -> Dict[str, float]:
    """
    Compute hypoglycemia detection metrics.

    Evaluates ability to detect hypoglycemic events (<70 mg/dL).

    Args:
        y_true: Reference glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        threshold: Hypoglycemia threshold (default: 70 mg/dL)

    Returns:
        Dictionary with detection metrics
    """
    actual_hypo = y_true < threshold
    predicted_hypo = y_pred < threshold

    # True positives: correctly predicted hypoglycemia
    tp = np.sum(actual_hypo & predicted_hypo)
    # False negatives: missed hypoglycemia (dangerous!)
    fn = np.sum(actual_hypo & ~predicted_hypo)
    # False positives: false alarms
    fp = np.sum(~actual_hypo & predicted_hypo)
    # True negatives: correctly predicted no hypoglycemia
    tn = np.sum(~actual_hypo & ~predicted_hypo)

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # F1 score
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0
    )

    return {
        "sensitivity": sensitivity * 100,  # % of hypos detected
        "specificity": specificity * 100,  # % of non-hypos correctly identified
        "precision": precision * 100,  # % of hypo predictions that are correct
        "false_alarm_rate": (1 - specificity) * 100,
        "miss_rate": (1 - sensitivity) * 100,  # Dangerous misses
        "f1_score": f1 * 100,
        "n_actual_hypo": int(np.sum(actual_hypo)),
        "n_predicted_hypo": int(np.sum(predicted_hypo)),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
    }


def hyperglycemia_detection(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 180.0
) -> Dict[str, float]:
    """
    Compute hyperglycemia detection metrics.

    Args:
        y_true: Reference glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        threshold: Hyperglycemia threshold (default: 180 mg/dL)

    Returns:
        Dictionary with detection metrics
    """
    actual_hyper = y_true > threshold
    predicted_hyper = y_pred > threshold

    tp = np.sum(actual_hyper & predicted_hyper)
    fn = np.sum(actual_hyper & ~predicted_hyper)
    fp = np.sum(~actual_hyper & predicted_hyper)
    tn = np.sum(~actual_hyper & ~predicted_hyper)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0
    )

    return {
        "sensitivity": sensitivity * 100,
        "specificity": specificity * 100,
        "precision": precision * 100,
        "false_alarm_rate": (1 - specificity) * 100,
        "miss_rate": (1 - sensitivity) * 100,
        "f1_score": f1 * 100,
        "n_actual_hyper": int(np.sum(actual_hyper)),
        "n_predicted_hyper": int(np.sum(predicted_hyper)),
    }


def time_in_range(glucose: np.ndarray) -> Dict[str, float]:
    """
    Calculate time in range metrics.

    Args:
        glucose: Glucose values (mg/dL)

    Returns:
        Dictionary with TIR percentages
    """
    n = len(glucose)

    return {
        "time_below_54": np.sum(glucose < 54) / n * 100,  # Severe hypo
        "time_below_70": np.sum(glucose < 70) / n * 100,  # Hypo
        "time_in_range": np.sum((glucose >= 70) & (glucose <= 180)) / n * 100,
        "time_above_180": np.sum(glucose > 180) / n * 100,  # Hyper
        "time_above_250": np.sum(glucose > 250) / n * 100,  # Severe hyper
    }


def compute_all_clinical_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, any]:
    """
    Compute all clinical metrics.

    Args:
        y_true: Reference glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        Dictionary with all clinical metrics
    """
    return {
        "clarke_grid": clarke_error_grid(y_true, y_pred),
        "hypo_detection": hypoglycemia_detection(y_true, y_pred),
        "hyper_detection": hyperglycemia_detection(y_true, y_pred),
        "time_in_range_actual": time_in_range(y_true),
        "time_in_range_predicted": time_in_range(y_pred),
    }
