"""
Evaluation metrics and visualization.
"""

from .clinical import (
    clarke_error_grid,
    compute_all_clinical_metrics,
    hyperglycemia_detection,
    hypoglycemia_detection,
    parkes_error_grid,
    time_in_range,
)
from .metrics import (
    MetricsTracker,
    compute_all_metrics,
    compute_glucose_metrics,
    compute_stratified_metrics,
    correlation,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r_squared,
    root_mean_square_error,
)
from .visualization import (
    COLORS,
    MODEL_COLORS,
    plot_ablation_heatmap,
    plot_clarke_error_grid,
    plot_drift_curves,
    plot_error_by_glucose_range,
    plot_glucose_distribution,
    plot_model_comparison,
    plot_predictions_vs_actual,
    plot_strategy_comparison,
    plot_training_history,
    save_all_figures,
)

__all__ = [
    # Metrics
    "mean_absolute_error",
    "root_mean_square_error",
    "mean_absolute_percentage_error",
    "correlation",
    "r_squared",
    "mean_squared_error",
    "compute_all_metrics",
    "compute_glucose_metrics",
    "compute_stratified_metrics",
    "MetricsTracker",
    # Clinical
    "clarke_error_grid",
    "parkes_error_grid",
    "hypoglycemia_detection",
    "hyperglycemia_detection",
    "time_in_range",
    "compute_all_clinical_metrics",
    # Visualization
    "plot_clarke_error_grid",
    "plot_predictions_vs_actual",
    "plot_drift_curves",
    "plot_strategy_comparison",
    "plot_model_comparison",
    "plot_training_history",
    "plot_ablation_heatmap",
    "plot_glucose_distribution",
    "plot_error_by_glucose_range",
    "save_all_figures",
    "COLORS",
    "MODEL_COLORS",
]
