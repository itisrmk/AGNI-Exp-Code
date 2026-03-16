"""
Publication-ready visualization functions.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

# Color palette for strategies
COLORS = {
    "static": "#2ecc71",  # Green
    "periodic": "#3498db",  # Blue
    "continual": "#e74c3c",  # Red
}

# Color palette for models
MODEL_COLORS = {
    "lstm": "#9b59b6",  # Purple
    "tcn": "#f39c12",  # Orange
    "transformer": "#1abc9c",  # Teal
}


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Clarke Error Grid",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot Clarke Error Grid with predictions.

    Args:
        y_true: Reference glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot zones (simplified boundaries)
    # Zone A boundaries
    ax.plot([0, 70], [0, 70], "k-", linewidth=1)
    ax.plot([70, 400], [70 * 0.8, 400 * 0.8], "k-", linewidth=1)
    ax.plot([70, 400], [70 * 1.2, 400 * 1.2], "k-", linewidth=1)

    # Reference lines
    ax.axhline(y=70, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axhline(y=180, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axvline(x=70, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)
    ax.axvline(x=180, color="gray", linestyle="--", alpha=0.5, linewidth=0.5)

    # Scatter plot of predictions
    ax.scatter(y_true, y_pred, alpha=0.5, s=10, c="steelblue")

    # Perfect prediction line
    ax.plot([0, 400], [0, 400], "k--", linewidth=1, label="Perfect prediction")

    # Labels
    ax.set_xlabel("Reference Glucose (mg/dL)")
    ax.set_ylabel("Predicted Glucose (mg/dL)")
    ax.set_title(title)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)

    # Zone labels
    ax.text(30, 30, "A", fontsize=14, fontweight="bold", alpha=0.5)
    ax.text(200, 350, "B", fontsize=14, fontweight="bold", alpha=0.5)
    ax.text(350, 200, "B", fontsize=14, fontweight="bold", alpha=0.5)

    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_predictions_vs_actual(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Glucose Predictions",
    n_points: int = 288,  # 1 day
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted vs actual glucose over time.

    Args:
        timestamps: Time axis
        y_true: Actual glucose values
        y_pred: Predicted glucose values
        title: Plot title
        n_points: Number of points to show
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Limit to n_points
    idx = slice(0, min(n_points, len(y_true)))

    # Plot
    ax.plot(range(len(y_true[idx])), y_true[idx], "b-", label="Actual", alpha=0.8)
    ax.plot(range(len(y_pred[idx])), y_pred[idx], "r--", label="Predicted", alpha=0.8)

    # Target range
    ax.axhline(y=70, color="green", linestyle=":", alpha=0.5, label="Target range")
    ax.axhline(y=180, color="green", linestyle=":", alpha=0.5)
    ax.fill_between(range(len(y_true[idx])), 70, 180, alpha=0.1, color="green")

    ax.set_xlabel("Time (5-min intervals)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_drift_curves(
    drift_data: Dict[str, np.ndarray],
    time_axis: np.ndarray = None,
    metric: str = "RMSE",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot prediction error over time for different strategies.

    Args:
        drift_data: Dictionary mapping strategy name to error array
        time_axis: Time values for x-axis
        metric: Name of the metric for y-axis label
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, errors in drift_data.items():
        color = COLORS.get(strategy.lower(), "gray")
        x = time_axis if time_axis is not None else np.arange(len(errors))
        ax.plot(x, errors, label=strategy.capitalize(), color=color, linewidth=2)

    ax.set_xlabel("Days Since Deployment")
    ax.set_ylabel(f"{metric} (mg/dL)")
    ax.set_title(f"Temporal Drift: {metric} Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_strategy_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "rmse",
    horizons: List[int] = [15, 30, 60],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing strategies across prediction horizons.

    Args:
        results: Nested dict {strategy: {metric_horizon: value}}
        metric: Metric name to plot
        horizons: Prediction horizons in minutes
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(horizons))
    width = 0.25

    for i, (strategy, data) in enumerate(results.items()):
        values = [data.get(f"{metric}_{h}min", data.get(metric, 0)) for h in horizons]
        color = COLORS.get(strategy.lower(), "gray")
        ax.bar(x + i * width, values, width, label=strategy.capitalize(), color=color)

    ax.set_xlabel("Prediction Horizon (minutes)")
    ax.set_ylabel(f"{metric.upper()} (mg/dL)")
    ax.set_title(f"{metric.upper()} by Strategy and Horizon")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{h} min" for h in horizons])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["mae", "rmse"],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare different model architectures.

    Args:
        results: {model_name: {metric: value}}
        metrics: List of metrics to compare
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    models = list(results.keys())

    for ax, metric in zip(axes, metrics):
        values = [results[model].get(metric, 0) for model in models]
        colors = [MODEL_COLORS.get(model.lower(), "gray") for model in models]

        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel(f"{metric.upper()} (mg/dL)")
        ax.set_title(f"{metric.upper()} by Model")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    ax.plot(epochs, history["val_loss"], "r-", label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_ablation_heatmap(
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    save_path: Optional[str] = None,
    cmap: str = "RdYlGn_r",
) -> plt.Figure:
    """
    Heatmap for ablation study results.

    Args:
        data: 2D array of values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap name

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_glucose_distribution(
    glucose: np.ndarray,
    title: str = "Glucose Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot histogram of glucose values with clinical thresholds.

    Args:
        glucose: Array of glucose values (mg/dL)
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(glucose, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

    # Clinical thresholds
    ax.axvline(x=70, color="red", linestyle="--", label="Hypo threshold (70)")
    ax.axvline(x=180, color="orange", linestyle="--", label="Hyper threshold (180)")

    ax.set_xlabel("Glucose (mg/dL)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def plot_error_by_glucose_range(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot prediction error stratified by glucose range.

    Args:
        y_true: Actual glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    errors = y_pred - y_true

    ranges = [
        ("Hypoglycemia (<70)", y_true < 70, "red"),
        ("In Range (70-180)", (y_true >= 70) & (y_true <= 180), "green"),
        ("Hyperglycemia (>180)", y_true > 180, "orange"),
    ]

    for ax, (name, mask, color) in zip(axes, ranges):
        if np.sum(mask) > 0:
            ax.hist(errors[mask], bins=30, alpha=0.7, color=color, edgecolor="black")
            ax.axvline(x=0, color="black", linestyle="--")
            ax.set_xlabel("Prediction Error (mg/dL)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{name}\nMAE: {np.mean(np.abs(errors[mask])):.1f} mg/dL")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(name)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def save_all_figures(
    figures: Dict[str, plt.Figure], output_dir: str, format: str = "pdf"
):
    """
    Save multiple figures to directory.

    Args:
        figures: Dictionary mapping names to Figure objects
        output_dir: Directory to save figures
        format: File format ('pdf', 'png', 'svg')
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, fig in figures.items():
        path = os.path.join(output_dir, f"{name}.{format}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Saved: {path}")
