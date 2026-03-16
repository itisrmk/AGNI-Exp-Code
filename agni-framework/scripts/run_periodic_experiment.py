#!/usr/bin/env python3
"""
AGNI Framework - Stage 2: Periodic Retraining Experiment Runner

This script runs periodic retraining experiments on the OhioT1DM dataset,
comparing the performance against the static baseline from Stage 1.

Usage:
    python scripts/run_periodic_experiment.py --model lstm --patient 559
    python scripts/run_periodic_experiment.py --model all --patient all
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from adaptation.periodic import PeriodicRetrainingExperiment
from data.dataset import GlucoseDataset
from data.ohio_loader import OhioT1DMLoader
from data.preprocessing import GlucoseNormalizer, preprocess_glucose_series
from models.lstm import LSTMPredictor
from models.tcn import TCNPredictor
from models.transformer import TransformerPredictor
from utils.device import get_device
from utils.reproducibility import set_seed


def get_model_class(model_name: str):
    """Get model class by name."""
    models = {
        "lstm": LSTMPredictor,
        "tcn": TCNPredictor,
        "transformer": TransformerPredictor,
    }
    return models.get(model_name.lower())


def get_model_config(model_name: str) -> dict:
    """Get model configuration."""
    configs = {
        "lstm": {
            "input_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "output_size": 1,
        },
        "tcn": {
            "input_size": 1,
            "num_channels": [32, 32, 32],
            "kernel_size": 3,
            "dropout": 0.2,
            "output_size": 1,
        },
        "transformer": {
            "input_size": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "output_size": 1,
        },
    }
    return configs.get(model_name.lower(), {})


def get_training_config() -> dict:
    """Get training configuration for periodic retraining."""
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,  # More epochs for better convergence
        "patience": 10,
        "weight_decay": 1e-5,
    }


def run_periodic_experiment(
    model_name: str,
    patient_id: str,
    data_path: Path,
    results_dir: Path,
    horizon_minutes: int = 30,
    retraining_interval_days: int = 7,
    device: str = "cpu",
) -> dict:
    """
    Run a single periodic retraining experiment.

    Args:
        model_name: Name of the model (lstm, tcn, transformer)
        patient_id: Patient ID
        data_path: Path to OhioT1DM data
        results_dir: Directory for results
        horizon_minutes: Prediction horizon in minutes
        retraining_interval_days: Days between retraining
        device: Device to use

    Returns:
        Experiment results dictionary
    """
    print(f"\n{'=' * 70}")
    print(f"Periodic Retraining: {model_name.upper()} | Patient {patient_id}")
    print(
        f"Horizon: {horizon_minutes}min | Retrain every {retraining_interval_days} days"
    )
    print(f"{'=' * 70}")

    # Set seed for reproducibility
    set_seed(42)

    # Load data
    loader = OhioT1DMLoader(data_path)
    patient_data = loader.load_patient(patient_id)

    if patient_data is None:
        print(f"ERROR: Could not load patient {patient_id}")
        return {"error": f"Could not load patient {patient_id}"}

    # Extract glucose values
    glucose_df = patient_data["glucose"]
    if glucose_df.empty:
        print(f"ERROR: No glucose data for patient {patient_id}")
        return {"error": "No glucose data"}

    # Preprocess
    glucose_values = glucose_df["glucose"].values
    timestamps = pd.to_datetime(glucose_df["timestamp"])

    glucose_clean, info = preprocess_glucose_series(
        glucose_values, timestamps=timestamps
    )

    # Remove any remaining NaN values
    valid_mask = ~np.isnan(glucose_clean)
    glucose_clean = glucose_clean[valid_mask]
    timestamps_clean = timestamps.values[valid_mask]

    # Normalize
    normalizer = GlucoseNormalizer()
    glucose_normalized = normalizer.fit_transform(glucose_clean)

    print(f"Data: {len(glucose_clean)} samples after preprocessing")

    # Create dataset
    seq_length = 24  # 2 hours of history
    horizon_steps = horizon_minutes // 5  # Convert to 5-min steps

    dataset = GlucoseDataset(
        glucose=glucose_normalized,
        timestamps=timestamps_clean,
        window_size=seq_length,
        horizon=horizon_steps,
    )

    print(f"Dataset: {len(dataset)} samples")

    # Get model class and configs
    model_class = get_model_class(model_name)
    model_config = get_model_config(model_name)
    training_config = get_training_config()

    # Create experiment
    experiment = PeriodicRetrainingExperiment(
        model_class=model_class,
        model_config=model_config,
        training_config=training_config,
        retraining_interval_days=retraining_interval_days,
        initial_training_days=7,
        device=device,
        results_dir=results_dir,
    )

    # Convert timestamps to datetime objects
    timestamps_dt = np.array(
        [pd.Timestamp(t).to_pydatetime() for t in timestamps_clean]
    )

    # Run experiment
    results = experiment.run(
        full_dataset=dataset,
        glucose_values=glucose_clean,
        timestamps=timestamps_dt,
        normalizer=normalizer,
        patient_id=patient_id,
    )

    # Add model info
    results["model"] = model_name
    results["horizon_minutes"] = horizon_minutes

    return results


def run_all_experiments(
    model_names: list,
    patient_ids: list,
    data_path: Path,
    results_dir: Path,
    horizon_minutes: int = 30,
    retraining_interval_days: int = 7,
    device: str = "cpu",
) -> dict:
    """Run experiments for all specified models and patients."""

    all_results = {}

    for model_name in model_names:
        all_results[model_name] = {}

        for patient_id in patient_ids:
            try:
                results = run_periodic_experiment(
                    model_name=model_name,
                    patient_id=patient_id,
                    data_path=data_path,
                    results_dir=results_dir / model_name,
                    horizon_minutes=horizon_minutes,
                    retraining_interval_days=retraining_interval_days,
                    device=device,
                )
                all_results[model_name][patient_id] = results
            except Exception as e:
                print(f"ERROR: {model_name}/{patient_id}: {e}")
                all_results[model_name][patient_id] = {"error": str(e)}

    return all_results


def print_summary(all_results: dict) -> None:
    """Print summary of all experiments."""
    print("\n" + "=" * 80)
    print("STAGE 2: PERIODIC RETRAINING - RESULTS SUMMARY")
    print("=" * 80)

    for model_name, model_results in all_results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)

        mae_values = []
        rmse_values = []
        clarke_values = []

        for patient_id, results in model_results.items():
            if "error" not in results and "summary" in results:
                summary = results["summary"]
                mae = summary["mae_mean"]
                rmse = summary["rmse_mean"]
                clarke = summary["clarke_ab_mean"]
                retrains = summary["total_retrains"]

                mae_values.append(mae)
                rmse_values.append(rmse)
                clarke_values.append(clarke)

                print(
                    f"  Patient {patient_id}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                    f"Clarke A+B={clarke:.1f}%, Retrains={retrains}"
                )

        if mae_values:
            print(
                f"\n  Mean: MAE={np.mean(mae_values):.2f} ± {np.std(mae_values):.2f}, "
                f"RMSE={np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f}, "
                f"Clarke A+B={np.mean(clarke_values):.1f}%"
            )


def compare_with_static(periodic_results: dict, static_results_path: Path) -> None:
    """Compare periodic results with static baseline."""
    print("\n" + "=" * 80)
    print("COMPARISON: STATIC vs PERIODIC RETRAINING")
    print("=" * 80)

    # Load static results if they exist
    # For now, use hardcoded values from Stage 1
    static_results = {
        "lstm": {"mae": 18.80, "rmse": 25.81, "clarke_ab": 97.68},
        "tcn": {"mae": 20.18, "rmse": 26.83, "clarke_ab": 96.61},
        "transformer": {"mae": 18.26, "rmse": 24.99, "clarke_ab": 97.52},
    }

    print(
        f"\n{'Model':<15} {'Strategy':<12} {'MAE':<10} {'RMSE':<10} {'Clarke A+B':<12}"
    )
    print("-" * 60)

    for model_name in periodic_results:
        # Static
        if model_name in static_results:
            static = static_results[model_name]
            print(
                f"{model_name.upper():<15} {'Static':<12} {static['mae']:<10.2f} "
                f"{static['rmse']:<10.2f} {static['clarke_ab']:<12.1f}%"
            )

        # Periodic
        model_data = periodic_results[model_name]
        mae_values = []
        rmse_values = []
        clarke_values = []

        for patient_id, results in model_data.items():
            if "error" not in results and "summary" in results:
                mae_values.append(results["summary"]["mae_mean"])
                rmse_values.append(results["summary"]["rmse_mean"])
                clarke_values.append(results["summary"]["clarke_ab_mean"])

        if mae_values:
            print(
                f"{'':<15} {'Periodic':<12} {np.mean(mae_values):<10.2f} "
                f"{np.mean(rmse_values):<10.2f} {np.mean(clarke_values):<12.1f}%"
            )

        # Improvement
        if model_name in static_results and mae_values:
            mae_improvement = static_results[model_name]["mae"] - np.mean(mae_values)
            print(f"{'':<15} {'Δ Improve':<12} {mae_improvement:+<10.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AGNI Framework - Stage 2: Periodic Retraining Experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lstm", "tcn", "transformer", "all"],
        help="Model to run (default: all)",
    )
    parser.add_argument(
        "--patient", type=str, default="all", help='Patient ID or "all" (default: all)'
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        choices=[15, 30, 60],
        help="Prediction horizon in minutes (default: 30)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=7,
        help="Retraining interval in days (default: 7)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/rahulkashyap/Desktop/DIT/OhioT1DM/archive",
        help="Path to OhioT1DM data",
    )

    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data_path)
    results_dir = (
        Path(__file__).parent.parent
        / "results"
        / f"periodic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Determine models and patients
    if args.model == "all":
        model_names = ["lstm", "tcn", "transformer"]
    else:
        model_names = [args.model]

    if args.patient == "all":
        patient_ids = ["559", "563", "570", "575", "588", "591"]
    else:
        patient_ids = [args.patient]

    print(f"\nRunning Stage 2: Periodic Retraining Experiments")
    print(f"Models: {model_names}")
    print(f"Patients: {patient_ids}")
    print(f"Horizon: {args.horizon} minutes")
    print(f"Retraining interval: {args.interval} days")
    print(f"Results directory: {results_dir}")

    # Run experiments
    all_results = run_all_experiments(
        model_names=model_names,
        patient_ids=patient_ids,
        data_path=data_path,
        results_dir=results_dir,
        horizon_minutes=args.horizon,
        retraining_interval_days=args.interval,
        device=device,
    )

    # Print summary
    print_summary(all_results)

    # Compare with static baseline
    compare_with_static(all_results, results_dir)

    # Save combined results
    combined_path = results_dir / "combined_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(combined_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
