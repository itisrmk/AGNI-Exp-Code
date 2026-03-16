#!/usr/bin/env python3
"""
AGNI Framework - Stage 3: Continual Learning Experiment Runner

This script runs continual learning experiments on the OhioT1DM dataset,
combining EWC and Experience Replay for adaptive glucose prediction.

Usage:
    python scripts/run_continual_experiment.py --model lstm --patient 559
    python scripts/run_continual_experiment.py --model all --patient all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from adaptation.continual import ContinualLearningExperiment
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
    """Get training configuration for continual learning."""
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "weight_decay": 1e-5,
        "update_lr": 0.0001,  # Lower LR for updates
        "update_epochs": 10,  # Fewer epochs for updates
    }


def run_continual_experiment(
    model_name: str,
    patient_id: str,
    data_path: Path,
    results_dir: Path,
    horizon_minutes: int = 30,
    ewc_lambda: float = 1000.0,
    replay_buffer_size: int = 500,
    device: str = "cpu",
) -> dict:
    """
    Run a single continual learning experiment.
    """
    print(f"\n{'=' * 70}")
    print(f"Continual Learning: {model_name.upper()} | Patient {patient_id}")
    print(
        f"Horizon: {horizon_minutes}min | EWC λ={ewc_lambda} | Buffer={replay_buffer_size}"
    )
    print(f"{'=' * 70}")

    # Set seed
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

    # Remove NaN values
    valid_mask = ~np.isnan(glucose_clean)
    glucose_clean = glucose_clean[valid_mask]
    timestamps_clean = timestamps.values[valid_mask]

    # Store raw glucose for stratification
    glucose_raw = glucose_clean.copy()

    # Normalize
    normalizer = GlucoseNormalizer()
    glucose_normalized = normalizer.fit_transform(glucose_clean)

    print(f"Data: {len(glucose_clean)} samples after preprocessing")

    # Create dataset
    seq_length = 24
    horizon_steps = horizon_minutes // 5

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

    # Convert timestamps to datetime
    timestamps_dt = np.array(
        [pd.Timestamp(t).to_pydatetime() for t in timestamps_clean]
    )

    # Create experiment
    experiment = ContinualLearningExperiment(
        model_class=model_class,
        model_config=model_config,
        training_config=training_config,
        ewc_lambda=ewc_lambda,
        replay_buffer_size=replay_buffer_size,
        replay_ratio=0.3,
        update_interval_days=1,
        initial_training_days=7,
        device=device,
        results_dir=results_dir,
    )

    # Run experiment
    results = experiment.run(
        full_dataset=dataset,
        glucose_raw=glucose_raw,
        timestamps=timestamps_dt,
        normalizer=normalizer,
        patient_id=patient_id,
    )

    results["model"] = model_name
    results["horizon_minutes"] = horizon_minutes

    return results


def run_all_experiments(
    model_names: list,
    patient_ids: list,
    data_path: Path,
    results_dir: Path,
    horizon_minutes: int = 30,
    ewc_lambda: float = 1000.0,
    replay_buffer_size: int = 500,
    device: str = "cpu",
) -> dict:
    """Run experiments for all specified models and patients."""

    all_results = {}

    for model_name in model_names:
        all_results[model_name] = {}

        for patient_id in patient_ids:
            try:
                results = run_continual_experiment(
                    model_name=model_name,
                    patient_id=patient_id,
                    data_path=data_path,
                    results_dir=results_dir / model_name,
                    horizon_minutes=horizon_minutes,
                    ewc_lambda=ewc_lambda,
                    replay_buffer_size=replay_buffer_size,
                    device=device,
                )
                all_results[model_name][patient_id] = results
            except Exception as e:
                print(f"ERROR: {model_name}/{patient_id}: {e}")
                import traceback

                traceback.print_exc()
                all_results[model_name][patient_id] = {"error": str(e)}

    return all_results


def print_summary(all_results: dict) -> None:
    """Print summary of all experiments."""
    print("\n" + "=" * 80)
    print("STAGE 3: CONTINUAL LEARNING - RESULTS SUMMARY")
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
                updates = summary["total_updates"]

                mae_values.append(mae)
                rmse_values.append(rmse)
                clarke_values.append(clarke)

                print(
                    f"  Patient {patient_id}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                    f"Clarke A+B={clarke:.1f}%, Updates={updates}"
                )

        if mae_values:
            print(
                f"\n  Mean: MAE={np.mean(mae_values):.2f} ± {np.std(mae_values):.2f}, "
                f"RMSE={np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f}, "
                f"Clarke A+B={np.mean(clarke_values):.1f}%"
            )


def compare_all_strategies(continual_results: dict) -> None:
    """Compare all three strategies."""
    print("\n" + "=" * 80)
    print("COMPARISON: STATIC vs PERIODIC vs CONTINUAL")
    print("=" * 80)

    # Previous results
    static_results = {
        "lstm": {"mae": 18.80, "rmse": 25.81, "clarke": 97.68},
        "tcn": {"mae": 20.18, "rmse": 26.83, "clarke": 96.61},
        "transformer": {"mae": 18.26, "rmse": 24.99, "clarke": 97.52},
    }

    periodic_results = {
        "lstm": {"mae": 32.37, "rmse": 39.56, "clarke": 97.2},
        "tcn": {"mae": 19.19, "rmse": 25.92, "clarke": 96.9},
        "transformer": {"mae": 18.23, "rmse": 24.99, "clarke": 97.1},
    }

    print(
        f"\n{'Model':<12} {'Strategy':<12} {'MAE':<10} {'RMSE':<10} {'Clarke A+B':<12}"
    )
    print("-" * 60)

    for model in ["lstm", "tcn", "transformer"]:
        # Static
        s = static_results[model]
        print(
            f"{model.upper():<12} {'Static':<12} {s['mae']:<10.2f} {s['rmse']:<10.2f} {s['clarke']:<12.1f}%"
        )

        # Periodic
        p = periodic_results[model]
        print(
            f"{'':<12} {'Periodic':<12} {p['mae']:<10.2f} {p['rmse']:<10.2f} {p['clarke']:<12.1f}%"
        )

        # Continual
        if model in continual_results:
            model_data = continual_results[model]
            mae_values = [
                v["summary"]["mae_mean"]
                for v in model_data.values()
                if "summary" in v and "error" not in v
            ]
            rmse_values = [
                v["summary"]["rmse_mean"]
                for v in model_data.values()
                if "summary" in v and "error" not in v
            ]
            clarke_values = [
                v["summary"]["clarke_ab_mean"]
                for v in model_data.values()
                if "summary" in v and "error" not in v
            ]

            if mae_values:
                c_mae = np.mean(mae_values)
                c_rmse = np.mean(rmse_values)
                c_clarke = np.mean(clarke_values)
                print(
                    f"{'':<12} {'Continual':<12} {c_mae:<10.2f} {c_rmse:<10.2f} {c_clarke:<12.1f}%"
                )

                # Best improvement
                best_static_mae = s["mae"]
                improvement = best_static_mae - c_mae
                print(f"{'':<12} {'Δ vs Static':<12} {improvement:+.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AGNI Framework - Stage 3: Continual Learning Experiments"
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
        "--ewc-lambda",
        type=float,
        default=1000.0,
        help="EWC regularization strength (default: 1000)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=500,
        help="Experience replay buffer size (default: 500)",
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
        / f"continual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    print(f"\nRunning Stage 3: Continual Learning Experiments")
    print(f"Models: {model_names}")
    print(f"Patients: {patient_ids}")
    print(f"Horizon: {args.horizon} minutes")
    print(f"EWC Lambda: {args.ewc_lambda}")
    print(f"Buffer Size: {args.buffer_size}")
    print(f"Results directory: {results_dir}")

    # Run experiments
    all_results = run_all_experiments(
        model_names=model_names,
        patient_ids=patient_ids,
        data_path=data_path,
        results_dir=results_dir,
        horizon_minutes=args.horizon,
        ewc_lambda=args.ewc_lambda,
        replay_buffer_size=args.buffer_size,
        device=device,
    )

    # Print summary
    print_summary(all_results)

    # Compare all strategies
    compare_all_strategies(all_results)

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
