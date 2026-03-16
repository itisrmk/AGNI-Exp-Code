#!/usr/bin/env python3
"""
Main experiment runner for AGNI framework - Stage 1: Static Baseline.

Usage:
    python scripts/run_experiment.py --model all --horizon 30
    python scripts/run_experiment.py --model lstm --patient 559
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from src.data.dataset import GlucoseDataset, TemporalSplitter
from src.data.ohio_loader import OhioT1DMLoader, get_patient_stats, prepare_patient_data
from src.data.preprocessing import GlucoseNormalizer, preprocess_glucose_series
from src.evaluation.clinical import (
    clarke_error_grid,
    compute_all_clinical_metrics,
    hypoglycemia_detection,
)
from src.evaluation.metrics import compute_all_metrics, compute_glucose_metrics
from src.experiments.trainer import Trainer, create_criterion, create_optimizer
from src.models.base import create_model
from src.utils.config import load_config, save_config
from src.utils.device import get_device
from src.utils.reproducibility import set_seed
from torch.utils.data import DataLoader


def run_single_experiment(
    patient_id: int,
    model_type: str,
    horizon: int,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """
    Run a single experiment for one patient/model/horizon combination.

    Args:
        patient_id: OhioT1DM patient ID
        model_type: Model architecture ('lstm', 'tcn', 'transformer')
        horizon: Prediction horizon in minutes
        config: Configuration dictionary
        device: Torch device
        verbose: Whether to print progress

    Returns:
        Dictionary with experiment results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Patient {patient_id} | Model: {model_type.upper()} | Horizon: {horizon}min"
        )
        print("=" * 60)

    # Load data
    data_dir = Path(__file__).parent.parent / config["data"]["data_dir"]
    loader = OhioT1DMLoader(data_dir)

    try:
        patient_data = loader.load_patient(patient_id, "training")
    except FileNotFoundError:
        print(f"Patient {patient_id} not found, skipping...")
        return None

    # Prepare data
    df = prepare_patient_data(patient_data)

    if df.empty or len(df) < 500:
        print(f"Insufficient data for patient {patient_id}")
        return None

    # Get patient stats
    stats = get_patient_stats(patient_data)
    if verbose:
        print(
            f"Data: {stats['n_readings']} readings over {stats['duration_days']} days"
        )
        print(
            f"Glucose: {stats['mean_glucose']:.1f} +/- {stats['std_glucose']:.1f} mg/dL"
        )
        print(f"TIR: {stats['time_in_range']:.1f}%")

    # Preprocess
    glucose_raw = df["glucose"].values
    glucose_clean, prep_info = preprocess_glucose_series(glucose_raw)

    # Normalize
    normalizer = GlucoseNormalizer()
    glucose_normalized = normalizer.fit_transform(glucose_clean)

    # Create dataset
    horizon_steps = horizon // 5  # Convert minutes to 5-min steps
    window_size = config["data"]["window_size"]

    dataset = GlucoseDataset(
        glucose=glucose_normalized,
        timestamps=df["timestamp"].values if "timestamp" in df.columns else None,
        window_size=window_size,
        horizon=horizon_steps,
        glucose_raw=glucose_raw,
    )

    if len(dataset) < 100:
        print(f"Insufficient samples for patient {patient_id}: {len(dataset)}")
        return None

    # Split data
    splitter = TemporalSplitter(
        train_ratio=config["data"]["train_split"], val_ratio=config["data"]["val_split"]
    )
    train_subset, val_subset, test_subset = splitter.split_dataset(dataset)

    if verbose:
        print(
            f"Split: {len(train_subset)} train, {len(val_subset)} val, {len(test_subset)} test"
        )

    # Create data loaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    test_loader = DataLoader(test_subset, batch_size=batch_size)

    # Create model
    model_config = config["models"].get(model_type, {})
    model_config["input_size"] = 1
    model = create_model(model_type, model_config)
    model = model.to(device)

    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")

    # Create trainer
    optimizer = create_optimizer(
        model, config["training"]["optimizer"], config["training"]["learning_rate"]
    )
    criterion = create_criterion(config["training"]["loss"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config["training"],
    )

    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        max_epochs=config["training"]["max_epochs"],
        verbose=verbose,
    )

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    y_pred_normalized = np.concatenate(all_preds)
    y_true_normalized = np.concatenate(all_targets)

    # Convert back to mg/dL
    y_pred = normalizer.inverse_transform(y_pred_normalized)
    y_true = normalizer.inverse_transform(y_true_normalized)

    # Compute metrics
    metrics = compute_glucose_metrics(y_true, y_pred)
    clinical = compute_all_clinical_metrics(y_true, y_pred)

    if verbose:
        print(f"\nTest Results:")
        print(f"  MAE:  {metrics['mae']:.2f} mg/dL")
        print(f"  RMSE: {metrics['rmse']:.2f} mg/dL")
        print(f"  Corr: {metrics['correlation']:.3f}")
        print(f"  Clarke A+B: {clinical['clarke_grid']['A+B']:.1f}%")

    # Compile results
    results = {
        "patient_id": patient_id,
        "model_type": model_type,
        "horizon_minutes": horizon,
        "n_train": len(train_subset),
        "n_val": len(val_subset),
        "n_test": len(test_subset),
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "metrics": metrics,
        "clinical": clinical,
        "patient_stats": stats,
        "normalizer_params": normalizer.get_params(),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run AGNI Stage 1 experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "tcn", "transformer", "all"],
        default="all",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[15, 30, 60],
        default=30,
        help="Prediction horizon in minutes",
    )
    parser.add_argument(
        "--patient",
        type=int,
        default=None,
        help="Single patient ID to run (default: all available)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {
            "data": {
                "data_dir": "../OhioT1DM/archive",
                "window_size": 24,
                "train_split": 0.5,
                "val_split": 0.2,
            },
            "models": {
                "lstm": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
                "tcn": {
                    "num_channels": [32, 32, 32, 32],
                    "kernel_size": 3,
                    "dropout": 0.2,
                },
                "transformer": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 1,
                    "dropout": 0.1,
                },
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "max_epochs": 100,
                "early_stopping_patience": 10,
                "optimizer": "adam",
                "loss": "mse",
            },
        }

    # Get device
    device = get_device(config.get("device", "mps"))
    print(f"Using device: {device}")

    # Determine models and patients
    models = ["lstm", "tcn", "transformer"] if args.model == "all" else [args.model]

    data_dir = Path(__file__).parent.parent / config["data"]["data_dir"]
    loader = OhioT1DMLoader(data_dir)

    if args.patient:
        patients = [args.patient]
    else:
        patients = loader.get_available_patients()

    print(f"\nExperiment Configuration:")
    print(f"  Models: {models}")
    print(f"  Horizon: {args.horizon} minutes")
    print(f"  Patients: {patients}")
    print(f"  Seed: {args.seed}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"static_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, output_dir / "config.yaml")

    # Run experiments
    all_results = []

    for patient_id in patients:
        for model_type in models:
            result = run_single_experiment(
                patient_id=patient_id,
                model_type=model_type,
                horizon=args.horizon,
                config=config,
                device=device,
                verbose=not args.quiet,
            )

            if result is not None:
                all_results.append(result)

                # Save individual result
                result_file = (
                    output_dir
                    / f"patient{patient_id}_{model_type}_{args.horizon}min.json"
                )
                with open(result_file, "w") as f:
                    # Convert numpy types for JSON
                    json_result = json.loads(
                        json.dumps(
                            result,
                            default=lambda x: float(x)
                            if isinstance(x, np.floating)
                            else x,
                        )
                    )
                    json.dump(json_result, f, indent=2)

    # Aggregate results
    if all_results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)

        # Create summary DataFrame
        summary_data = []
        for r in all_results:
            summary_data.append(
                {
                    "patient": r["patient_id"],
                    "model": r["model_type"],
                    "horizon": r["horizon_minutes"],
                    "mae": r["metrics"]["mae"],
                    "rmse": r["metrics"]["rmse"],
                    "corr": r["metrics"]["correlation"],
                    "clarke_ab": r["clinical"]["clarke_grid"]["A+B"],
                }
            )

        df_summary = pd.DataFrame(summary_data)

        # Print summary by model
        print("\nResults by Model:")
        print(
            df_summary.groupby("model")[["mae", "rmse", "corr", "clarke_ab"]]
            .mean()
            .round(2)
        )

        # Save summary
        df_summary.to_csv(output_dir / "summary.csv", index=False)

        print(f"\nResults saved to: {output_dir}")
    else:
        print("No results generated!")

    return all_results


if __name__ == "__main__":
    main()
