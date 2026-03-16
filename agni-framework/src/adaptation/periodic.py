"""
AGNI Framework - Periodic Retraining Adapter
Stage 2: Implements periodic model retraining strategy

This module provides:
- PeriodicAdapter: Determines when to retrain and handles the retraining process
- PeriodicRetrainingExperiment: Runs complete periodic retraining experiments
"""

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from ..data.dataset import GlucoseDataset
    from ..evaluation.clinical import clarke_error_grid, hypoglycemia_detection
    from ..evaluation.metrics import MetricsTracker, compute_all_metrics
    from ..experiments.trainer import Trainer
    from ..models.base import BaseGlucoseModel
except ImportError:
    from data.dataset import GlucoseDataset
    from evaluation.clinical import clarke_error_grid, hypoglycemia_detection
    from evaluation.metrics import MetricsTracker, compute_all_metrics
    from experiments.trainer import Trainer
    from models.base import BaseGlucoseModel


class PeriodicAdapter:
    """
    Periodic Retraining Adapter

    Strategy: Retrain the model from scratch at fixed intervals (e.g., every 7 days)
    using a sliding window of recent data.

    This establishes a baseline for comparison against continual learning approaches.
    """

    def __init__(
        self,
        model_class: type,
        model_config: Dict,
        training_config: Dict,
        retraining_interval_days: int = 7,
        window_size_days: int = 14,
        device: str = "cpu",
    ):
        """
        Initialize the Periodic Adapter.

        Args:
            model_class: Class of the model to instantiate (e.g., LSTMPredictor)
            model_config: Configuration dict for the model
            training_config: Configuration dict for training (epochs, lr, etc.)
            retraining_interval_days: Days between retraining (default: 7)
            window_size_days: Days of data to use for each retraining (default: 14)
            device: Device to use ('cpu', 'mps', 'cuda')
        """
        self.model_class = model_class
        self.model_config = model_config
        self.training_config = training_config
        self.retraining_interval_days = retraining_interval_days
        self.window_size_days = window_size_days
        self.device = device

        # Current model
        self.model: Optional[BaseGlucoseModel] = None

        # Tracking
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_history: List[Dict] = []
        self.daily_metrics: List[Dict] = []

    def initialize(self, initial_data: GlucoseDataset) -> None:
        """
        Initialize the adapter with initial training.

        Args:
            initial_data: Initial dataset for first training
        """
        self.model = self._create_model()
        self._train_model(initial_data, is_initial=True)
        self.last_retrain_time = datetime.now()

    def _create_model(self) -> BaseGlucoseModel:
        """Create a fresh instance of the model."""
        model = self.model_class(self.model_config)
        model.to(self.device)
        return model

    def _train_model(
        self, dataset: GlucoseDataset, is_initial: bool = False
    ) -> Dict[str, float]:
        """
        Train the model on the given dataset.

        Args:
            dataset: Training dataset
            is_initial: Whether this is the initial training

        Returns:
            Training metrics
        """
        # Split into train/val (80/20)
        n_samples = len(dataset)
        n_train = int(0.8 * n_samples)

        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_samples))

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.get("batch_size", 32),
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.get("batch_size", 32),
            shuffle=False,
        )

        # Create optimizer and criterion
        lr = self.training_config.get("learning_rate", 0.001)
        weight_decay = self.training_config.get("weight_decay", 1e-5)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        # Create trainer
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device(self.device),
            config=self.training_config,
        )

        # Train
        epochs = self.training_config.get("epochs", 50)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=epochs,
            verbose=False,
        )

        return {
            "final_train_loss": history["train_loss"][-1]
            if history["train_loss"]
            else 0,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else 0,
            "epochs_trained": len(history["train_loss"]),
        }

    def should_retrain(self, current_time: datetime) -> bool:
        """
        Check if it's time to retrain the model.

        Args:
            current_time: Current timestamp

        Returns:
            True if retraining should occur
        """
        if self.last_retrain_time is None:
            return True

        days_since_retrain = (current_time - self.last_retrain_time).days
        return days_since_retrain >= self.retraining_interval_days

    def retrain(
        self, recent_data: GlucoseDataset, current_time: datetime
    ) -> Dict[str, Any]:
        """
        Perform a complete retraining from scratch.

        Args:
            recent_data: Recent data to train on
            current_time: Current timestamp

        Returns:
            Retraining metrics
        """
        # Create fresh model
        self.model = self._create_model()

        # Train from scratch
        train_metrics = self._train_model(recent_data, is_initial=False)

        # Record retraining event
        retrain_record = {
            "timestamp": current_time.isoformat(),
            "data_samples": len(recent_data),
            **train_metrics,
        }
        self.retrain_history.append(retrain_record)

        # Update last retrain time
        self.last_retrain_time = current_time

        return retrain_record

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the current model.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    def evaluate(
        self, test_data: GlucoseDataset, normalizer: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate the current model on test data.

        Args:
            test_data: Test dataset
            normalizer: Normalizer for inverse transform (optional)

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                predictions = self.model(batch_x)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.numpy())

        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # Inverse transform if normalizer provided
        if normalizer is not None:
            predictions = normalizer.inverse_transform(predictions.flatten())
            targets = normalizer.inverse_transform(targets.flatten())
        else:
            predictions = predictions.flatten()
            targets = targets.flatten()

        # Compute metrics
        metrics = compute_all_metrics(predictions, targets)

        # Add Clarke Error Grid
        clarke = clarke_error_grid(targets, predictions)
        metrics["clarke_a"] = clarke["A"]
        metrics["clarke_b"] = clarke["B"]
        metrics["clarke_ab"] = clarke["A+B"]

        return metrics

    def record_daily_metrics(
        self, day: int, metrics: Dict[str, float], days_since_retrain: int
    ) -> None:
        """
        Record metrics for a specific day.

        Args:
            day: Day number
            metrics: Metrics dict
            days_since_retrain: Days since last retraining
        """
        self.daily_metrics.append(
            {"day": day, "days_since_retrain": days_since_retrain, **metrics}
        )

    def get_history(self) -> Dict[str, Any]:
        """Get complete history of the adapter."""
        return {
            "retrain_history": self.retrain_history,
            "daily_metrics": self.daily_metrics,
            "retraining_interval_days": self.retraining_interval_days,
            "window_size_days": self.window_size_days,
            "total_retrains": len(self.retrain_history),
        }


class PeriodicRetrainingExperiment:
    """
    Runs a complete periodic retraining experiment.

    Simulates temporal progression through the data, performing periodic
    retraining and evaluating on each day's data.
    """

    def __init__(
        self,
        model_class: type,
        model_config: Dict,
        training_config: Dict,
        retraining_interval_days: int = 7,
        initial_training_days: int = 7,
        device: str = "cpu",
        results_dir: Optional[Path] = None,
    ):
        """
        Initialize the experiment.

        Args:
            model_class: Model class to use
            model_config: Model configuration
            training_config: Training configuration
            retraining_interval_days: Days between retraining
            initial_training_days: Days of data for initial training
            device: Device to use
            results_dir: Directory for saving results
        """
        self.model_class = model_class
        self.model_config = model_config
        self.training_config = training_config
        self.retraining_interval_days = retraining_interval_days
        self.initial_training_days = initial_training_days
        self.device = device
        self.results_dir = results_dir

    def run(
        self,
        full_dataset: GlucoseDataset,
        glucose_values: np.ndarray,
        timestamps: np.ndarray,
        normalizer: Any,
        patient_id: str,
    ) -> Dict[str, Any]:
        """
        Run the complete periodic retraining experiment.

        Args:
            full_dataset: Complete dataset
            glucose_values: Raw glucose values
            timestamps: Timestamps for each sample
            normalizer: Normalizer for glucose values
            patient_id: Patient identifier

        Returns:
            Complete experiment results
        """
        print(f"\n{'=' * 60}")
        print(f"Running Periodic Retraining Experiment - Patient {patient_id}")
        print(f"Retraining interval: {self.retraining_interval_days} days")
        print(f"{'=' * 60}")

        # Convert timestamps to datetime if needed
        if isinstance(timestamps[0], (int, float, np.integer, np.floating)):
            # Assume timestamps are Unix timestamps
            timestamps = np.array([datetime.fromtimestamp(t) for t in timestamps])

        # Determine the date range
        min_date = min(timestamps)
        max_date = max(timestamps)
        total_days = (max_date - min_date).days + 1

        print(f"Data spans {total_days} days: {min_date.date()} to {max_date.date()}")

        # Create adapter
        adapter = PeriodicAdapter(
            model_class=self.model_class,
            model_config=self.model_config,
            training_config=self.training_config,
            retraining_interval_days=self.retraining_interval_days,
            device=self.device,
        )

        # Get indices for each day
        days_data = self._split_by_day(full_dataset, timestamps, min_date, total_days)

        # Initial training on first N days
        initial_indices = []
        for day in range(min(self.initial_training_days, len(days_data))):
            if day in days_data:
                initial_indices.extend(days_data[day])

        if not initial_indices:
            raise ValueError("Not enough data for initial training")

        initial_dataset = Subset(full_dataset, initial_indices)
        print(
            f"\nInitial training on {len(initial_indices)} samples ({self.initial_training_days} days)"
        )

        # Initialize adapter
        adapter.model = adapter._create_model()
        adapter._train_model(initial_dataset, is_initial=True)
        adapter.last_retrain_time = min_date + timedelta(
            days=self.initial_training_days
        )

        # Track results
        all_metrics = []
        retrain_events = []

        # Iterate through remaining days
        for day in range(self.initial_training_days, total_days):
            if day not in days_data or len(days_data[day]) == 0:
                continue

            current_date = min_date + timedelta(days=day)
            days_since_retrain = (current_date - adapter.last_retrain_time).days

            # Check if retraining is needed
            if adapter.should_retrain(current_date):
                # Gather data from window (use 14-day window for better training data)
                window_size = max(14, self.retraining_interval_days * 2)
                window_start = max(0, day - window_size)
                window_indices = []
                for d in range(window_start, day):
                    if d in days_data:
                        window_indices.extend(days_data[d])

                if window_indices:
                    window_dataset = Subset(full_dataset, window_indices)
                    print(f"\nDay {day}: Retraining on {len(window_indices)} samples")
                    adapter.retrain(window_dataset, current_date)
                    days_since_retrain = 0
                    retrain_events.append(day)

            # Evaluate on current day's data
            day_indices = days_data[day]
            if len(day_indices) > 10:  # Need minimum samples
                day_dataset = Subset(full_dataset, day_indices)
                metrics = adapter.evaluate(day_dataset, normalizer)
                metrics["day"] = day
                metrics["days_since_retrain"] = days_since_retrain
                all_metrics.append(metrics)

                if day % 7 == 0:
                    print(
                        f"Day {day}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
                        f"Clarke A+B={metrics['clarke_ab']:.1f}%"
                    )

        # Compute summary statistics
        results = self._compute_summary(all_metrics, retrain_events, patient_id)

        # Save results
        if self.results_dir:
            self._save_results(results, patient_id)

        return results

    def _split_by_day(
        self,
        dataset: GlucoseDataset,
        timestamps: np.ndarray,
        start_date: datetime,
        total_days: int,
    ) -> Dict[int, List[int]]:
        """Split dataset indices by day."""
        days_data = {d: [] for d in range(total_days)}

        for idx in range(len(dataset)):
            # Map sample to day
            sample_time = timestamps[idx + dataset.window_size]  # Adjust for sequence
            day = (sample_time - start_date).days
            if 0 <= day < total_days:
                days_data[day].append(idx)

        return days_data

    def _compute_summary(
        self, all_metrics: List[Dict], retrain_events: List[int], patient_id: str
    ) -> Dict[str, Any]:
        """Compute summary statistics from daily metrics."""
        if not all_metrics:
            return {"error": "No metrics collected"}

        # Convert to numpy arrays
        mae_values = np.array([m["mae"] for m in all_metrics])
        rmse_values = np.array([m["rmse"] for m in all_metrics])
        clarke_ab = np.array([m["clarke_ab"] for m in all_metrics])
        days_since_retrain = np.array([m["days_since_retrain"] for m in all_metrics])

        # Group by days since retrain (for sawtooth analysis)
        sawtooth_data = {}
        for m in all_metrics:
            dsr = m["days_since_retrain"]
            if dsr not in sawtooth_data:
                sawtooth_data[dsr] = {"mae": [], "rmse": [], "clarke_ab": []}
            sawtooth_data[dsr]["mae"].append(m["mae"])
            sawtooth_data[dsr]["rmse"].append(m["rmse"])
            sawtooth_data[dsr]["clarke_ab"].append(m["clarke_ab"])

        # Average by days since retrain
        sawtooth_summary = {}
        for dsr in sorted(sawtooth_data.keys()):
            sawtooth_summary[dsr] = {
                "mae_mean": np.mean(sawtooth_data[dsr]["mae"]),
                "mae_std": np.std(sawtooth_data[dsr]["mae"]),
                "rmse_mean": np.mean(sawtooth_data[dsr]["rmse"]),
                "rmse_std": np.std(sawtooth_data[dsr]["rmse"]),
                "n_samples": len(sawtooth_data[dsr]["mae"]),
            }

        return {
            "patient_id": patient_id,
            "strategy": "periodic",
            "retraining_interval_days": self.retraining_interval_days,
            "summary": {
                "mae_mean": float(np.mean(mae_values)),
                "mae_std": float(np.std(mae_values)),
                "rmse_mean": float(np.mean(rmse_values)),
                "rmse_std": float(np.std(rmse_values)),
                "clarke_ab_mean": float(np.mean(clarke_ab)),
                "clarke_ab_std": float(np.std(clarke_ab)),
                "total_days_evaluated": len(all_metrics),
                "total_retrains": len(retrain_events),
            },
            "sawtooth_pattern": sawtooth_summary,
            "daily_metrics": all_metrics,
            "retrain_events": retrain_events,
        }

    def _save_results(self, results: Dict, patient_id: str) -> None:
        """Save results to file."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
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

        results = convert_numpy(results)

        filepath = self.results_dir / f"periodic_{patient_id}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
