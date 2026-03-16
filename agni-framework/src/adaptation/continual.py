"""
AGNI Framework - Continual Learning Adapter
Stage 3: Combines EWC + Experience Replay for adaptive glucose prediction

This module provides the ContinualAdapter class that:
1. Uses EWC to protect important learned parameters
2. Uses Experience Replay to remember critical past events
3. Performs daily micro-updates instead of full retraining
"""

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

try:
    from ..data.dataset import GlucoseDataset
    from ..evaluation.clinical import clarke_error_grid
    from ..evaluation.metrics import compute_all_metrics
    from ..models.base import BaseGlucoseModel
    from .ewc import EWC, OnlineEWC
    from .replay_buffer import StratifiedReplayBuffer
except ImportError:
    from adaptation.ewc import EWC, OnlineEWC
    from adaptation.replay_buffer import StratifiedReplayBuffer
    from data.dataset import GlucoseDataset
    from evaluation.clinical import clarke_error_grid
    from evaluation.metrics import compute_all_metrics
    from models.base import BaseGlucoseModel


class ContinualAdapter:
    """
    Continual Learning Adapter combining EWC and Experience Replay.

    Strategy:
    1. Initial training on first N days of data
    2. Consolidate knowledge with EWC
    3. Daily updates using:
       - New day's data
       - Replay buffer samples (mixed in)
       - EWC regularization (prevents forgetting)
    """

    def __init__(
        self,
        model_class: type,
        model_config: Dict,
        training_config: Dict,
        ewc_lambda: float = 1000.0,
        replay_buffer_size: int = 500,
        replay_ratio: float = 0.3,
        update_interval_days: int = 1,
        device: str = "cpu",
    ):
        """
        Initialize the Continual Adapter.

        Args:
            model_class: Class of the model to instantiate
            model_config: Configuration dict for the model
            training_config: Configuration dict for training
            ewc_lambda: Strength of EWC penalty
            replay_buffer_size: Size of experience replay buffer
            replay_ratio: Fraction of batch to sample from replay
            update_interval_days: Days between updates (default: 1)
            device: Device to use
        """
        self.model_class = model_class
        self.model_config = model_config
        self.training_config = training_config
        self.ewc_lambda = ewc_lambda
        self.replay_buffer_size = replay_buffer_size
        self.replay_ratio = replay_ratio
        self.update_interval_days = update_interval_days
        self.device = device

        # Model and components
        self.model: Optional[BaseGlucoseModel] = None
        self.ewc: Optional[EWC] = None
        self.replay_buffer: Optional[StratifiedReplayBuffer] = None

        # Tracking
        self.last_update_time: Optional[datetime] = None
        self.update_history: List[Dict] = []
        self.daily_metrics: List[Dict] = []

    def initialize(
        self, initial_dataset: GlucoseDataset, glucose_raw: np.ndarray
    ) -> Dict[str, float]:
        """
        Initialize with initial training and consolidation.

        Args:
            initial_dataset: Dataset for initial training
            glucose_raw: Raw glucose values for replay buffer stratification

        Returns:
            Training metrics
        """
        # Create model
        self.model = self._create_model()

        # Create EWC and replay buffer
        self.ewc = OnlineEWC(self.model, ewc_lambda=self.ewc_lambda)
        self.replay_buffer = StratifiedReplayBuffer(capacity=self.replay_buffer_size)

        # Initial training (without EWC)
        train_metrics = self._train_initial(initial_dataset)

        # Populate replay buffer with initial data
        self._populate_replay_buffer(initial_dataset, glucose_raw)

        # Consolidate with EWC
        train_loader = DataLoader(initial_dataset, batch_size=32, shuffle=False)
        criterion = nn.MSELoss()
        self.ewc.consolidate(train_loader, criterion, torch.device(self.device))

        self.last_update_time = datetime.now()

        return train_metrics

    def _create_model(self) -> BaseGlucoseModel:
        """Create a fresh instance of the model."""
        model = self.model_class(self.model_config)
        model.to(self.device)
        return model

    def _train_initial(self, dataset: GlucoseDataset) -> Dict[str, float]:
        """Initial training without EWC."""
        # Split into train/val
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

        # Training setup
        lr = self.training_config.get("learning_rate", 0.001)
        weight_decay = self.training_config.get("weight_decay", 1e-5)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        # Train
        epochs = self.training_config.get("epochs", 50)
        best_val_loss = float("inf")
        patience = self.training_config.get("patience", 10)
        patience_counter = 0

        for epoch in range(epochs):
            # Train epoch
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(train_dataset)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_loss /= len(val_dataset)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
        }

    def _populate_replay_buffer(
        self, dataset: GlucoseDataset, glucose_raw: np.ndarray
    ) -> None:
        """Populate replay buffer with dataset samples."""
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            # Get raw glucose value for stratification
            # glucose_raw should be aligned with dataset indices
            if idx < len(glucose_raw):
                glucose_value = glucose_raw[idx]
            else:
                glucose_value = 120.0  # Default to normal range
            self.replay_buffer.add(x, y, glucose_value)

    def should_update(self, current_time: datetime) -> bool:
        """Check if it's time for an update."""
        if self.last_update_time is None:
            return True

        days_since_update = (current_time - self.last_update_time).days
        return days_since_update >= self.update_interval_days

    def update(
        self, new_data: GlucoseDataset, glucose_raw: np.ndarray, current_time: datetime
    ) -> Dict[str, Any]:
        """
        Perform a continual learning update.

        Args:
            new_data: New data to learn from
            glucose_raw: Raw glucose values for stratification
            current_time: Current timestamp

        Returns:
            Update metrics
        """
        # Add new data to replay buffer
        self._populate_replay_buffer(new_data, glucose_raw)

        # Create combined training data: new data + replay samples
        new_loader = DataLoader(new_data, batch_size=32, shuffle=True)

        # Training setup
        lr = self.training_config.get("update_lr", 0.0001)  # Lower LR for updates
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Fewer epochs for updates
        update_epochs = self.training_config.get("update_epochs", 10)

        self.model.train()
        total_loss = 0.0
        ewc_loss = 0.0
        n_batches = 0

        for epoch in range(update_epochs):
            for batch_x, batch_y in new_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Get replay samples
                replay_size = int(batch_x.size(0) * self.replay_ratio)
                if len(self.replay_buffer) > 0 and replay_size > 0:
                    replay_x, replay_y = self.replay_buffer.sample(replay_size)
                    replay_x = replay_x.to(self.device)
                    replay_y = replay_y.to(self.device)

                    # Combine new data with replay
                    batch_x = torch.cat([batch_x, replay_x], dim=0)
                    batch_y = torch.cat([batch_y, replay_y], dim=0)

                optimizer.zero_grad()
                output = self.model(batch_x)
                task_loss = criterion(output, batch_y)

                # Add EWC regularization
                total_batch_loss = self.ewc.get_regularized_loss(task_loss)

                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                ewc_loss += self.ewc.penalty().item()
                n_batches += 1

        # Re-consolidate with EWC (update Fisher information)
        all_loader = DataLoader(new_data, batch_size=32, shuffle=False)
        self.ewc.consolidate(all_loader, criterion, torch.device(self.device))

        # Record update
        update_record = {
            "timestamp": current_time.isoformat(),
            "data_samples": len(new_data),
            "replay_buffer_size": len(self.replay_buffer),
            "avg_loss": total_loss / n_batches if n_batches > 0 else 0,
            "avg_ewc_penalty": ewc_loss / n_batches if n_batches > 0 else 0,
            "epochs": update_epochs,
        }
        self.update_history.append(update_record)
        self.last_update_time = current_time

        return update_record

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    def evaluate(
        self, test_data: GlucoseDataset, normalizer: Optional[Any] = None
    ) -> Dict[str, float]:
        """Evaluate on test data."""
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

    def get_history(self) -> Dict[str, Any]:
        """Get complete history."""
        return {
            "update_history": self.update_history,
            "daily_metrics": self.daily_metrics,
            "replay_buffer_stats": self.replay_buffer.get_stats()
            if self.replay_buffer
            else {},
            "ewc_consolidations": self.ewc.n_consolidations if self.ewc else 0,
            "total_updates": len(self.update_history),
        }


class ContinualLearningExperiment:
    """
    Runs a complete continual learning experiment.
    """

    def __init__(
        self,
        model_class: type,
        model_config: Dict,
        training_config: Dict,
        ewc_lambda: float = 1000.0,
        replay_buffer_size: int = 500,
        replay_ratio: float = 0.3,
        update_interval_days: int = 1,
        initial_training_days: int = 7,
        device: str = "cpu",
        results_dir: Optional[Path] = None,
    ):
        """Initialize experiment."""
        self.model_class = model_class
        self.model_config = model_config
        self.training_config = training_config
        self.ewc_lambda = ewc_lambda
        self.replay_buffer_size = replay_buffer_size
        self.replay_ratio = replay_ratio
        self.update_interval_days = update_interval_days
        self.initial_training_days = initial_training_days
        self.device = device
        self.results_dir = results_dir

    def run(
        self,
        full_dataset: GlucoseDataset,
        glucose_raw: np.ndarray,
        timestamps: np.ndarray,
        normalizer: Any,
        patient_id: str,
    ) -> Dict[str, Any]:
        """Run the complete experiment."""
        print(f"\n{'=' * 60}")
        print(f"Running Continual Learning Experiment - Patient {patient_id}")
        print(f"EWC Lambda: {self.ewc_lambda}, Buffer: {self.replay_buffer_size}")
        print(f"{'=' * 60}")

        # Convert timestamps
        if isinstance(timestamps[0], (int, float, np.integer, np.floating)):
            timestamps = np.array([datetime.fromtimestamp(t) for t in timestamps])

        min_date = min(timestamps)
        max_date = max(timestamps)
        total_days = (max_date - min_date).days + 1

        print(f"Data spans {total_days} days: {min_date.date()} to {max_date.date()}")

        # Create adapter
        adapter = ContinualAdapter(
            model_class=self.model_class,
            model_config=self.model_config,
            training_config=self.training_config,
            ewc_lambda=self.ewc_lambda,
            replay_buffer_size=self.replay_buffer_size,
            replay_ratio=self.replay_ratio,
            update_interval_days=self.update_interval_days,
            device=self.device,
        )

        # Split by day
        days_data = self._split_by_day(full_dataset, timestamps, min_date, total_days)

        # Initial training
        initial_indices = []
        for day in range(min(self.initial_training_days, len(days_data))):
            if day in days_data:
                initial_indices.extend(days_data[day])

        if not initial_indices:
            raise ValueError("Not enough data for initial training")

        initial_dataset = Subset(full_dataset, initial_indices)
        # Get glucose values for the target (prediction) points
        # Each sample at index i predicts glucose at i + window_size + horizon
        initial_glucose_indices = np.array(initial_indices) + full_dataset.window_size
        # Clip to valid range
        initial_glucose_indices = np.clip(
            initial_glucose_indices, 0, len(glucose_raw) - 1
        )
        initial_glucose = glucose_raw[initial_glucose_indices]

        print(
            f"\nInitial training on {len(initial_indices)} samples ({self.initial_training_days} days)"
        )

        # Wrap Subset to work with our adapter
        class SubsetWrapper:
            def __init__(self, subset, window_size):
                self.subset = subset
                self.window_size = window_size

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                return self.subset[idx]

        wrapped_initial = SubsetWrapper(initial_dataset, full_dataset.window_size)
        adapter.initialize(wrapped_initial, initial_glucose)
        adapter.last_update_time = min_date + timedelta(days=self.initial_training_days)

        # Track results
        all_metrics = []
        update_events = []

        # Iterate through days
        for day in range(self.initial_training_days, total_days):
            if day not in days_data or len(days_data[day]) == 0:
                continue

            current_date = min_date + timedelta(days=day)
            days_since_update = (current_date - adapter.last_update_time).days

            # Check for update
            if adapter.should_update(current_date):
                # Get recent data for update
                window_start = max(0, day - 3)  # Last 3 days for update
                window_indices = []
                for d in range(window_start, day):
                    if d in days_data:
                        window_indices.extend(days_data[d])

                if window_indices:
                    window_dataset = Subset(full_dataset, window_indices)
                    window_glucose_indices = (
                        np.array(window_indices) + full_dataset.window_size
                    )
                    window_glucose_indices = np.clip(
                        window_glucose_indices, 0, len(glucose_raw) - 1
                    )
                    window_glucose = glucose_raw[window_glucose_indices]
                    wrapped_window = SubsetWrapper(
                        window_dataset, full_dataset.window_size
                    )

                    print(f"\nDay {day}: Updating on {len(window_indices)} samples")
                    adapter.update(wrapped_window, window_glucose, current_date)
                    days_since_update = 0
                    update_events.append(day)

            # Evaluate on current day
            day_indices = days_data[day]
            if len(day_indices) > 10:
                day_dataset = Subset(full_dataset, day_indices)
                wrapped_day = SubsetWrapper(day_dataset, full_dataset.window_size)
                metrics = adapter.evaluate(wrapped_day, normalizer)
                metrics["day"] = day
                metrics["days_since_update"] = days_since_update
                all_metrics.append(metrics)

                if day % 7 == 0:
                    print(
                        f"Day {day}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
                        f"Clarke A+B={metrics['clarke_ab']:.1f}%"
                    )

        # Compute summary
        results = self._compute_summary(all_metrics, update_events, patient_id, adapter)

        # Save results
        if self.results_dir:
            self._save_results(results, patient_id)

        return results

    def _split_by_day(self, dataset, timestamps, start_date, total_days):
        """Split dataset indices by day."""
        days_data = {d: [] for d in range(total_days)}

        for idx in range(len(dataset)):
            sample_time = timestamps[idx + dataset.window_size]
            day = (sample_time - start_date).days
            if 0 <= day < total_days:
                days_data[day].append(idx)

        return days_data

    def _compute_summary(self, all_metrics, update_events, patient_id, adapter):
        """Compute summary statistics."""
        if not all_metrics:
            return {"error": "No metrics collected"}

        mae_values = np.array([m["mae"] for m in all_metrics])
        rmse_values = np.array([m["rmse"] for m in all_metrics])
        clarke_ab = np.array([m["clarke_ab"] for m in all_metrics])

        return {
            "patient_id": patient_id,
            "strategy": "continual",
            "ewc_lambda": self.ewc_lambda,
            "replay_buffer_size": self.replay_buffer_size,
            "summary": {
                "mae_mean": float(np.mean(mae_values)),
                "mae_std": float(np.std(mae_values)),
                "rmse_mean": float(np.mean(rmse_values)),
                "rmse_std": float(np.std(rmse_values)),
                "clarke_ab_mean": float(np.mean(clarke_ab)),
                "clarke_ab_std": float(np.std(clarke_ab)),
                "total_days_evaluated": len(all_metrics),
                "total_updates": len(update_events),
            },
            "daily_metrics": all_metrics,
            "update_events": update_events,
            "adapter_history": adapter.get_history(),
        }

    def _save_results(self, results, patient_id):
        """Save results to file."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

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

        filepath = self.results_dir / f"continual_{patient_id}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
