"""
Model training utilities.
"""

import copy
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training when it stops improving.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score
            model: Model to save state from

        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = (
                self.best_score is None or score < self.best_score - self.min_delta
            )
        else:
            improved = (
                self.best_score is None or score > self.best_score + self.min_delta
            )

        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def load_best_state(self, model: nn.Module):
        """Load the best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    """
    Training loop with early stopping.

    Handles the standard train/validate loop with checkpointing.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        config: Training configuration
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        self.patience = config.get("early_stopping_patience", 10)
        self.early_stopping = EarlyStopping(patience=self.patience)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)

        return total_loss / n_samples

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)

        return total_loss / n_samples

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_epochs: Maximum number of epochs
            verbose: Whether to show progress bar

        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": []}

        iterator = (
            tqdm(range(max_epochs), desc="Training") if verbose else range(max_epochs)
        )

        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose:
                iterator.set_postfix(
                    {"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"}
                )

            # Check early stopping
            if self.early_stopping(val_loss, self.model):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Load best model
        self.early_stopping.load_best_state(self.model)

        return history

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Get predictions for a dataset.

        Args:
            data_loader: DataLoader to predict on

        Returns:
            Tensor of predictions
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.cpu())

        return torch.cat(predictions)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.

    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ("adam", "sgd", "adamw")
        learning_rate: Learning rate
        weight_decay: L2 regularization weight

    Returns:
        Optimizer instance
    """
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    optimizer_class = optimizers.get(optimizer_type.lower(), torch.optim.Adam)

    if optimizer_type.lower() == "sgd":
        return optimizer_class(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    return optimizer_class(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def create_criterion(loss_type: str = "mse") -> nn.Module:
    """
    Create loss function.

    Args:
        loss_type: Type of loss ("mse", "mae", "huber")

    Returns:
        Loss function module
    """
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
    }

    return losses.get(loss_type.lower(), nn.MSELoss)()
