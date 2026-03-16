"""
AGNI Framework - Elastic Weight Consolidation (EWC)
Stage 3: Prevents catastrophic forgetting during continual learning

EWC adds a regularization term that penalizes changes to parameters
that are important for previously learned tasks.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """
    Elastic Weight Consolidation for continual learning.

    EWC prevents catastrophic forgetting by:
    1. Computing Fisher Information Matrix after learning a task
    2. Adding a penalty that discourages changes to important parameters

    The loss becomes: L_total = L_task + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    Where:
    - L_task: Current task loss
    - lambda: EWC strength hyperparameter
    - F_i: Fisher information for parameter i
    - theta_i: Current parameter value
    - theta*_i: Optimal parameter value from previous task
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        online: bool = True,
        gamma: float = 0.9,
    ):
        """
        Initialize EWC.

        Args:
            model: Neural network model
            ewc_lambda: Strength of EWC penalty (higher = more protection)
            online: Whether to use online EWC (accumulate Fisher across tasks)
            gamma: Decay factor for online EWC (0-1, higher = remember older tasks more)
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        # Store optimal parameters and Fisher information
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.fisher_info: Dict[str, torch.Tensor] = {}

        # Track number of consolidation events
        self.n_consolidations = 0

    def compute_fisher(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Fisher Information Matrix (diagonal approximation).

        The Fisher information measures how sensitive the loss is to each parameter.
        Parameters with high Fisher values are important and should be protected.

        Args:
            dataloader: DataLoader for computing Fisher
            criterion: Loss function
            device: Device to use
            n_samples: Number of samples to use (None = all)

        Returns:
            Dictionary mapping parameter names to Fisher values
        """
        fisher = {}

        # Initialize Fisher to zero
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        self.model.eval()
        n_total = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            if n_samples is not None and n_total >= n_samples:
                break

            x = x.to(device)
            y = y.to(device)

            self.model.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * x.size(0)

            n_total += x.size(0)

        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= n_total

        return fisher

    def consolidate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> None:
        """
        Consolidate current knowledge after learning a task.

        This computes the Fisher information and stores the current parameters
        as the "optimal" values to protect.

        Args:
            dataloader: DataLoader for computing Fisher
            criterion: Loss function
            device: Device to use
            n_samples: Number of samples for Fisher computation
        """
        # Compute Fisher information
        new_fisher = self.compute_fisher(dataloader, criterion, device, n_samples)

        if self.online and self.n_consolidations > 0:
            # Online EWC: Decay old Fisher and add new
            for name in new_fisher:
                if name in self.fisher_info:
                    self.fisher_info[name] = (
                        self.gamma * self.fisher_info[name] + new_fisher[name]
                    )
                else:
                    self.fisher_info[name] = new_fisher[name]
        else:
            # Standard EWC: Replace Fisher
            self.fisher_info = new_fisher

        # Store current parameters as optimal
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

        self.n_consolidations += 1

    def penalty(self) -> torch.Tensor:
        """
        Compute the EWC penalty term.

        Returns:
            Scalar tensor with the penalty value
        """
        if self.n_consolidations == 0:
            return torch.tensor(0.0)

        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                penalty += (
                    self.fisher_info[name] * (param - self.optimal_params[name]).pow(2)
                ).sum()

        return penalty

    def get_regularized_loss(self, task_loss: torch.Tensor) -> torch.Tensor:
        """
        Get the total loss including EWC penalty.

        Args:
            task_loss: Loss from the current task

        Returns:
            Total loss = task_loss + (ewc_lambda/2) * penalty
        """
        ewc_penalty = self.penalty()
        return task_loss + (self.ewc_lambda / 2) * ewc_penalty

    def get_importance_scores(self) -> Dict[str, float]:
        """
        Get overall importance scores for each parameter.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        scores = {}
        for name, fisher in self.fisher_info.items():
            scores[name] = fisher.mean().item()
        return scores

    def save_state(self) -> Dict:
        """Save EWC state for checkpointing."""
        return {
            "optimal_params": {k: v.cpu() for k, v in self.optimal_params.items()},
            "fisher_info": {k: v.cpu() for k, v in self.fisher_info.items()},
            "n_consolidations": self.n_consolidations,
            "ewc_lambda": self.ewc_lambda,
            "online": self.online,
            "gamma": self.gamma,
        }

    def load_state(self, state: Dict, device: torch.device) -> None:
        """Load EWC state from checkpoint."""
        self.optimal_params = {
            k: v.to(device) for k, v in state["optimal_params"].items()
        }
        self.fisher_info = {k: v.to(device) for k, v in state["fisher_info"].items()}
        self.n_consolidations = state["n_consolidations"]
        self.ewc_lambda = state["ewc_lambda"]
        self.online = state["online"]
        self.gamma = state["gamma"]


class OnlineEWC(EWC):
    """
    Online EWC variant that's more memory efficient for many tasks.

    Instead of storing Fisher for each task separately, online EWC
    maintains a running estimate with exponential decay.
    """

    def __init__(
        self, model: nn.Module, ewc_lambda: float = 1000.0, gamma: float = 0.9
    ):
        super().__init__(model, ewc_lambda, online=True, gamma=gamma)
