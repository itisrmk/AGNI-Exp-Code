"""
Base model class for glucose prediction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseGlucoseModel(nn.Module, ABC):
    """
    Abstract base class for glucose prediction models.

    All models should:
    - Accept (batch_size, window_size, 1) input
    - Return (batch_size,) output (single prediction per sample)

    Args:
        config: Model configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)

        Returns:
            Output tensor of shape (batch_size,)
        """
        pass

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information summary."""
        return {
            "name": self.__class__.__name__,
            "parameters": self.count_parameters(),
            "config": self.config,
        }


def create_model(model_type: str, config: Dict[str, Any]) -> BaseGlucoseModel:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('lstm', 'tcn', 'transformer')
        config: Model configuration

    Returns:
        Instantiated model
    """
    from .lstm import LSTMPredictor
    from .tcn import TCNPredictor
    from .transformer import TransformerPredictor

    models = {
        "lstm": LSTMPredictor,
        "tcn": TCNPredictor,
        "transformer": TransformerPredictor,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )

    return models[model_type](config)
