"""
Temporal Convolutional Network for glucose prediction.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .base import BaseGlucoseModel


class CausalConv1d(nn.Module):
    """
    Causal convolution with proper padding.

    Ensures that convolution output at time t only depends on inputs at times <= t.
    Uses left padding to maintain causality.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        dilation: Dilation factor for dilated convolutions
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ):
        super().__init__()
        # Padding needed for causal convolution
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of same seq_len due to causal padding
        """
        out = self.conv(x)
        # Remove the future padding to maintain causality
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Residual block for Temporal Convolutional Network.

    Structure:
    - CausalConv -> BatchNorm -> ReLU -> Dropout
    - CausalConv -> BatchNorm -> ReLU -> Dropout
    - Residual connection (with 1x1 conv if channel mismatch)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # First convolution
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second convolution
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        # First conv block
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        # Second conv block
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        # Residual
        return self.relu(out + self.residual(x))


class TCNPredictor(BaseGlucoseModel):
    """
    Temporal Convolutional Network for glucose prediction.

    Uses dilated causal convolutions to capture long-range dependencies
    while maintaining causality.

    Default config yields ~45,000 parameters.

    Architecture:
    - Stack of TCN blocks with exponentially increasing dilation
    - Global average pooling
    - Fully connected output layer

    Args:
        config: Dictionary with keys:
            - num_channels: List of channel sizes per block (default: [32, 32, 32, 32])
            - kernel_size: Convolution kernel size (default: 3)
            - dropout: Dropout probability (default: 0.2)
            - input_size: Input feature dimension (default: 1)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_channels = config.get("num_channels", [32, 32, 32, 32])
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout = config.get("dropout", 0.2)
        self.input_size = config.get("input_size", 1)

        # Build TCN layers
        layers = []
        in_channels = self.input_size

        for i, out_channels in enumerate(self.num_channels):
            # Exponentially increasing dilation: 1, 2, 4, 8, ...
            dilation = 2**i
            layers.append(
                TCNBlock(
                    in_channels, out_channels, self.kernel_size, dilation, self.dropout
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)

        # Output layer
        self.fc = nn.Linear(self.num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)

        Returns:
            Predictions of shape (batch_size,)
        """
        # Transpose for Conv1d: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)

        # TCN layers
        out = self.network(x)  # (batch, channels, seq_len)

        # Global average pooling over time dimension
        out = out.mean(dim=2)  # (batch, channels)

        # Output
        output = self.fc(out)  # (batch, 1)

        return output.squeeze(-1)  # (batch,)

    def receptive_field(self) -> int:
        """
        Calculate the receptive field of the TCN.

        Returns:
            Number of input time steps that influence each output
        """
        # Receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        dilations = [2**i for i in range(len(self.num_channels))]
        return 1 + 2 * (self.kernel_size - 1) * sum(dilations)


class TCNPredictorV2(BaseGlucoseModel):
    """
    Enhanced TCN with weight normalization and gated activations.

    Uses gated linear units (GLU) instead of ReLU for potentially
    better gradient flow.

    Args:
        config: Model configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_channels = config.get("num_channels", [32, 32, 32, 32])
        self.kernel_size = config.get("kernel_size", 3)
        self.dropout = config.get("dropout", 0.2)
        self.input_size = config.get("input_size", 1)

        layers = []
        in_channels = self.input_size

        for i, out_channels in enumerate(self.num_channels):
            dilation = 2**i
            layers.append(
                self._make_block(in_channels, out_channels, self.kernel_size, dilation)
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(self.num_channels[-1], 1)

    def _make_block(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> nn.Module:
        """Create a TCN block with weight normalization."""
        padding = (kernel_size - 1) * dilation

        return nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            nn.GLU(dim=1),  # Gated Linear Unit
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(1, 2)

        for layer in self.network:
            # Apply layer
            out = layer(x)
            # Trim to match input length (causal padding)
            out = out[:, :, : x.size(2)]
            # Residual (if dimensions match)
            if out.size(1) == x.size(1):
                out = out + x
            x = out

        out = x.mean(dim=2)
        return self.fc(out).squeeze(-1)
