"""
LSTM model for glucose prediction.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseGlucoseModel


class LSTMPredictor(BaseGlucoseModel):
    """
    Stacked LSTM for glucose prediction.

    Architecture:
    - Multi-layer LSTM for sequence processing
    - Fully connected layer for prediction

    Default config yields ~53,000 parameters.

    Args:
        config: Dictionary with keys:
            - hidden_size: LSTM hidden dimension (default: 64)
            - num_layers: Number of LSTM layers (default: 2)
            - dropout: Dropout probability (default: 0.2)
            - input_size: Input feature dimension (default: 1)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.input_size = config.get("input_size", 1)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

        # Output layer
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)

        Returns:
            Predictions of shape (batch_size,)
        """
        # LSTM forward
        # lstm_out: (batch_size, window_size, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        # c_n: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state from final layer
        last_hidden = h_n[-1]  # (batch_size, hidden_size)

        # Predict
        output = self.fc(last_hidden)  # (batch_size, 1)

        return output.squeeze(-1)  # (batch_size,)


class BidirectionalLSTMPredictor(BaseGlucoseModel):
    """
    Bidirectional LSTM for glucose prediction.

    Uses both forward and backward context for prediction.

    Args:
        config: Dictionary with model configuration
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.input_size = config.get("input_size", 1)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True,
        )

        # 2x hidden size due to bidirectional
        self.fc = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bidirectional LSTM."""
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate final hidden states from both directions
        # h_n shape: (num_layers * 2, batch_size, hidden_size)
        forward_hidden = h_n[-2]  # Last layer forward
        backward_hidden = h_n[-1]  # Last layer backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        output = self.fc(combined)
        return output.squeeze(-1)


class AttentionLSTMPredictor(BaseGlucoseModel):
    """
    LSTM with attention mechanism for glucose prediction.

    Applies attention over LSTM outputs to focus on relevant time steps.

    Args:
        config: Dictionary with model configuration
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.input_size = config.get("input_size", 1)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1),
        )

        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention weights
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)

        output = self.fc(context)
        return output.squeeze(-1)
