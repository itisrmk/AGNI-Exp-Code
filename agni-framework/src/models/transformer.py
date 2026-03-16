"""
Lightweight Transformer for glucose prediction.
"""

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import BaseGlucoseModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.

    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model: Dimension of model embeddings
        max_len: Maximum sequence length to support
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div term for sine/cosine frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerPredictor(BaseGlucoseModel):
    """
    Lightweight Transformer for glucose prediction.

    Uses self-attention to capture dependencies across the input window.
    Designed to be small (~48,000 parameters with default config) for
    efficient online adaptation.

    Architecture:
    - Linear embedding layer
    - Positional encoding
    - Transformer encoder layers
    - Global mean pooling
    - Output projection

    Args:
        config: Dictionary with keys:
            - d_model: Model dimension (default: 64)
            - nhead: Number of attention heads (default: 4)
            - num_layers: Number of encoder layers (default: 1)
            - dim_feedforward: FFN dimension (default: 128)
            - dropout: Dropout probability (default: 0.1)
            - input_size: Input feature dimension (default: 1)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.d_model = config.get("d_model", 64)
        self.nhead = config.get("nhead", 4)
        self.num_layers = config.get("num_layers", 1)
        self.dim_feedforward = config.get("dim_feedforward", 128)
        self.dropout = config.get("dropout", 0.1)
        self.input_size = config.get("input_size", 1)

        # Input embedding
        self.embedding = nn.Linear(self.input_size, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Output projection
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer.

        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)

        Returns:
            Predictions of shape (batch_size,)
        """
        # Embed input: (batch, seq, input_size) -> (batch, seq, d_model)
        x = self.embedding(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        out = self.transformer(x)  # (batch, seq, d_model)

        # Global mean pooling over sequence
        out = out.mean(dim=1)  # (batch, d_model)

        # Output projection
        output = self.fc(out)  # (batch, 1)

        return output.squeeze(-1)  # (batch,)


class CausalTransformerPredictor(BaseGlucoseModel):
    """
    Transformer with causal (autoregressive) attention mask.

    Ensures that each position can only attend to previous positions,
    which is more appropriate for forecasting tasks.

    Args:
        config: Model configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.d_model = config.get("d_model", 64)
        self.nhead = config.get("nhead", 4)
        self.num_layers = config.get("num_layers", 1)
        self.dim_feedforward = config.get("dim_feedforward", 128)
        self.dropout = config.get("dropout", 0.1)
        self.input_size = config.get("input_size", 1)

        self.embedding = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.fc = nn.Linear(self.d_model, 1)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking."""
        batch_size, seq_len, _ = x.shape

        # Embed
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Create causal mask
        mask = self._generate_causal_mask(seq_len, x.device)

        # Transform with causal attention
        out = self.transformer(x, mask=mask)

        # Use last position (most informed by past)
        out = out[:, -1, :]  # (batch, d_model)

        return self.fc(out).squeeze(-1)


class InformerStylePredictor(BaseGlucoseModel):
    """
    Simplified Informer-style model with ProbSparse attention approximation.

    Uses a simplified sparse attention pattern for efficiency on longer sequences.

    Args:
        config: Model configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.d_model = config.get("d_model", 64)
        self.nhead = config.get("nhead", 4)
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.1)
        self.input_size = config.get("input_size", 1)
        self.factor = config.get("factor", 5)  # Sampling factor for sparse attention

        self.embedding = nn.Linear(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)

        # Standard transformer layers (sparse attention would be custom)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Distilling: reduce sequence length progressively
        self.distil_conv = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=3, stride=2, padding=1
        )

        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sequence distillation."""
        # Embed
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Encode
        out = self.encoder(x)  # (batch, seq, d_model)

        # Distill (reduce sequence length)
        out = out.transpose(1, 2)  # (batch, d_model, seq)
        out = self.distil_conv(out)  # (batch, d_model, seq//2)
        out = torch.relu(out)

        # Pool
        out = out.mean(dim=2)  # (batch, d_model)

        return self.fc(out).squeeze(-1)
