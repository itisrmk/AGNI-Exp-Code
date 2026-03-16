"""
Configuration management utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Configuration to override with

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # General
    seed: int = 42
    device: str = "mps"

    # Data
    data_dir: str = "../OhioT1DM/archive"
    window_size: int = 24
    horizon: int = 6  # 30 minutes
    train_split: float = 0.5
    val_split: float = 0.2

    # Model
    model_type: str = "lstm"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Adaptation
    strategy: str = "static"  # static, periodic, continual

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "device": self.device,
            "data_dir": self.data_dir,
            "window_size": self.window_size,
            "horizon": self.horizon,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "model_type": self.model_type,
            "model_config": self.model_config,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "strategy": self.strategy,
        }
