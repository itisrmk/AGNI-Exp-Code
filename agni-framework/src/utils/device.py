"""
Device management for Mac (MPS) and CPU.
"""

from typing import Optional

import torch


def get_device(preferred: str = "mps") -> torch.device:
    """
    Get the best available device.

    Priority: MPS (Apple Silicon) > CPU

    Args:
        preferred: Preferred device ("mps", "cuda", "cpu")

    Returns:
        torch.device object for the selected device
    """
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get information about available devices.

    Returns:
        Dictionary with device availability information
    """
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_available": True,
    }

    if info["mps_available"]:
        info["mps_built"] = torch.backends.mps.is_built()

    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info


def move_to_device(obj, device: torch.device):
    """
    Move tensor or model to device.

    Args:
        obj: Tensor or Module to move
        device: Target device

    Returns:
        Object on target device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    return obj
