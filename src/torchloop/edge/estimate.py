"""Model size and latency estimation helpers for edge devices."""

from __future__ import annotations

from typing import Any, Dict, Literal

import torch

LATENCY_PER_MFLOP = {
    "esp32": 0.5,
    "rpi": 0.05,
    "jetson": 0.01,
    "desktop": 0.001,
}


def estimate_model(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    target_device: Literal["esp32", "rpi", "jetson", "desktop"] = "desktop",
) -> Dict[str, Any]:
    """Estimate model footprint and inferred latency for a target device.

    Args:
        model: PyTorch model to inspect.
        input_shape: Input tensor shape.
        target_device: Target device profile.

    Returns:
        Dictionary with params, size, RAM estimate, latency estimate, and flops.

    Raises:
        ValueError: If inputs are invalid.
    """
    if len(input_shape) < 2:
        raise ValueError("input_shape must contain at least batch and feature dims")
    if target_device not in LATENCY_PER_MFLOP:
        raise ValueError("target_device must be one of esp32, rpi, jetson, desktop")

    params = int(sum(p.numel() for p in model.parameters()))
    param_size_mb = (params * 4) / (1024 * 1024)
    activation_mb = _estimate_activation_mb(input_shape)
    estimated_ram_mb = param_size_mb + max(activation_mb, param_size_mb * 1.5)

    flops = _estimate_flops(model, input_shape)
    mflops = flops / 1_000_000.0
    latency = mflops * LATENCY_PER_MFLOP[target_device]

    return {
        "params": params,
        "param_size_mb": round(param_size_mb, 4),
        "estimated_ram_mb": round(estimated_ram_mb, 4),
        "estimated_latency_ms": round(latency, 4),
        "flops": int(flops),
    }


def _estimate_activation_mb(input_shape: tuple[int, ...]) -> float:
    """Approximate activation memory based on input tensor size."""
    numel = 1
    for dim in input_shape:
        numel *= dim
    return (numel * 4 * 2) / (1024 * 1024)


def _estimate_flops(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
) -> int:
    """Estimate FLOPs using hooks for common modules (Conv2d, Linear)."""
    flops = 0
    hooks = []

    def conv_hook(
        module: torch.nn.Conv2d,
        _inp: tuple[torch.Tensor],
        out: torch.Tensor,
    ) -> None:
        nonlocal flops
        batch = out.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels / module.groups)
        )
        flops += int(batch * out_h * out_w * module.out_channels * kernel_ops * 2)

    def linear_hook(
        module: torch.nn.Linear,
        inp: tuple[torch.Tensor],
        _out: torch.Tensor,
    ) -> None:
        nonlocal flops
        batch = inp[0].shape[0]
        flops += int(batch * module.in_features * module.out_features * 2)

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, torch.nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    model.eval()
    device = _get_model_device(model)
    with torch.no_grad():
        model(torch.randn(*input_shape, device=device))

    for handle in hooks:
        handle.remove()

    return flops


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Return model device, defaulting to CPU for parameterless modules."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
