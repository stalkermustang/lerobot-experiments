"""Utilities for running lightweight inference loops with LeRobot policies."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch


def run_model_inference(
    policy: Any, observation: Mapping[str, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Run a single forward pass of ``policy`` on ``observation``.

    Parameters
    ----------
    policy:
        A pretrained LeRobot policy returned by :func:`lerobot.policies.factory.make_policy`.
    observation:
        Observation dictionary produced by :class:`lerobot.datasets.lerobot_dataset.LeRobotDataset`.
    device:
        Target torch device for inference (retrieved via ``lerobot.utils.utils.get_safe_torch_device``).

    Returns
    -------
    torch.Tensor
        The chunk of actions predicted by the policy on CPU.
    """
    policy.eval()

    use_amp = policy.config.use_amp
    batch: dict[str, torch.Tensor] = {}
    for name, value in observation.items():
        if name.startswith("observation"):
            batch[name] = value.unsqueeze(0).to(device, non_blocking=True)
        elif name == "task":
            batch[name] = value

    amp_context = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        actions = policy.predict_action_chunk(batch)
    return actions.to("cpu")
