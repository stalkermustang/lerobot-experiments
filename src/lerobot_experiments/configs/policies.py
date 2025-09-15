"""Custom policy configuration presets used in the SO101 project."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

__all__ = ["So101ACTConfig", "So101SmolVLAConfig"]


@PreTrainedConfig.register_subclass("so101_act")
@dataclass
class So101ACTConfig(ACTConfig):
    """ACT preset tuned for the SO101 experiments."""

    device: str | None = "cuda"
    chunk_size: int = 50
    n_action_steps: int = 25


@PreTrainedConfig.register_subclass("so101_smolvla")
@dataclass
class So101SmolVLAConfig(SmolVLAConfig):
    """SmolVLA preset tuned for the SO101 experiments."""

    device: str | None = "cuda"
    n_action_steps: int = 25
    optimizer_lr: float = 6e-4
    scheduler_warmup_steps: int = 200
    scheduler_decay_steps: int = 50_000
