"""Training pipeline configuration defaults used by the experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.train import TrainPipelineConfig as BaseTrainPipelineConfig

from .default import WandBConfig

__all__ = ["TrainPipelineConfig"]


@dataclass
class TrainPipelineConfig(BaseTrainPipelineConfig):
    """Override the training defaults (batch size, steps, logging)."""

    num_workers: int = 16
    batch_size: int = 32
    steps: int = 50_000
    save_freq: int = 10_000
    wandb: WandBConfig = field(default_factory=WandBConfig)
