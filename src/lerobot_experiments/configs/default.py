"""Configuration overrides for the SO101 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.default import (
    DatasetConfig,
    EvalConfig,
    WandBConfig as BaseWandBConfig,
)

__all__ = ["DatasetConfig", "EvalConfig", "WandBConfig"]


@dataclass
class WandBConfig(BaseWandBConfig):
    """Enable Weights & Biases logging with the project defaults used in the experiments."""

    enable: bool = True
    project: str = "lerobot_so101"
    notes: str | None = "P16"
