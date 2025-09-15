"""CLI entry-points for lerobot-experiments."""

from .record import main as record_main, record
from .train import main as train_main, train

__all__ = [
    "record",
    "record_main",
    "train",
    "train_main",
]
