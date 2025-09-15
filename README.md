# LeRobot Experiments

Custom utilities and configuration presets used for SO101 experiments on top of the upstream [`lerobot`](https://github.com/huggingface/lerobot) library.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

This pulls the published `lerobot` package as a dependency.

## CLI entry points

- `lerobot-experiments-train` – wraps the stock training loop while applying SO101-specific defaults.
- `lerobot-experiments-record` – records teleop datasets and toggles torque when no policy is provided.
- `lerobot-experiments-remove-episodes` – utility to prune and reindex recorded datasets.

## Offline policy visualisation

The notebook `offline_policy_visualisation.ipynb` demonstrates how the helpers in this package are used to compare policy rollouts against recorded episodes.

## License

This project continues to use the Apache 2.0 license distributed with the original repository (see `LICENSE`).
