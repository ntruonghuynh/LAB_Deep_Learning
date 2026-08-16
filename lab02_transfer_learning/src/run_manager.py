"""Run artifact management: safe run-id creation and saving config/history/checkpoints.

Ensures official experiment runs are never overwritten and stay traceable: each run's
config, description, metrics, plots, and checkpoint all live together under
runs/<model_name>/run_XXX/.
"""

import csv
import json
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def next_run_dir(model_name: str) -> Path:
    """Return the next unused run directory path: runs/<model_name>/run_XXX (not yet created)."""
    model_dir = RUNS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = []
    for child in model_dir.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                existing_ids.append(int(child.name.split("_")[1]))
            except (IndexError, ValueError):
                continue

    next_id = max(existing_ids, default=0) + 1
    return model_dir / f"run_{next_id:03d}"


def create_run(model_name: str, config: dict, description: str) -> Path:
    """Create a new official run directory containing config.yaml and description.md.

    Artifact subfolders (metrics/, plots/, checkpoints/, logs/) are created only when
    something is actually saved to them, not upfront.
    """
    run_dir = next_run_dir(model_name)
    run_dir.mkdir(parents=True, exist_ok=False)  # exist_ok=False: never silently reuse a run dir

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open(run_dir / "description.md", "w", encoding="utf-8") as f:
        f.write(description)

    return run_dir


def save_history(run_dir: Path, history: list[dict]) -> Path:
    """Save per-epoch training history as metrics/history.csv."""
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / "history.csv"
    fieldnames = list(history[0].keys())
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return history_path


def save_final_metrics(run_dir: Path, metrics: dict) -> Path:
    """Save summary metrics (best_epoch, best_val_loss, best_val_accuracy, test metrics, ...)."""
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path


def save_checkpoint(run_dir: Path, state_dict: dict, filename: str = "best_model.pt") -> Path:
    """Save a model's state_dict (not the whole model object) under checkpoints/."""
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / filename
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def append_description(run_dir: Path, extra_text: str) -> None:
    """Append observed result / interpretation / conclusion to an existing description.md."""
    with open(run_dir / "description.md", "a", encoding="utf-8") as f:
        f.write("\n" + extra_text)
