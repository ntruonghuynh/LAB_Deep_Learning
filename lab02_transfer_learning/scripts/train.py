"""Train a model from a YAML config and save a full official run.

Saves config.yaml, description.md, metrics/history.csv, metrics/final_metrics.json,
plots/loss_curve.png, plots/accuracy_curve.png, and checkpoints/best_model.pt under a
freshly created runs/<model>/run_XXX/ directory.

Usage:
    python scripts/train.py --config configs/mlp_baseline.yaml
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import yaml
from torch.utils.tensorboard import SummaryWriter

from src import run_manager, visualization
from src.dataset import get_dataloaders, get_device
from src.engine import train_model
from src.models import build_model
from src.seed import set_seed


def build_description(config: dict) -> str:
    """Build description.md content from the experiment section of a config."""
    experiment = config["experiment"]
    lines = [
        f"# {experiment['name']}",
        "",
        "## Purpose",
        experiment.get("purpose", "").strip(),
        "",
        "## Hypothesis",
        experiment.get("hypothesis", "").strip(),
        "",
        "## Changed from baseline",
        str(experiment.get("changed_from_baseline", "")).strip(),
        "",
        "## Kept constant",
        str(experiment.get("kept_constant", "")).strip(),
        "",
        "## Expected observation",
        experiment.get("expected_observation", "").strip(),
        "",
    ]
    return "\n".join(lines)


def build_result_section(history_df: pd.DataFrame, best_epoch: int, best_val_loss: float, best_val_accuracy: float) -> str:
    """Build the observed result section appended after training finishes."""
    final_train_acc = history_df.iloc[-1]["train_accuracy"]
    final_val_acc = history_df.iloc[-1]["val_accuracy"]
    gap = final_train_acc - final_val_acc

    lines = [
        "## Observed result",
        f"- Best epoch: {best_epoch} (lowest validation loss)",
        f"- Best validation loss: {best_val_loss:.4f}",
        f"- Best validation accuracy: {best_val_accuracy:.4f}",
        f"- Final train accuracy: {final_train_acc:.4f}",
        f"- Final validation accuracy: {final_val_acc:.4f}",
        f"- Final train/validation accuracy gap: {gap:.4f}",
        "",
        "## Interpretation",
        "See notebooks/02_training_and_comparison.ipynb for comparison with other runs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FashionMNIST model from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    set_seed(seed)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _test_loader = get_dataloaders(
        batch_size=training_config["batch_size"],
        validation_ratio=data_config["validation_ratio"],
        augmentation=data_config.get("augmentation", False),
        image_size=data_config.get("image_size", 28),
        input_channels=data_config.get("input_channels", 1),
        normalization=data_config.get("normalization", "fashion_mnist"),
        seed=seed,
    )

    model = build_model(model_config["name"], model_config, config.get("transfer_learning", {}))

    description = build_description(config)
    run_dir = run_manager.create_run(model_config["name"], config, description)
    print(f"Run directory created at: {run_dir}")

    tensorboard_config = config.get("tensorboard", {})
    writer = None
    if tensorboard_config.get("enabled", False):
        log_dir = run_dir / tensorboard_config.get("log_dir", "tensorboard")
        writer = SummaryWriter(log_dir=str(log_dir))

    try:
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config["epochs"],
            learning_rate=training_config["learning_rate"],
            device=device,
            optimizer_name=training_config.get("optimizer", "adam"),
            writer=writer,
        )
    finally:
        if writer is not None:
            writer.close()

    run_manager.save_history(run_dir, result["history"])
    run_manager.save_checkpoint(run_dir, result["best_state_dict"])

    history_df = pd.DataFrame(result["history"])
    experiment_name = config["experiment"]["name"]

    loss_fig = visualization.plot_loss_curve(history_df, title=f"{experiment_name} - Loss")
    visualization.save_figure(loss_fig, run_dir / "plots" / "loss_curve.png")

    acc_fig = visualization.plot_accuracy_curve(history_df, title=f"{experiment_name} - Accuracy")
    visualization.save_figure(acc_fig, run_dir / "plots" / "accuracy_curve.png")

    final_metrics = {
        "run_id": run_dir.name,
        "model": model_config["name"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "best_val_accuracy": result["best_val_accuracy"],
        "final_train_accuracy": float(history_df.iloc[-1]["train_accuracy"]),
        "final_val_accuracy": float(history_df.iloc[-1]["val_accuracy"]),
    }
    run_manager.save_final_metrics(run_dir, final_metrics)

    result_section = build_result_section(
        history_df, result["best_epoch"], result["best_val_loss"], result["best_val_accuracy"]
    )
    run_manager.append_description(run_dir, result_section)

    print(
        f"Best epoch: {result['best_epoch']} | "
        f"best val_loss: {result['best_val_loss']:.4f} | "
        f"best val_acc: {result['best_val_accuracy']:.4f}"
    )
    print(f"Artifacts saved at: {run_dir}")


if __name__ == "__main__":
    main()
