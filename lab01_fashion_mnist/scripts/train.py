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

import pandas as pd
import yaml

from src import run_manager, visualization
from src.dataset import get_dataloaders, get_device
from src.engine import train_model
from src.models import build_model
from src.seed import set_seed


def build_description(config: dict) -> str:
    """Build description.md content (purpose/hypothesis/etc.) from the run's config."""
    experiment = config["experiment"]
    lines = [
        f"# {experiment['name']}",
        "",
        "## Mục đích thí nghiệm",
        experiment.get("purpose", "").strip(),
        "",
        "## Giả thuyết",
        experiment.get("hypothesis", "").strip(),
        "",
        "## Thay đổi so với baseline",
        str(experiment.get("changed_from_baseline", "")).strip(),
        "",
        "## Những gì được giữ nguyên",
        str(experiment.get("kept_constant", "")).strip(),
        "",
        "## Dự đoán kết quả",
        experiment.get("expected_observation", "").strip(),
        "",
    ]
    return "\n".join(lines)


def build_result_section(history_df: pd.DataFrame, best_epoch: int, best_val_loss: float, best_val_accuracy: float) -> str:
    """Build the 'observed result / interpretation' section appended after training finishes."""
    final_train_acc = history_df.iloc[-1]["train_accuracy"]
    final_val_acc = history_df.iloc[-1]["val_accuracy"]
    gap = final_train_acc - final_val_acc

    lines = [
        "## Kết quả quan sát được",
        f"- Epoch tốt nhất: {best_epoch} (validation loss thấp nhất)",
        f"- Validation loss tốt nhất: {best_val_loss:.4f}",
        f"- Validation accuracy tốt nhất: {best_val_accuracy:.4f}",
        f"- Train accuracy ở epoch cuối: {final_train_acc:.4f}",
        f"- Validation accuracy ở epoch cuối: {final_val_acc:.4f}",
        f"- Khoảng cách train/validation accuracy cuối cùng: {gap:.4f}",
        "",
        "## Diễn giải",
        "Xem notebooks/02_training_and_comparison.ipynb để so sánh với các lần chạy khác.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FashionMNIST model from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    set_seed(seed)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    device = get_device()
    print(f"Sử dụng thiết bị: {device}")

    train_loader, val_loader, _test_loader = get_dataloaders(
        batch_size=training_config["batch_size"],
        validation_ratio=data_config["validation_ratio"],
        augmentation=data_config.get("augmentation", False),
        seed=seed,
    )

    model = build_model(model_config["name"], model_config)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        device=device,
        optimizer_name=training_config.get("optimizer", "adam"),
    )

    description = build_description(config)
    run_dir = run_manager.create_run(model_config["name"], config, description)
    print(f"Lần chạy được tạo tại: {run_dir}")

    run_manager.save_history(run_dir, result["history"])
    run_manager.save_checkpoint(run_dir, result["best_state_dict"])

    history_df = pd.DataFrame(result["history"])
    experiment_name = config["experiment"]["name"]

    loss_fig = visualization.plot_loss_curve(history_df, title=f"{experiment_name} - Mất mát")
    visualization.save_figure(loss_fig, run_dir / "plots" / "loss_curve.png")

    acc_fig = visualization.plot_accuracy_curve(history_df, title=f"{experiment_name} - Độ chính xác")
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
        f"Epoch tốt nhất: {result['best_epoch']} | "
        f"val_loss tốt nhất: {result['best_val_loss']:.4f} | "
        f"val_acc tốt nhất: {result['best_val_accuracy']:.4f}"
    )
    print(f"Artifact được lưu tại: {run_dir}")


if __name__ == "__main__":
    main()
