
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml

from src import evaluation, visualization
from src.dataset import CLASS_NAMES, get_dataloaders, get_device
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run's best model on the official test set.")
    parser.add_argument("--run", type=str, required=True, help="Path to a run directory, e.g. runs/cnn/run_001")
    args = parser.parse_args()

    run_dir = Path(args.run)
    with open(run_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    seed = config["seed"]

    device = get_device()
    print(f"Sử dụng thiết bị: {device}")

    # Giống train.py, nhưng ở đây chỉ cần test_loader )
    # test_loader được sử dụng để đánh giá mô hình tốt nhất trên tập kiểm tra chính thức
    _train_loader, _val_loader, test_loader = get_dataloaders(
        batch_size=training_config["batch_size"],
        validation_ratio=data_config["validation_ratio"],
        augmentation=data_config.get("augmentation", False),
        seed=seed,
    )

    model = build_model(model_config["name"], model_config)
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Đã nạp checkpoint: {checkpoint_path}")

    predictions_data = evaluation.collect_predictions(model, test_loader, device)
    labels = predictions_data["labels"]
    predictions = predictions_data["predictions"]
    confidences = predictions_data["confidences"]

    test_accuracy = evaluation.overall_accuracy(labels, predictions)
    conf_matrix = evaluation.compute_confusion_matrix(labels, predictions, num_classes=len(CLASS_NAMES))
    per_class_df = evaluation.per_class_metrics(labels, predictions, CLASS_NAMES)
    misclassified_df = evaluation.build_misclassified_table(labels, predictions, confidences)

    print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    conf_matrix_df = pd.DataFrame(conf_matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    conf_matrix_df.to_csv(metrics_dir / "confusion_matrix.csv")
    per_class_df.to_csv(metrics_dir / "per_class_metrics.csv", index=False)
    misclassified_df.to_csv(metrics_dir / "misclassified.csv", index=False)

    conf_matrix_fig = visualization.plot_confusion_matrix(conf_matrix, CLASS_NAMES, title="Ma trận nhầm lẫn (Tập kiểm tra)")
    visualization.save_figure(conf_matrix_fig, run_dir / "plots" / "confusion_matrix.png")

    # Cập nhật final_metrics.json với độ chính xác trên tập kiểm tra
    final_metrics_path = metrics_dir / "final_metrics.json"
    final_metrics = {}
    if final_metrics_path.exists():
        with open(final_metrics_path, "r") as f:
            final_metrics = json.load(f)
    final_metrics["test_accuracy"] = test_accuracy
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Đã lưu các artifact đánh giá tại: {run_dir}")


if __name__ == "__main__":
    main()
