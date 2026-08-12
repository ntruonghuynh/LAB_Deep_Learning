"""Scan all official runs, rebuild experiments/experiment_summary.csv, and print a
comparison table across runs.

Only runs that have both config.yaml and metrics/final_metrics.json are included, so
debug/smoke-test executions (which never call run_manager.create_run) are naturally
excluded. The 'conclusion' column is intentionally left for the student to fill in after
reviewing results in notebooks/02_training_and_comparison.ipynb - it is not fabricated here.

Usage:
    python scripts/compare_models.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import yaml

RUNS_DIR = PROJECT_ROOT / "runs"
SUMMARY_PATH = PROJECT_ROOT / "experiments" / "experiment_summary.csv"


def collect_run_records() -> list[dict]:
    """Read config.yaml + metrics/final_metrics.json from every official run directory."""
    records = []
    if not RUNS_DIR.exists():
        return records

    for model_dir in sorted(RUNS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue

            config_path = run_dir / "config.yaml"
            final_metrics_path = run_dir / "metrics" / "final_metrics.json"
            if not config_path.exists() or not final_metrics_path.exists():
                continue  # incomplete or non-official run; skip

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            with open(final_metrics_path, "r", encoding="utf-8") as f:
                final_metrics = json.load(f)

            records.append({
                "run_id": f"{model_dir.name}/{run_dir.name}",
                "model": config["model"]["name"],
                "description": config["experiment"]["name"],
                "learning_rate": config["training"]["learning_rate"],
                "batch_size": config["training"]["batch_size"],
                "epochs": config["training"]["epochs"],
                "dropout": config["model"].get("dropout", ""),
                "augmentation": config["data"].get("augmentation", False),
                "best_epoch": final_metrics.get("best_epoch"),
                "best_val_loss": final_metrics.get("best_val_loss"),
                "best_val_accuracy": final_metrics.get("best_val_accuracy"),
                "test_accuracy_if_final": final_metrics.get("test_accuracy", ""),
                "conclusion": "",
            })
    return records


def merge_existing_conclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve manually-written 'conclusion' text from a previous summary, keyed by run_id.

    This lets a student write a conclusion once after reviewing a run, and keeps it even
    after compare_models.py is re-run to pick up new runs.
    """
    if not SUMMARY_PATH.exists():
        return df
    previous_df = pd.read_csv(SUMMARY_PATH, encoding="utf-8")
    if "conclusion" not in previous_df.columns:
        return df
    previous_conclusions = dict(zip(previous_df["run_id"], previous_df["conclusion"]))
    df["conclusion"] = df["run_id"].map(lambda run_id: previous_conclusions.get(run_id, ""))
    df["conclusion"] = df["conclusion"].fillna("")
    return df


def main() -> None:
    records = collect_run_records()
    if not records:
        print("Không tìm thấy lần chạy chính thức nào dưới runs/. Chưa có gì để so sánh.")
        return

    df = pd.DataFrame(records)
    df = merge_existing_conclusions(df)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")

    print(f"Đã cập nhật {SUMMARY_PATH} với {len(df)} lần chạy.\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
