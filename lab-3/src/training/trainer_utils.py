"""
Training utilities for Hugging Face models.

Functions for creating training arguments and monitoring training.
"""

from typing import Dict, Any, Optional, Callable
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import os
import json
import pandas as pd


def create_training_arguments(
    output_dir: str = './outputs/exercise2',
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    logging_steps: int = 10,
    eval_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "accuracy",
    **kwargs
) -> TrainingArguments:
    """
    Create TrainingArguments with sensible defaults.

    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps (None = every epoch)
        save_steps: Save checkpoint every N steps (None = every epoch)
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Whether to load best model at end
        metric_for_best_model: Metric to use for best model selection
        **kwargs: Additional arguments to pass to TrainingArguments

    Returns:
        TrainingArguments object
    """
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=logging_steps,
        evaluation_strategy="epoch" if eval_steps is None else "steps",
        eval_steps=eval_steps,
        save_strategy="epoch" if save_steps is None else "steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        **kwargs
    )

    print(f"✓ Training arguments created")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Train batch size: {per_device_train_batch_size}")
    print(f"  Eval batch size: {per_device_eval_batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Evaluation: every {'epoch' if eval_steps is None else f'{eval_steps} steps'}")

    return training_args


def create_compute_metrics_fn(metric_names: list = ['accuracy', 'f1', 'precision', 'recall']) -> Callable:
    """
    Create a compute_metrics function for the Trainer.

    Args:
        metric_names: List of metric names to compute

    Returns:
        compute_metrics function
    """
    import evaluate

    # Load metrics
    metrics = {name: evaluate.load(name) for name in metric_names}

    def compute_metrics(eval_pred):
        """Compute metrics for evaluation."""
        logits, labels = eval_pred
        predictions = logits.argmax(-1)

        results = {}
        for name, metric in metrics.items():
            if name in ['f1', 'precision', 'recall']:
                # For binary/multiclass classification
                result = metric.compute(predictions=predictions, references=labels, average='weighted')
            else:
                result = metric.compute(predictions=predictions, references=labels)

            results.update(result)

        return results

    return compute_metrics


def save_training_history(
    trainer: Trainer,
    output_dir: str = './outputs/exercise2'
) -> None:
    """
    Save training history to CSV and JSON files.

    Args:
        trainer: Trained Trainer object
        output_dir: Directory to save history files
    """
    os.makedirs(output_dir, exist_ok=True)

    history = trainer.state.log_history

    # Save as JSON
    json_path = os.path.join(output_dir, 'training_history.json')
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save as CSV
    csv_path = os.path.join(output_dir, 'training_history.csv')
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)

    print(f"✓ Training history saved")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def print_training_summary(trainer: Trainer) -> None:
    """
    Print a summary of training results.

    Args:
        trainer: Trained Trainer object
    """
    history = trainer.state.log_history

    # Extract final metrics
    eval_results = [log for log in history if 'eval_loss' in log]

    if not eval_results:
        print("No evaluation results found")
        return

    final_eval = eval_results[-1]

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Total epochs: {trainer.state.epoch}")
    print(f"Total steps: {trainer.state.global_step}")

    print(f"\nFinal evaluation metrics:")
    for key, value in final_eval.items():
        if key != 'epoch' and key != 'step':
            print(f"  {key}: {value:.4f}")

    # Find best checkpoint
    if hasattr(trainer.state, 'best_model_checkpoint'):
        print(f"\nBest checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best metric value: {trainer.state.best_metric:.4f}")

    print(f"{'='*60}\n")


def monitor_training_progress(
    trainer: Trainer,
    checkpoint_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Monitor training progress and extract key metrics.

    Args:
        trainer: Trainer object
        checkpoint_dir: Optional directory to check for checkpoints

    Returns:
        Dictionary of training progress information
    """
    history = trainer.state.log_history

    # Separate train and eval logs
    train_logs = [log for log in history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in history if 'eval_loss' in log]

    progress = {
        'total_steps': trainer.state.global_step,
        'epoch': trainer.state.epoch,
        'train_losses': [log.get('loss') for log in train_logs if 'loss' in log],
        'eval_losses': [log.get('eval_loss') for log in eval_logs],
        'eval_metrics': {}
    }

    # Extract evaluation metrics
    if eval_logs:
        for key in eval_logs[-1].keys():
            if key not in ['epoch', 'step', 'eval_loss']:
                progress['eval_metrics'][key] = [log.get(key) for log in eval_logs]

    # Check for checkpoints
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        progress['num_checkpoints'] = len(checkpoints)
        progress['checkpoints'] = sorted(checkpoints)

    return progress


def create_trainer_with_callbacks(
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    compute_metrics: Callable,
    tokenizer,
    early_stopping_patience: Optional[int] = None,
    early_stopping_threshold: float = 0.0
) -> Trainer:
    """
    Create a Trainer with optional callbacks.

    Args:
        model: Model to train
        training_args: TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        compute_metrics: Function to compute metrics
        tokenizer: Tokenizer
        early_stopping_patience: Patience for early stopping (None = disabled)
        early_stopping_threshold: Minimum improvement threshold

    Returns:
        Trainer object with callbacks
    """
    callbacks = []

    if early_stopping_patience is not None:
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )
        callbacks.append(early_stop)
        print(f"✓ Early stopping enabled (patience={early_stopping_patience})")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    return trainer


def log_hyperparameters(
    training_args: TrainingArguments,
    model_name: str,
    output_file: str = './outputs/exercise2/hyperparameters.json'
) -> None:
    """
    Log all hyperparameters to a JSON file.

    Args:
        training_args: TrainingArguments object
        model_name: Name of the model
        output_file: Path to save hyperparameters
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    hyperparams = {
        'model_name': model_name,
        'num_train_epochs': training_args.num_train_epochs,
        'per_device_train_batch_size': training_args.per_device_train_batch_size,
        'per_device_eval_batch_size': training_args.per_device_eval_batch_size,
        'learning_rate': training_args.learning_rate,
        'weight_decay': training_args.weight_decay,
        'warmup_steps': training_args.warmup_steps,
        'evaluation_strategy': training_args.evaluation_strategy,
        'save_strategy': training_args.save_strategy,
        'save_total_limit': training_args.save_total_limit,
        'metric_for_best_model': training_args.metric_for_best_model,
    }

    with open(output_file, 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print(f"✓ Hyperparameters saved to {output_file}")
