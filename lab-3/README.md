# Lab 3: Hugging Face Transformers for NLP

## Overview

This lab introduces **Hugging Face Transformers** for Natural Language Processing (NLP) tasks, focusing on sentiment analysis and text classification using pre-trained models.

### Objectives
- Understand how to use pre-trained transformer models
- Learn tokenization and text preprocessing
- Finetune a pre-trained model for binary text classification
- Evaluate model performance with comprehensive metrics
- Analyze model errors and understand failure patterns

---

## Lab Structure

### Exercise 1: Sentiment Analysis with Pre-trained Models
Use a ready-made sentiment analysis pipeline from Hugging Face Hub to:
- Analyze sentiment of text samples
- Understand tokenization process
- Interpret model outputs and confidence scores

**Time**: ~30 minutes

### Exercise 2: Finetuning for Binary Text Classification
Finetune a pre-trained BERT-family model on a custom dataset:
- Load and explore dataset
- Create reproducible train/val/test splits
- Tokenize and prepare data
- Train with monitoring
- Evaluate with comprehensive metrics
- Analyze errors and model behavior

**Time**: ~2 hours

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Exercise 1: Quick Test

```python
from transformers import pipeline

# Load sentiment analyzer
sentiment = pipeline("sentiment-analysis")

# Test it
result = sentiment("I love using Hugging Face!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Exercise 2: Training Pipeline

```python
from src.data.dataset_utils import create_stratified_splits, save_splits
from src.models.model_utils import load_model_and_tokenizer
from src.training.trainer_utils import create_training_arguments
from transformers import Trainer

# 1. Create and save splits (once)
train_idx, val_idx, test_idx = create_stratified_splits(dataset)
save_splits(train_idx, val_idx, test_idx)

# 2. Load model
model, tokenizer = load_model_and_tokenizer("distilbert-base-uncased", num_labels=2)

# 3. Train
training_args = create_training_arguments(output_dir='./outputs/exercise2')
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()

# 4. Evaluate
predictions = trainer.predict(test_ds)
# ... analyze results
```

---

## Key Rules

### Must Follow
1. **Save splits immediately** - Create train/val/test splits once and save to JSON
2. **Monitor training** - Track loss and metrics every epoch
3. **Save everything** - Training history, models, metrics as CSV/JSON
4. **Confusion matrix** - Always generate and explain
5. **Error analysis** - Identify and understand misclassified examples
6. **Document** - Log all experiments with configs, results, and conclusions

### Never
- Create new random splits - always load from saved JSON files
- Report only accuracy - always include precision, recall, F1
- Skip error analysis - must understand what the model struggles with

---

## Documentation

- **LAB3_HUGGINGFACE_RULES.md** - Detailed workflow rules and best practices
- **QUICK_REFERENCE.md** - Code snippets and cheat sheet
- **experiments/** - Experiment logs (document all training runs here)

---

## Utilities

All utility functions are in `src/`:

### Data (`src/data/`)
- `create_stratified_splits()` - Create balanced splits
- `save_splits()` / `load_splits()` - Save/load split indices
- `get_dataset_statistics()` - Compute dataset statistics
- `prepare_dataset_for_training()` - Tokenize and format for Trainer

### Models (`src/models/`)
- `load_model_and_tokenizer()` - Load pre-trained model
- `demonstrate_tokenization()` - Show tokenization step-by-step
- `print_model_info()` - Display model architecture and parameters

### Training (`src/training/`)
- `create_training_arguments()` - Setup training configuration
- `create_compute_metrics_fn()` - Define metrics to track
- `save_training_history()` - Save training logs

### Evaluation (`src/evaluation/`)
- `compute_classification_metrics()` - Calculate all metrics
- `compute_confusion_matrix()` - Generate confusion matrix
- `explain_confusion_matrix()` - Explain matrix structure
- `analyze_errors()` - Find misclassified examples
- `analyze_error_patterns()` - Identify common failure patterns

### Visualization (`src/utils/`)
- `plot_training_history()` - Plot loss and metrics curves
- `plot_confusion_matrix()` - Heatmap visualization
- `plot_error_distribution()` - Error patterns by class
- `plot_text_length_distribution()` - Length analysis
- `plot_confidence_distribution()` - Confidence analysis

---

## Output Structure

After completing exercises, you'll have:

```
outputs/exercise2/
├── training_history.csv          # Raw training data
├── training_history.json         # Same in JSON
├── training_curves.png           # Loss/metrics plots
├── confusion_matrix.png          # CM heatmap
├── metrics.json                  # Test metrics
├── error_analysis.json           # Error patterns
├── hyperparameters.json          # Config
├── checkpoint-best/              # Best model
└── checkpoint-last/              # Last checkpoint
```

---

## Understanding Confusion Matrix

For binary classification:

```
              Predicted
           Negative  Positive
Actual Neg [  TN       FP   ]  ← Type I error
       Pos [  FN       TP   ]  ← Type II error
```

- **Diagonal (TN + TP)**: Correct predictions ✓
- **Off-diagonal (FP + FN)**: Errors ✗

**Goal**: Maximize diagonal, minimize off-diagonal

---

## Common Issues

### GPU Memory Error
```python
# Reduce batch size
per_device_train_batch_size = 8  # Instead of 16
gradient_accumulation_steps = 2  # Simulate larger batch
```

### Slow Training
```python
# Use smaller model
model_name = "distilbert-base-uncased"  # Instead of bert-base
```

### Not Reproducible
```python
# Always load saved splits
train_idx, val_idx, test_idx = load_splits('splits/')  # NOT create_stratified_splits()
```

---

## Expected Results

### Exercise 1
- Successfully use sentiment analysis pipeline
- Understand tokenization process
- Interpret model outputs

### Exercise 2
- Train a model with ~85-90% accuracy (depends on dataset)
- Generate comprehensive evaluation metrics
- Identify error patterns (e.g., struggles with negation, sarcasm)
- Document findings in experiment log

---

## References

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/)
- [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model)
- [Sentiment Analysis Guide](https://huggingface.co/blog/sentiment-analysis-python)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

---

## Tips

1. **Start small**: Test pipeline with a subset of data first
2. **Monitor closely**: Watch training curves for overfitting
3. **Analyze errors**: Understanding failures is more valuable than high accuracy
4. **Document everything**: Future you will thank present you
5. **Save often**: Training history, models, metrics - save everything

---

## Getting Help

- Check **QUICK_REFERENCE.md** for code snippets
- Review **LAB3_HUGGINGFACE_RULES.md** for workflow rules
- Look at utility functions in `src/` for examples
- Check experiment logs in `experiments/` for guidance

---

**Remember**: This lab is about understanding transformers, not just achieving high accuracy. Focus on the process, monitor carefully, and analyze thoroughly.

Good luck! 🚀
