# Lab 3 Hugging Face Rules

## Core Workflow Rules

### R1: Exercise Structure
**PRINCIPLE**: Each exercise must have clear separation between exploration, training, and analysis.

**WHY**: Exercise 1 focuses on understanding pre-trained models; Exercise 2 focuses on finetuning. Mixing them creates confusion.

**APPLY**:
- Exercise 1: Use pipelines, demonstrate tokenization, explain outputs
- Exercise 2: Load dataset → split → train → evaluate → analyze errors
- Keep exercises in separate notebooks

---

### R2: Dataset Split Immutability
**PRINCIPLE**: Train/validation/test splits must be created once and saved to JSON files immediately.

**WHY**: Reproducibility requires identical data splits across all experiments.

**APPLY**:
```python
from src.data.dataset_utils import save_splits, load_splits
from sklearn.model_selection import train_test_split

# Create splits once
train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=42, stratify=labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

# Save immediately
save_splits(train_idx, val_idx, test_idx, output_dir='splits/')

# Always load from saved files
train_idx, val_idx, test_idx = load_splits('splits/')
```

---

### R3: Training Monitoring
**PRINCIPLE**: Training loss and metrics must be tracked every epoch and visualized.

**WHY**: Monitoring reveals overfitting, underfitting, and convergence issues.

**APPLY**:
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./outputs/exercise2',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# After training, plot curves
from src.utils.visualization import plot_training_history
plot_training_history(trainer.state.log_history, save_path='outputs/exercise2/training_curves.png')
```

---

### R4: Artifact Preservation
**PRINCIPLE**: Save training history as CSV, best model, and last checkpoint.

**WHY**: Raw data enables future analysis without re-training.

**APPLY**:
```python
import pandas as pd
import json

# Save training history
history_df = pd.DataFrame(trainer.state.log_history)
history_df.to_csv('outputs/exercise2/training_history.csv', index=False)

# Save as JSON for easy loading
with open('outputs/exercise2/training_history.json', 'w') as f:
    json.dump(trainer.state.log_history, f, indent=2)

# Models are automatically saved by Trainer to:
# - outputs/exercise2/checkpoint-{step}/  (intermediate checkpoints)
# - outputs/exercise2/  (final model if save_strategy="no" at end)
```

---

### R5: Confusion Matrix Analysis
**PRINCIPLE**: Every classification task must include confusion matrix with detailed explanation.

**WHY**: Understanding error patterns reveals model weaknesses and guides improvements.

**APPLY**:
```python
from src.evaluation.eval_metrics import compute_confusion_matrix, explain_confusion_matrix
from src.utils.visualization import plot_confusion_matrix

# Get predictions
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids

# Compute and plot
cm = compute_confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, class_names=['Negative', 'Positive'], 
                     save_path='outputs/exercise2/confusion_matrix.png')

# Explain what it means
explain_confusion_matrix(cm, class_names=['Negative', 'Positive'])
```

**Confusion Matrix Structure**:
```
              Predicted
              Neg    Pos
Actual  Neg  [ TN     FP ]
        Pos  [ FN     TP ]
```

- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Wrongly predicted positive (Type I error)
- **FN (False Negative)**: Wrongly predicted negative (Type II error)
- **TP (True Positive)**: Correctly predicted positive

**Diagonal** = Correct predictions  
**Off-diagonal** = Errors  
**Good model** = High values on diagonal, low elsewhere

---

### R6: Comprehensive Metrics
**PRINCIPLE**: Report accuracy, precision, recall, F1-score for each class.

**WHY**: Accuracy alone is misleading with imbalanced datasets.

**APPLY**:
```python
from src.evaluation.eval_metrics import compute_classification_metrics

metrics = compute_classification_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")

# Save metrics
import json
with open('outputs/exercise2/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

### R7: Error Analysis
**PRINCIPLE**: Identify and analyze misclassified examples to understand failure patterns.

**WHY**: Error analysis reveals what the model struggles with and suggests improvements.

**APPLY**:
```python
from src.evaluation.error_analysis import analyze_errors, show_error_examples

# Get misclassified indices
errors = analyze_errors(y_true, y_pred, test_dataset)

# Show examples
show_error_examples(errors, test_dataset, tokenizer, num_examples=10)

# Document patterns in experiment log:
# - Length bias? (short vs long texts)
# - Sentiment confusion? (sarcasm, negation)
# - Domain-specific issues?
```

---

### R8: Tokenization Explanation
**PRINCIPLE**: Exercise 1 must demonstrate and explain tokenization process.

**WHY**: Understanding tokenization is fundamental to working with transformers.

**APPLY**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sentence = "I love this product! It's amazing."

# Show tokenization steps
print("Original:", sentence)
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", ids)

# Show full encoding
encoded = tokenizer(sentence, return_tensors="pt")
print("Encoded:", encoded)
print("Decoded:", tokenizer.decode(encoded['input_ids'][0]))
```

---

### R9: Model Selection Documentation
**PRINCIPLE**: Document why specific models and hyperparameters were chosen.

**WHY**: Decisions should be justified and traceable.

**APPLY** in experiment log:
```markdown
## Model Selection
- **Model**: distilbert-base-uncased
- **Why**: Smaller and faster than BERT, good for binary classification
- **Alternatives considered**: bert-base-uncased (slower), roberta-base (better but heavier)

## Hyperparameters
- **Learning rate**: 2e-5 (standard for BERT-family models)
- **Batch size**: 16 (limited by GPU memory)
- **Epochs**: 3 (more risks overfitting on small dataset)
- **Warmup steps**: 500 (gradual learning rate increase)
```

---

### R10: Experiment Logging
**PRINCIPLE**: Every training run must be documented in the experiment log.

**WHY**: Tracks what was tried, what worked, what failed.

**APPLY**:
```markdown
## Run 1
**Date**: 2026-08-18
**Config**: distilbert-base-uncased, lr=2e-5, batch=16, epochs=3
**Results**: Acc=0.87, F1=0.86
**Issues**: Some negation handling errors
**Next**: Try increasing epochs to 5

## Run 2
**Date**: 2026-08-18
**Config**: distilbert-base-uncased, lr=2e-5, batch=16, epochs=5
**Results**: Acc=0.89, F1=0.88, but validation loss increased after epoch 3
**Conclusion**: 3 epochs is optimal, overfitting after that
```

---

## Anti-Patterns

### ❌ Don't: Use random splits every time
```python
# BAD - splits change every run
train_test_split(dataset, test_size=0.2)
```

### ✅ Do: Save and load splits
```python
# GOOD - reproducible splits
save_splits(train_idx, val_idx, test_idx, 'splits/')
train_idx, val_idx, test_idx = load_splits('splits/')
```

---

### ❌ Don't: Only report accuracy
```python
# BAD - incomplete metrics
print(f"Accuracy: {accuracy}")
```

### ✅ Do: Report comprehensive metrics
```python
# GOOD - full picture
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
```

---

### ❌ Don't: Skip error analysis
```python
# BAD - no understanding of failures
print("Model trained! Accuracy: 0.85")
```

### ✅ Do: Analyze errors
```python
# GOOD - understand failures
errors = analyze_errors(y_true, y_pred, test_dataset)
print(f"Found {len(errors)} errors")
show_error_examples(errors, test_dataset, tokenizer, num_examples=10)
# Document patterns in experiment log
```

---

## Completion Checklist

### Exercise 1
- [ ] Installed transformers library
- [ ] Loaded sentiment analysis pipeline
- [ ] Demonstrated tokenization with examples
- [ ] Analyzed multiple sentences with different sentiments
- [ ] Explained model outputs (labels, scores)
- [ ] Interpreted results in context

### Exercise 2
- [ ] Installed transformers, datasets, evaluate libraries
- [ ] Loaded dataset and explored statistics
- [ ] Created stratified train/val/test splits
- [ ] Saved splits to JSON files
- [ ] Tokenized dataset with proper preprocessing
- [ ] Defined TrainingArguments with monitoring
- [ ] Trained model with Trainer API
- [ ] Saved training history as CSV and JSON
- [ ] Evaluated on test set
- [ ] Generated confusion matrix with explanation
- [ ] Computed comprehensive metrics (accuracy, precision, recall, F1)
- [ ] Performed error analysis
- [ ] Documented all runs in experiment log
- [ ] Saved all artifacts to outputs/ directory

---

## File Organization

```
Lab-3/
├── notebooks/
│   ├── 01_exercise1_sentiment_analysis.ipynb
│   ├── 02_exercise2_data_exploration.ipynb
│   ├── 03_exercise2_training.ipynb
│   └── 04_exercise2_evaluation.ipynb
│
├── splits/
│   ├── train_indices.json
│   ├── val_indices.json
│   └── test_indices.json
│
├── outputs/
│   ├── exercise1/
│   │   └── examples.txt
│   └── exercise2/
│       ├── training_history.csv
│       ├── training_history.json
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       ├── metrics.json
│       ├── checkpoint-best/
│       └── checkpoint-last/
│
└── experiments/
    └── Exercise2_Finetuning_Log.md
```

---

## Quick Reference

### Load Pre-trained Model
```python
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")
result = sentiment_analyzer("I love this!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Tokenize Text
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encoded = tokenizer("Hello world", padding=True, truncation=True, return_tensors="pt")
```

### Train with Trainer
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./outputs',
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

### Evaluate Model
```python
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids

from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred))
```

---

*Follow these rules to ensure reproducibility, thorough analysis, and clear documentation of your work.*
