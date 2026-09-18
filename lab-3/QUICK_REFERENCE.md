# Lab 3 Hugging Face Quick Reference

## Pipeline Overview

### Exercise 1: Sentiment Analysis
```
Install → Load Pipeline → Tokenize → Analyze → Interpret
```

### Exercise 2: Finetuning
```
Load Data → Explore → Split → Save → Tokenize → Train → Evaluate → Analyze
```

---

## Core Rules (Must Follow)

### ✅ ALWAYS
1. **Save splits immediately** after creating them
2. **Monitor training** every epoch (loss + metrics)
3. **Save training history** as CSV and JSON
4. **Generate confusion matrix** with explanation
5. **Perform error analysis** on misclassified examples
6. **Document everything** in experiment log

### ❌ NEVER
1. Create new splits - always load from saved JSON
2. Report only accuracy - always include precision, recall, F1
3. Skip error analysis - must understand failures
4. Save only plots - always save raw metrics as CSV/JSON

---

## Essential Code Snippets

### 1. Create and Save Splits
```python
from src.data.dataset_utils import create_stratified_splits, save_splits, load_splits

# Create splits ONCE
train_idx, val_idx, test_idx = create_stratified_splits(
    dataset, 
    train_size=0.7, 
    val_size=0.15, 
    test_size=0.15,
    random_state=42
)

# Save immediately
save_splits(train_idx, val_idx, test_idx, output_dir='splits/')

# Always load from saved files
train_idx, val_idx, test_idx = load_splits('splits/')
```

### 2. Load Model and Tokenizer
```python
from src.models.model_utils import load_model_and_tokenizer, print_model_info

model, tokenizer = load_model_and_tokenizer(
    model_name="distilbert-base-uncased",
    num_labels=2
)

print_model_info(model, tokenizer)
```

### 3. Tokenize Dataset
```python
from src.data.dataset_utils import prepare_dataset_for_training

train_dataset = prepare_dataset_for_training(
    train_subset,
    tokenizer,
    text_key='text',
    label_key='label',
    max_length=512
)
```

### 4. Setup Training
```python
from src.training.trainer_utils import create_training_arguments, create_compute_metrics_fn
from transformers import Trainer

# Create training arguments
training_args = create_training_arguments(
    output_dir='./outputs/exercise2',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    eval_steps=None,  # Evaluate every epoch
    save_total_limit=2,
    load_best_model_at_end=True
)

# Create compute metrics function
compute_metrics = create_compute_metrics_fn(['accuracy', 'f1', 'precision', 'recall'])

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)
```

### 5. Train and Save History
```python
from src.training.trainer_utils import save_training_history, print_training_summary

# Train
trainer.train()

# Save training history
save_training_history(trainer, output_dir='./outputs/exercise2')

# Print summary
print_training_summary(trainer)
```

### 6. Evaluate and Get Predictions
```python
# Evaluate on test set
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids
y_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
```

### 7. Compute Metrics
```python
from src.evaluation.eval_metrics import (
    compute_classification_metrics,
    print_classification_report,
    save_metrics
)

# Compute metrics
metrics = compute_classification_metrics(y_true, y_pred)

# Print detailed report
print_classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])

# Save metrics
save_metrics(metrics, output_path='./outputs/exercise2/metrics.json')
```

### 8. Confusion Matrix
```python
from src.evaluation.eval_metrics import compute_confusion_matrix, explain_confusion_matrix
from src.utils.visualization import plot_confusion_matrix

# Compute
cm = compute_confusion_matrix(y_true, y_pred)

# Plot
plot_confusion_matrix(
    cm, 
    class_names=['Negative', 'Positive'],
    save_path='./outputs/exercise2/confusion_matrix.png'
)

# Explain
explain_confusion_matrix(cm, class_names=['Negative', 'Positive'])
```

### 9. Error Analysis
```python
from src.evaluation.error_analysis import (
    analyze_errors,
    show_error_examples,
    analyze_error_patterns,
    print_error_patterns
)

# Get errors
errors = analyze_errors(y_true, y_pred, test_dataset, text_key='text')

# Show examples
show_error_examples(errors, num_examples=10, class_names=['Negative', 'Positive'])

# Analyze patterns
patterns = analyze_error_patterns(errors)
print_error_patterns(patterns, class_names=['Negative', 'Positive'])
```

### 10. Plot Training Curves
```python
from src.utils.visualization import plot_training_history

plot_training_history(
    trainer.state.log_history,
    save_path='./outputs/exercise2/training_curves.png'
)
```

---

## Exercise 1 Checklist

- [ ] Install transformers library
- [ ] Load sentiment analysis pipeline
- [ ] Test on multiple sentences (positive, negative, neutral)
- [ ] Demonstrate tokenization step-by-step
- [ ] Show token IDs and decoded output
- [ ] Explain model outputs (labels, scores)
- [ ] Interpret confidence scores

### Example Code for Exercise 1
```python
from transformers import pipeline, AutoTokenizer

# Load pipeline
sentiment = pipeline("sentiment-analysis")

# Test sentences
sentences = [
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
]

for text in sentences:
    result = sentiment(text)
    print(f"Text: {text}")
    print(f"Result: {result}\n")

# Demonstrate tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text = "I love this product!"

print("Original:", text)
print("Tokens:", tokenizer.tokenize(text))
print("Token IDs:", tokenizer.encode(text))
print("Decoded:", tokenizer.decode(tokenizer.encode(text)))
```

---

## Exercise 2 Checklist

### Data Phase
- [ ] Load dataset with `datasets` library
- [ ] Print dataset statistics (size, label distribution, text lengths)
- [ ] Show random examples
- [ ] Create stratified train/val/test splits
- [ ] Save splits to JSON files
- [ ] Verify splits can be loaded

### Training Phase
- [ ] Load pre-trained model and tokenizer
- [ ] Tokenize and prepare datasets
- [ ] Define TrainingArguments with monitoring
- [ ] Define compute_metrics function
- [ ] Create Trainer
- [ ] Log hyperparameters
- [ ] Train model
- [ ] Save training history as CSV and JSON
- [ ] Save best and last checkpoints

### Evaluation Phase
- [ ] Load best model
- [ ] Predict on test set
- [ ] Compute accuracy, precision, recall, F1
- [ ] Generate confusion matrix
- [ ] Plot confusion matrix
- [ ] Explain confusion matrix structure
- [ ] Print classification report
- [ ] Save all metrics as JSON

### Analysis Phase
- [ ] Identify misclassified examples
- [ ] Show error examples with true/predicted labels
- [ ] Analyze error patterns (confusion pairs, text lengths)
- [ ] Compare text lengths: correct vs incorrect
- [ ] Analyze prediction confidence
- [ ] Find high-confidence errors
- [ ] Document findings in experiment log

### Visualization Phase
- [ ] Plot training and validation loss curves
- [ ] Plot evaluation metrics over epochs
- [ ] Plot confusion matrix heatmap
- [ ] Plot error distribution by confusion pair
- [ ] Plot text length distribution
- [ ] Plot confidence distribution

---

## Common Hyperparameters

### Recommended for BERT-family models
```python
learning_rate = 2e-5  # Standard for BERT
num_train_epochs = 3  # 3-5 epochs typical
per_device_train_batch_size = 16  # Adjust based on GPU memory
warmup_steps = 500  # Gradual LR warmup
weight_decay = 0.01  # L2 regularization
```

### If GPU memory issues
```python
per_device_train_batch_size = 8  # Reduce batch size
gradient_accumulation_steps = 2  # Simulate larger batch
fp16 = True  # Use mixed precision (if GPU supports)
```

---

## File Structure

```
outputs/exercise2/
├── training_history.csv          # Training metrics per step/epoch
├── training_history.json         # Same in JSON format
├── training_curves.png           # Loss and metrics plots
├── confusion_matrix.png          # Confusion matrix heatmap
├── metrics.json                  # Final test metrics
├── error_analysis.json           # Error patterns and examples
├── hyperparameters.json          # Training configuration
├── checkpoint-best/              # Best model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── checkpoint-last/              # Last checkpoint
```

---

## Confusion Matrix Guide

### Binary Classification
```
              Predicted
           Neg      Pos
Actual Neg [ TN      FP ]
       Pos [ FN      TP ]
```

- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Wrongly predicted positive (Type I error)
- **FN (False Negative)**: Wrongly predicted negative (Type II error)  
- **TP (True Positive)**: Correctly predicted positive

**Good model**: High TN and TP (diagonal), low FP and FN (off-diagonal)

---

## Metrics Explained

```python
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)  # Of predicted positive, how many are correct?
Recall    = TP / (TP + FN)  # Of actual positive, how many did we find?
F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)  # Harmonic mean
```

**When to focus on**:
- **Precision**: When false positives are costly (e.g., spam detection)
- **Recall**: When false negatives are costly (e.g., disease detection)
- **F1**: When you need a balance

---

## Troubleshooting

### Training is slow
- Check GPU: `torch.cuda.is_available()`
- Reduce batch size
- Use smaller model (distilbert instead of bert-base)

### Out of memory
- Reduce `per_device_train_batch_size`
- Reduce `max_length` in tokenization
- Enable `fp16=True` in TrainingArguments

### Model not improving
- Check learning rate (try 2e-5, 3e-5, 5e-5)
- Check data preprocessing
- Verify labels are correct
- Try more epochs (but watch for overfitting)

### Results not reproducible
- Ensure you're loading the same splits from JSON
- Check random seeds in splits and training

---

## Quick Commands

### Check installations
```python
import transformers, datasets, torch
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

### Load saved model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./outputs/exercise2/checkpoint-best')
tokenizer = AutoTokenizer.from_pretrained('./outputs/exercise2/checkpoint-best')
```

### Quick test
```python
text = "This is a great product!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
print(f"Prediction: {prediction}")
```

---

**Remember**: Follow the rules, document everything, and analyze errors thoroughly!
