# Lab 3: Hugging Face Transformers for NLP - Project Overview

## 🎯 What You'll Learn

This lab teaches you to:
1. **Use pre-trained models** - Leverage Hugging Face's model hub
2. **Understand tokenization** - See how text becomes model input
3. **Finetune transformers** - Adapt pre-trained models to your task
4. **Evaluate properly** - Go beyond accuracy to understand model behavior
5. **Analyze errors** - Learn what your model struggles with

## 📋 Exercises

### Exercise 1: Sentiment Analysis (30 min)
**Goal**: Understand pre-trained models and tokenization

**What you'll do**:
- Load a sentiment analysis pipeline
- Analyze various text samples
- Examine tokenization step-by-step
- Understand model outputs

**Key concepts**: Pipelines, tokenization, confidence scores

### Exercise 2: Finetuning (2-3 hours)
**Goal**: Train a model for binary text classification

**What you'll do**:
- Load and explore a dataset
- Create reproducible train/val/test splits
- Finetune a BERT-family model
- Monitor training with loss/metrics curves
- Evaluate with comprehensive metrics
- Analyze errors and patterns
- Document everything

**Key concepts**: Transfer learning, training monitoring, confusion matrix, error analysis

## 🗂️ Project Structure

```
Lab-3/
├── README.md                          # Quick start guide
├── LAB3_HUGGINGFACE_RULES.md         # Detailed workflow rules
├── QUICK_REFERENCE.md                # Code snippets cheat sheet
├── requirements.txt                   # Python dependencies
│
├── src/                              # Utility modules
│   ├── data/                         # Data handling
│   │   └── dataset_utils.py         # Splits, tokenization, stats
│   ├── models/                       # Model utilities
│   │   └── model_utils.py           # Loading, tokenization demo
│   ├── training/                     # Training config
│   │   └── trainer_utils.py         # TrainingArguments, metrics
│   ├── evaluation/                   # Metrics & analysis
│   │   ├── eval_metrics.py          # Accuracy, P, R, F1, CM
│   │   └── error_analysis.py        # Error patterns
│   └── utils/                        # Visualization
│       └── visualization.py         # Plots for everything
│
├── notebooks/                        # Your work goes here
│   ├── 01_exercise1_sentiment_analysis.ipynb
│   ├── 02_exercise2_data_exploration.ipynb
│   ├── 03_exercise2_training.ipynb
│   ├── 04_exercise2_evaluation.ipynb
│   └── 05_exercise2_error_analysis.ipynb
│
├── experiments/                      # Experiment logs
│   └── Exercise2_Finetuning_Log.md  # Document all training runs
│
├── splits/                           # Saved train/val/test splits
│   ├── train_indices.json           # Created once, reused forever
│   ├── val_indices.json
│   └── test_indices.json
│
└── outputs/                          # Training artifacts
    └── exercise2/
        ├── training_history.csv      # Loss & metrics per epoch
        ├── training_curves.png       # Visualization
        ├── confusion_matrix.png      # CM heatmap
        ├── metrics.json              # Test results
        ├── error_analysis.json       # Error patterns
        └── checkpoint-best/          # Best model weights
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Exercise 1: Quick Test
```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
result = sentiment("I love Hugging Face!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 3. Exercise 2: Full Pipeline
```python
# Import utilities
from src.data import create_stratified_splits, save_splits, load_splits
from src.models import load_model_and_tokenizer
from src.training import create_training_arguments
from src.evaluation import compute_confusion_matrix, analyze_errors
from src.utils import plot_training_history

# Load dataset (example: IMDB)
from datasets import load_dataset
dataset = load_dataset("imdb")

# Create and save splits (do this ONCE)
train_idx, val_idx, test_idx = create_stratified_splits(
    dataset['train'], 
    train_size=0.7, 
    val_size=0.15, 
    test_size=0.15
)
save_splits(train_idx, val_idx, test_idx, output_dir='splits/')

# From now on, always load (never recreate)
train_idx, val_idx, test_idx = load_splits('splits/')

# Load model
model, tokenizer = load_model_and_tokenizer(
    "distilbert-base-uncased", 
    num_labels=2
)

# Tokenize
def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

tokenized_train = dataset['train'].select(train_idx).map(tokenize_fn, batched=True)
tokenized_val = dataset['train'].select(val_idx).map(tokenize_fn, batched=True)
tokenized_test = dataset['train'].select(test_idx).map(tokenize_fn, batched=True)

# Train
from transformers import Trainer
training_args = create_training_arguments(output_dir='outputs/exercise2')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)
trainer.train()

# Evaluate
predictions = trainer.predict(tokenized_test)
# ... compute metrics, plot confusion matrix, analyze errors
```

## 📊 What You'll Produce

### Artifacts
- ✅ Training history (CSV/JSON)
- ✅ Training curves (loss & metrics plots)
- ✅ Confusion matrix (heatmap with explanation)
- ✅ Comprehensive metrics (accuracy, precision, recall, F1)
- ✅ Error analysis (patterns, examples)
- ✅ Best model checkpoint

### Documentation
- ✅ Experiment log with all runs
- ✅ Configuration for each run
- ✅ Results and comparisons
- ✅ Error patterns identified
- ✅ Conclusions and recommendations

## 🎓 Key Concepts

### Confusion Matrix
```
              Predicted
           Negative  Positive
Actual Neg [  TN       FP   ]  ← Type I error (False alarm)
       Pos [  FN       TP   ]  ← Type II error (Miss)
```
- **Diagonal (TN + TP)**: Correct predictions ✓
- **Off-diagonal (FP + FN)**: Errors ✗
- **Goal**: Maximize diagonal, minimize off-diagonal

### Metrics Explained
- **Accuracy**: (TP + TN) / Total - Overall correctness
- **Precision**: TP / (TP + FP) - "Of predicted positives, how many are correct?"
- **Recall**: TP / (TP + FN) - "Of actual positives, how many did we find?"
- **F1-Score**: 2 × (P × R) / (P + R) - Harmonic mean of precision and recall

### Why Not Just Accuracy?
Example: 95% negative samples, always predict negative → 95% accuracy but useless model!

## 🔧 Utilities You'll Use

### Data Management
```python
# Create splits once
create_stratified_splits(dataset, train_size=0.7, val_size=0.15, test_size=0.15)

# Save for reproducibility
save_splits(train_idx, val_idx, test_idx, output_dir='splits/')

# Always load (never recreate)
train_idx, val_idx, test_idx = load_splits('splits/')

# Dataset statistics
stats = get_dataset_statistics(dataset)
print_dataset_statistics(stats)
```

### Model & Tokenization
```python
# Load pre-trained sentiment model
pipeline = load_sentiment_pipeline()

# Demonstrate tokenization
demonstrate_tokenization("I love NLP!", tokenizer)

# Load for finetuning
model, tokenizer = load_model_and_tokenizer("distilbert-base-uncased", num_labels=2)
```

### Training
```python
# Configure training
training_args = create_training_arguments(
    output_dir='outputs/exercise2',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)

# Create metrics function
compute_metrics = create_compute_metrics_fn()

# Save training history
save_training_history(trainer.state.log_history, 'outputs/exercise2')
```

### Evaluation
```python
# Compute metrics
metrics = compute_classification_metrics(predictions, labels)

# Confusion matrix
cm = compute_confusion_matrix(labels, predictions)
explain_confusion_matrix(cm, class_names=['Negative', 'Positive'])

# Error analysis
errors = analyze_errors(dataset, predictions, labels)
patterns = analyze_error_patterns(errors)
```

### Visualization
```python
# Training curves
plot_training_history('outputs/exercise2/training_history.csv')

# Confusion matrix heatmap
plot_confusion_matrix(cm, class_names=['Negative', 'Positive'])

# Error distributions
plot_error_distribution(error_patterns)
```

## ✅ Success Criteria

### Exercise 1
- [ ] Successfully run sentiment analysis pipeline
- [ ] Understand tokenization process
- [ ] Interpret model outputs and confidence

### Exercise 2
- [ ] Create and save reproducible splits
- [ ] Train model with monitoring (loss + metrics per epoch)
- [ ] Achieve reasonable performance (~85%+ accuracy depends on dataset)
- [ ] Generate confusion matrix with explanation
- [ ] Compute all metrics (accuracy, precision, recall, F1)
- [ ] Identify error patterns
- [ ] Document everything in experiment log

## 🚫 Common Mistakes to Avoid

1. **Creating new splits every run** → Always load from saved JSON
2. **Reporting only accuracy** → Always include precision, recall, F1
3. **Skipping error analysis** → Understanding failures is critical
4. **Not saving artifacts** → Save everything (history, models, metrics)
5. **No documentation** → Document all runs in experiment log
6. **Ignoring confusion matrix** → Must generate and explain it

## 💡 Tips for Success

1. **Start small**: Test pipeline on subset before full training
2. **Monitor closely**: Watch for overfitting (val loss increasing)
3. **Save everything**: Training history, models, metrics
4. **Understand errors**: Error analysis teaches more than high accuracy
5. **Document as you go**: Don't wait until the end
6. **Use the utilities**: All helper functions are in `src/`

## 📚 Resources

### Documentation
- **README.md** - Quick start and overview
- **LAB3_HUGGINGFACE_RULES.md** - Detailed workflow rules
- **QUICK_REFERENCE.md** - Code snippets and cheat sheet
- **notebooks/README.md** - Notebook-specific guidance

### External Links
- [Hugging Face Docs](https://huggingface.co/docs/transformers/)
- [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/en/training)
- [Sentiment Analysis Guide](https://huggingface.co/blog/sentiment-analysis-python)

## 🆘 Getting Help

1. Check **QUICK_REFERENCE.md** for code snippets
2. Review **LAB3_HUGGINGFACE_RULES.md** for workflow
3. Look at utility functions in `src/` for examples
4. Check **notebooks/README.md** for notebook-specific issues

## 🎉 Final Deliverables

When you're done, you should have:

1. **5 notebooks** (1 for exercise 1, 4 for exercise 2)
2. **Saved splits** in `splits/` directory
3. **Training artifacts** in `outputs/exercise2/`
4. **Completed experiment log** in `experiments/`
5. **Best model checkpoint** ready for deployment

---

**Remember**: This lab is about understanding transformers and proper evaluation, not just achieving high accuracy. Focus on the process, monitor carefully, and analyze thoroughly.

Good luck! 🚀
