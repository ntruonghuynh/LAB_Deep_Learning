# Getting Started with Lab 3

This is a step-by-step guide to get you up and running quickly.

## 🚀 Setup (5 minutes)

### Step 1: Verify Python Installation
```bash
python --version  # Should be Python 3.8+
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Expected packages:
- `transformers` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library
- `torch` - PyTorch (deep learning framework)
- `scikit-learn` - For metrics and evaluation
- `matplotlib`, `seaborn` - For visualization
- `pandas`, `numpy` - Data manipulation
- `jupyter` - For notebooks

### Step 3: Verify Installation
```bash
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import datasets; print('datasets:', datasets.__version__)"
```

### Step 4: Start Jupyter
```bash
jupyter notebook notebooks/
```

## 📝 Exercise 1: Sentiment Analysis (30 minutes)

### Goal
Understand how to use pre-trained models from Hugging Face.

### What to Do

Create notebook: `notebooks/01_exercise1_sentiment_analysis.ipynb`

```python
# Cell 1: Setup
from transformers import pipeline
from src.models import load_sentiment_pipeline, demonstrate_tokenization

# Cell 2: Load pre-trained sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis")

# Cell 3: Test on simple examples
texts = [
    "I love Hugging Face!",
    "This is terrible.",
    "Not bad, but could be better.",
    "Absolutely amazing experience!",
]

for text in texts:
    result = sentiment_analyzer(text)
    print(f"Text: {text}")
    print(f"Result: {result}\n")

# Cell 4: Understand tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

text = "I love Hugging Face!"
demonstrate_tokenization(text, tokenizer)

# Cell 5: Analyze output structure
result = sentiment_analyzer("I love this!")[0]
print(f"Label: {result['label']}")
print(f"Confidence: {result['score']:.4f}")
```

### Expected Output
- Sentiment predictions with confidence scores
- Understanding of tokenization process (tokens, IDs, attention masks)
- Grasp of model output structure

### Deliverable
✅ One notebook showing pre-trained model usage and tokenization

---

## 🎓 Exercise 2: Finetuning (2-3 hours)

Split into 4 notebooks for clarity.

### Part 1: Data Exploration (30 minutes)

**Notebook**: `02_exercise2_data_exploration.ipynb`

```python
# Cell 1: Setup
import sys
sys.path.append('..')

from datasets import load_dataset
from src.data import (
    create_stratified_splits, 
    save_splits, 
    get_dataset_statistics,
    print_dataset_statistics,
    show_examples
)

# Cell 2: Load dataset
dataset = load_dataset("imdb")  # or another binary classification dataset
print(f"Dataset: {dataset}")
print(f"Features: {dataset['train'].features}")

# Cell 3: Show examples
show_examples(dataset['train'], n=3)

# Cell 4: Compute statistics
stats = get_dataset_statistics(dataset['train'])
print_dataset_statistics(stats)

# Cell 5: Create splits (DO THIS ONCE ONLY!)
train_idx, val_idx, test_idx = create_stratified_splits(
    dataset['train'],
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42
)

print(f"Train samples: {len(train_idx)}")
print(f"Val samples: {len(val_idx)}")
print(f"Test samples: {len(test_idx)}")

# Cell 6: Save splits
save_splits(train_idx, val_idx, test_idx, output_dir='../splits')
print("✓ Splits saved to ../splits/")
```

**Deliverable**: ✅ Splits saved to `splits/*.json`

---

### Part 2: Training (1 hour)

**Notebook**: `03_exercise2_training.ipynb`

```python
# Cell 1: Setup
import sys
sys.path.append('..')

from datasets import load_dataset
from transformers import Trainer
from src.data import load_splits, prepare_dataset_for_training
from src.models import load_model_and_tokenizer
from src.training import create_training_arguments, create_compute_metrics_fn

# Cell 2: Load dataset and splits
dataset = load_dataset("imdb")
train_idx, val_idx, test_idx = load_splits('../splits')
print(f"Loaded splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

# Cell 3: Load model and tokenizer
model_name = "distilbert-base-uncased"
model, tokenizer = load_model_and_tokenizer(model_name, num_labels=2)

# Cell 4: Prepare datasets
train_dataset = prepare_dataset_for_training(
    dataset['train'].select(train_idx), 
    tokenizer, 
    text_column='text',
    label_column='label'
)

val_dataset = prepare_dataset_for_training(
    dataset['train'].select(val_idx), 
    tokenizer, 
    text_column='text',
    label_column='label'
)

# Cell 5: Configure training
training_args = create_training_arguments(
    output_dir='../outputs/exercise2',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Cell 6: Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=create_compute_metrics_fn()
)

# Cell 7: Train! (this will take time)
print("Starting training...")
trainer.train()
print("✓ Training complete!")

# Cell 8: Save model
trainer.save_model('../outputs/exercise2/checkpoint-best')
print("✓ Model saved to ../outputs/exercise2/checkpoint-best")
```

**Deliverable**: ✅ Trained model and training history in `outputs/exercise2/`

---

### Part 3: Evaluation (30 minutes)

**Notebook**: `04_exercise2_evaluation.ipynb`

```python
# Cell 1: Setup
import sys
sys.path.append('..')

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data import load_splits, prepare_dataset_for_training
from src.evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix,
    explain_confusion_matrix,
    save_metrics
)
from src.utils import plot_training_history, plot_confusion_matrix

# Cell 2: Load model and data
model = AutoModelForSequenceClassification.from_pretrained('../outputs/exercise2/checkpoint-best')
tokenizer = AutoTokenizer.from_pretrained('../outputs/exercise2/checkpoint-best')

dataset = load_dataset("imdb")
train_idx, val_idx, test_idx = load_splits('../splits')

test_dataset = prepare_dataset_for_training(
    dataset['train'].select(test_idx),
    tokenizer,
    text_column='text',
    label_column='label'
)

# Cell 3: Get predictions
from transformers import Trainer, TrainingArguments
trainer = Trainer(model=model, args=TrainingArguments(output_dir='../outputs/temp'))
predictions = trainer.predict(test_dataset)

pred_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

# Cell 4: Compute metrics
metrics = compute_classification_metrics(pred_labels, true_labels)
print("Test Metrics:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1-Score:  {metrics['f1']:.4f}")

# Cell 5: Confusion matrix
cm = compute_confusion_matrix(true_labels, pred_labels)
print("\nConfusion Matrix:")
print(cm)

# Cell 6: Explain confusion matrix
class_names = ['Negative', 'Positive']
explain_confusion_matrix(cm, class_names)

# Cell 7: Plot confusion matrix
plot_confusion_matrix(cm, class_names, save_path='../outputs/exercise2/confusion_matrix.png')

# Cell 8: Plot training history
plot_training_history('../outputs/exercise2/training_history.csv')

# Cell 9: Save metrics
save_metrics(metrics, '../outputs/exercise2/metrics.json')
print("✓ Metrics saved")
```

**Deliverable**: ✅ Metrics, confusion matrix, and plots in `outputs/exercise2/`

---

### Part 4: Error Analysis (30 minutes)

**Notebook**: `05_exercise2_error_analysis.ipynb`

```python
# Cell 1: Setup
import sys
sys.path.append('..')

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data import load_splits
from src.evaluation import (
    analyze_errors,
    show_error_examples,
    analyze_error_patterns,
    print_error_patterns,
    save_error_analysis
)

# Cell 2: Load everything
model = AutoModelForSequenceClassification.from_pretrained('../outputs/exercise2/checkpoint-best')
tokenizer = AutoTokenizer.from_pretrained('../outputs/exercise2/checkpoint-best')
dataset = load_dataset("imdb")
train_idx, val_idx, test_idx = load_splits('../splits')

# Get test data
test_data = dataset['train'].select(test_idx)

# Cell 3: Get predictions with confidence
from transformers import Trainer, TrainingArguments
trainer = Trainer(model=model, args=TrainingArguments(output_dir='../outputs/temp'))

# Tokenize test data
def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

test_tokenized = test_data.map(tokenize_fn, batched=True)
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

predictions = trainer.predict(test_tokenized)
pred_labels = predictions.predictions.argmax(-1)
pred_probs = np.exp(predictions.predictions) / np.exp(predictions.predictions).sum(-1, keepdims=True)
confidences = pred_probs.max(-1)

# Cell 4: Analyze errors
errors = analyze_errors(
    test_data,
    pred_labels,
    predictions.label_ids,
    confidences,
    text_column='text'
)

print(f"Total errors: {len(errors)}")
show_error_examples(errors, n=5)

# Cell 5: Error patterns
patterns = analyze_error_patterns(errors)
print_error_patterns(patterns)

# Cell 6: Save error analysis
save_error_analysis(
    errors, 
    patterns, 
    '../outputs/exercise2/error_analysis.json'
)
print("✓ Error analysis saved")
```

**Deliverable**: ✅ Error analysis in `outputs/exercise2/error_analysis.json`

---

## 📋 Final Checklist

### Exercise 1
- [ ] Created `01_exercise1_sentiment_analysis.ipynb`
- [ ] Demonstrated sentiment analysis pipeline
- [ ] Showed tokenization process
- [ ] Understood model outputs

### Exercise 2
- [ ] Created 4 notebooks (data, training, eval, error analysis)
- [ ] Saved splits to `splits/` (once!)
- [ ] Trained model with monitoring
- [ ] Generated confusion matrix with explanation
- [ ] Computed all metrics (accuracy, P, R, F1)
- [ ] Analyzed errors and patterns
- [ ] Filled out `experiments/Exercise2_Finetuning_Log.md`

### Artifacts Created
- [ ] `splits/train_indices.json`
- [ ] `splits/val_indices.json`
- [ ] `splits/test_indices.json`
- [ ] `outputs/exercise2/training_history.csv`
- [ ] `outputs/exercise2/confusion_matrix.png`
- [ ] `outputs/exercise2/metrics.json`
- [ ] `outputs/exercise2/error_analysis.json`
- [ ] `outputs/exercise2/checkpoint-best/`

---

## 💡 Tips

1. **GPU vs CPU**: Training will be faster on GPU but works fine on CPU for small datasets
2. **Batch size**: If out of memory, reduce `per_device_train_batch_size` to 8 or 4
3. **Model choice**: `distilbert-base-uncased` is faster, `bert-base-uncased` is more accurate
4. **Epochs**: 3 epochs is usually enough for sentiment analysis

## 🚨 Common Issues

**Issue**: "Module 'src' not found"  
**Fix**: Add `sys.path.append('..')` at the start of each notebook

**Issue**: Splits change every run  
**Fix**: Always use `load_splits()`, never `create_stratified_splits()` after the first time

**Issue**: Training is slow  
**Fix**: Reduce batch size or use a smaller model

**Issue**: Out of memory  
**Fix**: Reduce `per_device_train_batch_size` or set `training_args.no_cuda = True`

---

## 🎉 You're Ready!

Start with Exercise 1, then move to Exercise 2. Follow the notebook structure above, and you'll be fine!

Remember: **Save everything, document everything, analyze errors.**

Good luck! 🚀
