# Notebooks for Lab 3

This directory contains Jupyter notebooks for completing Lab 3 exercises.

## Recommended Notebook Structure

### Exercise 1: Sentiment Analysis (Quick Exploration)

**`01_exercise1_sentiment_analysis.ipynb`**
- Load pre-trained sentiment analysis pipeline
- Test on various examples
- Demonstrate tokenization step-by-step
- Show model outputs and confidence scores
- **Time**: ~30 minutes

### Exercise 2: Finetuning for Binary Classification

Split into multiple focused notebooks:

**`02_exercise2_data_exploration.ipynb`**
- Load dataset
- Explore statistics and distribution
- Create and save train/val/test splits
- Visualize text length distribution
- Show example texts from each class
- **Output**: `splits/*.json`

**`03_exercise2_training.ipynb`**
- Load saved splits
- Load model and tokenizer
- Tokenize datasets
- Configure training arguments
- Train model with monitoring
- Save training history and models
- **Output**: `outputs/exercise2/training_history.csv`, checkpoints

**`04_exercise2_evaluation.ipynb`**
- Load best model
- Evaluate on test set
- Compute all metrics (accuracy, precision, recall, F1)
- Generate confusion matrix
- Visualize training curves
- **Output**: `outputs/exercise2/metrics.json`, `confusion_matrix.png`

**`05_exercise2_error_analysis.ipynb`**
- Analyze misclassified examples
- Find error patterns
- Compare correct vs incorrect predictions
- Identify high-confidence errors
- Document findings
- **Output**: `outputs/exercise2/error_analysis.json`

## Quick Start Template

Each notebook should start with:

```python
# Add src to path
import sys
sys.path.append('..')

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Our utilities
from src.data import create_stratified_splits, save_splits, load_splits
from src.models import load_model_and_tokenizer, demonstrate_tokenization
from src.training import create_training_arguments, create_compute_metrics_fn
from src.evaluation import compute_classification_metrics, analyze_errors
from src.utils import plot_training_history, plot_confusion_matrix

# Configuration
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
%matplotlib inline
```

## Best Practices

### 1. Use Markdown Headers
Structure notebooks with clear sections:
```markdown
# Exercise 2: Data Exploration
## 1. Load Dataset
## 2. Basic Statistics
## 3. Create Splits
## 4. Visualize Distribution
```

### 2. Add Explanatory Text
Before each code cell, explain what you're doing:
```markdown
We'll create stratified splits to ensure balanced class distribution across train/val/test sets.
```

### 3. Show Outputs
Don't just compute - display results:
```python
# Good
stats = get_dataset_statistics(dataset)
print_dataset_statistics(stats)

# Not as good
stats = get_dataset_statistics(dataset)
```

### 4. Save Incrementally
Save important artifacts as you go:
```python
# After creating splits
save_splits(train_idx, val_idx, test_idx, output_dir='../splits')
print("✓ Splits saved to ../splits/")

# After training
trainer.save_model('../outputs/exercise2/checkpoint-best')
print("✓ Model saved")
```

### 5. Clear Cell Outputs Before Committing
Keep notebooks clean:
```bash
# In terminal
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

## Tips for Exercise 2

### Memory Management
```python
# If GPU memory is tight
import torch
torch.cuda.empty_cache()

# Use smaller batch size
training_args = create_training_arguments(
    output_dir='../outputs/exercise2',
    per_device_train_batch_size=8,  # Reduce if needed
    gradient_accumulation_steps=2   # Simulate larger batch
)
```

### Monitoring Training
```python
# Training progress is shown automatically
# To manually check a specific epoch:
history = pd.read_csv('../outputs/exercise2/training_history.csv')
print(f"Epoch 2: Loss={history.loc[1, 'train_loss']:.4f}")
```

### Quick Testing
```python
# Test on subset before full training
small_train = train_dataset.select(range(100))
small_val = val_dataset.select(range(50))

# Train for 1 epoch to verify pipeline works
training_args.num_train_epochs = 1
```

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution**: Add `sys.path.append('..')` at notebook start

### Issue: Splits keep changing
**Solution**: Always load from saved JSON, never recreate:
```python
# Wrong - creates new splits each time
train_idx, val_idx, test_idx = create_stratified_splits(dataset)

# Right - loads saved splits
train_idx, val_idx, test_idx = load_splits('../splits')
```

### Issue: Training is too slow
**Solution**: Use a smaller model or reduce batch size:
```python
# Try distilbert instead of bert
model_name = "distilbert-base-uncased"

# Reduce batch size
per_device_train_batch_size = 8
```

### Issue: "CUDA out of memory"
**Solution**: 
```python
# Reduce batch size
per_device_train_batch_size = 4
gradient_accumulation_steps = 4

# Or use CPU
training_args.no_cuda = True
```

## Notebook Checklist

Before considering a notebook complete:

- [ ] All cells run without errors
- [ ] Outputs are visible and explained
- [ ] Artifacts are saved to correct directories
- [ ] Plots have titles and labels
- [ ] Key findings are documented in markdown cells
- [ ] Code follows the utilities in `src/`
- [ ] Notebook follows the workflow rules in `LAB3_HUGGINGFACE_RULES.md`

## Example Notebook Flow (Exercise 2)

```
Data Exploration
→ Load & explore dataset
→ Create splits ONCE
→ Save to ../splits/

Training
→ Load saved splits
→ Tokenize
→ Train with monitoring
→ Save history & models

Evaluation
→ Load best model
→ Test set metrics
→ Confusion matrix
→ Training curves

Error Analysis
→ Find misclassifications
→ Pattern analysis
→ Document in experiment log
```

---

**Remember**: Notebooks are for exploration and presentation. Reusable code goes in `src/`.

Good luck! 🚀
