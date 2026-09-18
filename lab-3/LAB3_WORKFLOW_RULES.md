# Lab 3: CNN vs MLP - Workflow Rules & Requirements

## Overview
This lab compares CNN and MLP architectures on image classification, with emphasis on monitoring, traceability, and understanding model behavior.

---

## Complete Workflow Pipeline

```
DATA (Raw images)
    ↓
PREPROCESSING (Normalize, resize, augment)
    ↓
TRAIN / VAL / TEST SPLIT (固定 splits for reproducibility)
    ↓
SAVE SPLITS (JSON indices for exact reproducibility)
    ↓
MODEL DEFINITION (CNN architecture vs MLP architecture)
    ↓
TRAINING LOOP (with monitoring)
    ↓
MONITOR (Loss + Accuracy per epoch, both train & val)
    ↓
SAVE ARTIFACTS
    • Training history (CSV/JSON with all metrics)
    • Best model checkpoint (highest val accuracy)
    • Last model checkpoint (final epoch)
    ↓
TESTING (on held-out test set)
    ↓
COMPUTE METRICS
    • Overall Accuracy
    • Per-class Precision
    • Per-class Recall
    • Per-class F1-score
    ↓
CONFUSION MATRIX (with visualization & interpretation)
    ↓
ERROR ANALYSIS
    • Which classes are confused?
    • Visualize misclassified samples
    • Compare CNN vs MLP failure patterns
    ↓
PERFORMANCE COMPARISON
    • Why does MLP have lower computation but higher latency?
    • Compare training time, inference time, parameter count
```

---

## Mandatory Requirements

### R1: Split Preservation
**RULE**: Use fixed train/val/test splits saved as JSON index files.
**WHY**: Ensure fair comparison between CNN and MLP on identical data.
**APPLY**:
- Generate splits once at the beginning
- Save indices to `splits/train_indices.json`, `val_indices.json`, `test_indices.json`
- Load same splits for all experiments
- Document split ratios (e.g., 70/15/15 or 80/10/10)

### R2: Training Monitoring Visibility
**RULE**: The instructor must see the training process monitored in real-time or post-hoc.
**WHY**: Understand convergence behavior, detect overfitting, validate learning.
**APPLY**:
- Plot loss curves (train vs val) per epoch
- Plot accuracy curves (train vs val) per epoch
- Show these plots in the notebook
- Save raw history data (CSV/JSON) for re-plotting

### R3: Artifact Preservation
**RULE**: Save both training history AND model checkpoints.
**WHY**: Enable result verification and model reuse.
**APPLY**:
- Save training history: `outputs/<run_id>/metrics/history.csv`
- Save best model: `outputs/<run_id>/model/best_model.h5` or `.pth`
- Save last model: `outputs/<run_id>/model/last_model.h5` or `.pth`
- Include metadata: config used, final metrics, timestamp

### R4: Confusion Matrix Interpretation
**RULE**: Display confusion matrix with full explanation of rows, columns, and cell meanings.
**WHY**: Many students don't understand CM semantics; instructor wants explicit teaching.
**APPLY**:
- **Rows**: True labels (actual class)
- **Columns**: Predicted labels (model's prediction)
- **Cell [i, j]**: Number of samples with true class i predicted as class j
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- Annotate the visualization with these explanations
- Show per-class precision/recall derived from CM

### R5: Performance Analysis - Computation vs Latency
**RULE**: Explain why MLP has fewer FLOPs but potentially higher latency than CNN.
**WHY**: Counter-intuitive result that reveals hardware optimization differences.
**APPLY**:
- Count parameters: CNN vs MLP
- Measure training time per epoch
- Measure inference time per batch
- Explain:
  - CNN uses convolution (highly optimized kernels: cuDNN, MKL)
  - MLP uses fully-connected layers (less memory-efficient, poor cache locality)
  - CNN exploits spatial locality and weight sharing
  - MLP has large weight matrices → memory bandwidth bottleneck

### R6: Notebook as Story
**RULE**: Notebooks should tell the experimental story, not just run code.
**WHY**: Reader should understand methodology and conclusions without reading source code.
**APPLY**:
- Use markdown cells to explain each step
- Show visualizations inline
- Interpret results after each major step
- Compare CNN vs MLP throughout, not just at the end

### R7: Error Analysis Beyond Metrics
**RULE**: Don't stop at accuracy numbers; analyze failure modes.
**WHY**: Understanding errors drives improvement.
**APPLY**:
- Identify most-confused class pairs
- Visualize samples from these pairs (true class A predicted as B)
- Hypothesize why confusion occurs (visual similarity, data imbalance)
- Compare: does CNN handle these cases better than MLP? Why?

---

## Proposed Directory Structure

```
Lab-3/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # EDA, visualize dataset
│   ├── 02_preprocessing_splits.ipynb     # Create & save splits
│   ├── 03_cnn_training.ipynb             # Train CNN, monitor, save
│   ├── 04_mlp_training.ipynb             # Train MLP, monitor, save
│   └── 05_comparison_analysis.ipynb      # Compare results, error analysis
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                    # Dataset class, loading logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py                        # CNN architecture definition
│   │   └── mlp.py                        # MLP architecture definition
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                    # Training loop with monitoring
│   │   └── callbacks.py                  # Checkpoint, early stopping
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                    # Precision, recall, F1
│   │   └── confusion_matrix.py           # CM computation & visualization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py              # Normalization, augmentation
│       ├── visualization.py              # Plot utilities
│       └── io_utils.py                   # Save/load models, histories
│
├── splits/
│   ├── train_indices.json                # Fixed train split
│   ├── val_indices.json                  # Fixed validation split
│   └── test_indices.json                 # Fixed test split
│
├── outputs/
│   ├── cnn_run_001/
│   │   ├── config.json                   # Hyperparameters used
│   │   ├── metrics/
│   │   │   ├── history.csv               # Epoch-by-epoch metrics
│   │   │   ├── test_metrics.json         # Final test results
│   │   │   └── confusion_matrix.npy      # Raw CM data
│   │   ├── model/
│   │   │   ├── best_model.pth            # Best checkpoint
│   │   │   └── last_model.pth            # Final checkpoint
│   │   └── plots/
│   │       ├── loss_curve.png
│   │       ├── accuracy_curve.png
│   │       └── confusion_matrix.png
│   │
│   └── mlp_run_001/
│       └── (same structure as CNN)
│
├── experiments/
│   ├── CNN_Experiment_Log.md             # CNN experiment documentation
│   └── MLP_Experiment_Log.md             # MLP experiment documentation
│
├── RULES.md                              # General lab rules
├── LAB3_WORKFLOW_RULES.md                # This file
├── README.md                             # Lab overview & reproduction guide
├── requirements.txt
└── .gitignore
```

---

## Experiment Logging Template

Each experiment log (`CNN_Experiment_Log.md`, `MLP_Experiment_Log.md`) should contain:

```markdown
# [Model Name] Experiment Log

## Run ID: [model]_run_[number]
**Date**: YYYY-MM-DD  
**Purpose**: [Why this run exists]

### Configuration
- Architecture: [Detailed layer spec]
- Optimizer: [Name, learning rate, momentum, etc.]
- Batch size: [Number]
- Epochs: [Number]
- Loss function: [Name]
- Data split: train=[%], val=[%], test=[%]

### Training Observations
- Convergence behavior: [Smooth? Unstable? Overfitting?]
- Training time per epoch: [X seconds]
- Total training time: [Y minutes]

### Results
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | X.XX | X.XX | X.XX |
| Loss | X.XX | X.XX | X.XX |

### Error Analysis
- Most confused classes: [A ↔ B, C ↔ D]
- Failure patterns: [Observations]

### Conclusions
- [Key insights from this run]
- [Next steps or hypotheses]
```

---

## Key Questions to Answer in Notebooks

### 1. Data Understanding
- What is the dataset? (classes, images per class, resolution)
- Any class imbalance?
- Visual characteristics of each class?

### 2. Preprocessing
- What normalization strategy? (mean/std from training set)
- Any augmentation? (rotation, flip, crop)
- Why these choices?

### 3. Model Architecture
- CNN: How many conv layers? Filter sizes? Pooling?
- MLP: How many hidden layers? Neurons per layer?
- Parameter count comparison?

### 4. Training Monitoring
- Do both models converge?
- Any overfitting? (train accuracy >> val accuracy)
- Which converges faster?

### 5. Performance Metrics
- Which model has higher accuracy?
- Are there per-class performance differences?
- Precision vs recall trade-offs?

### 6. Confusion Matrix Interpretation
- **For CNN**: Which classes does it confuse?
- **For MLP**: Same question
- **Comparison**: Does CNN handle spatial patterns better?

### 7. Computation vs Latency Paradox
- **FLOPs**: MLP may have fewer operations
- **Latency**: CNN may be faster due to:
  - Optimized conv kernels (cuDNN)
  - Better memory access patterns
  - Weight sharing reduces memory bandwidth
- **Evidence**: Time both models on same hardware

---

## Checklist Before Submission

- [ ] Splits saved and reused across all experiments
- [ ] Training curves (loss & accuracy) plotted for both models
- [ ] Raw history saved as CSV/JSON
- [ ] Best and last model checkpoints saved
- [ ] Confusion matrix displayed with axis labels and explanation
- [ ] Per-class precision/recall/F1 computed
- [ ] Error analysis: most confused classes identified and visualized
- [ ] Computation vs latency explained with measurements
- [ ] Experiment logs document all runs
- [ ] README explains how to reproduce results
- [ ] Notebooks have markdown narratives, not just code
- [ ] All plots traceable to specific run IDs

---

## Anti-Patterns Specific to This Lab

❌ Generating new random splits each run (breaks comparability)  
❌ Only showing final accuracy without training curves  
❌ Confusion matrix without row/column labels or explanation  
❌ Claiming "CNN is better" without error analysis  
❌ Not measuring actual training/inference time  
❌ Saving only plots, not raw metrics  
❌ Notebooks that are just code cells without explanation  
❌ Overwriting previous runs when tuning hyperparameters  

---

## Success Criteria

✅ Instructor can see the full training process (curves, convergence)  
✅ Anyone can load splits and reproduce exact results  
✅ Confusion matrix is fully explained (not just shown)  
✅ Error analysis reveals why models fail, not just that they fail  
✅ Computation/latency paradox addressed with evidence  
✅ Notebooks tell a clear experimental story  
✅ All results traceable to saved artifacts  
