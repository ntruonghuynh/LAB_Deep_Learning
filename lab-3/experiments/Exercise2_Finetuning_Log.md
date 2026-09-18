# Exercise 2 - Finetuning Experiment Log

## Purpose
Document all training runs, configurations, results, and insights for Exercise 2 (finetuning a pre-trained model for binary text classification).

---

## Dataset Information

**Dataset Name**: IMDb Movie Reviews (Binary Sentiment Classification)

**Source**: Hugging Face datasets library (`datasets.load_dataset('imdb')`)

**Task**: Binary text classification (Negative: 0, Positive: 1)

**Statistics**:
- Total samples: 25,000 (train set partitioned into stratified train/val/test)
- Train samples: 17,500 (70%)
- Validation samples: 3,750 (15%)
- Test samples: 3,750 (15%)
- Class distribution: 50% positive (1,875), 50% negative (1,875) per eval set
- Average text length: 234 words (Median: 174 words, Max: 2,470 words)

**Observations**:
- Perfectly balanced class distribution (no class imbalance mitigation required).
- Text length varies significantly, max length capped at 256 tokens for efficient training with minimal information loss.

---

## Run 1: DistilBERT Baseline Fine-tuning

**Date**: 2026-08-22

**Goal**: Establish baseline performance with DistilBERT using Hugging Face Trainer and AdamW with linear warmup schedule.

### Configuration
- **Model**: `distilbert-base-uncased` (66,955,010 total / trainable parameters)
- **Epochs**: 3
- **Batch size**: 32 (Train per-device), 64 (Eval per-device)
- **Learning rate**: 2e-5
- **Weight decay**: 0.01
- **Warmup ratio**: 0.1 (10% of total steps)
- **Max sequence length**: 256
- **Optimizer**: AdamW with `fp16` mixed precision
- **Evaluation Strategy**: Epoch-level evaluation & checkpointing (`save_strategy='epoch'`)
- **Metric for Best Model**: `accuracy` (`load_best_model_at_end=True`)

### Results

**Training**:
- Final train loss: 0.2224
- Training time: ~387 seconds (~6.5 minutes)
- Throughput: 135.6 samples/second

**Validation Progression**:
- Epoch 1: Val Loss = 0.2309, Val Accuracy = 0.9104, Val F1 = 0.9091
- Epoch 2: Val Loss = 0.2338, Val Accuracy = 0.9109, Val F1 = 0.9110
- Epoch 3: Val Loss = 0.2754, Val Accuracy = 0.9157, Val F1 = 0.9163
- Best Epoch: Epoch 3 (Accuracy: 0.9157, F1: 0.9163)

**Test Metrics**:
- Accuracy: **0.9147** (91.47%)
- Precision (Weighted): **0.9147** (Negative: 0.9142, Positive: 0.9151)
- Recall (Weighted): **0.9147** (Negative: 0.9152, Positive: 0.9141)
- F1-Score (Weighted): **0.9147** (Negative: 0.9147, Positive: 0.9146)

**Confusion Matrix**:
```
              Predicted
              NEG     POS
Actual NEG [ 1716     159 ]
       POS [  161    1714 ]
```
- **TN (True Negative)**: 1,716
- **FP (False Positive - Type I error)**: 159
- **FN (False Negative - Type II error)**: 161
- **TP (True Positive)**: 1,714

---

### Analysis

**Training Behavior**:
- The model converges smoothly across 3 epochs. Training loss drops from 0.6923 down to ~0.0875 towards the end of training.
- Validation accuracy consistently improved from 91.04% in Epoch 1 to 91.57% in Epoch 3.
- Minimal overfitting observed, with test accuracy matching validation performance closely (91.47% vs 91.57%).

**Error Analysis**:
- Total errors: 320 / 3,750 (8.53% error rate)
- False Positives (NEG predicted as POS): 159
- False Negatives (POS predicted as NEG): 161
- Error patterns:
  - **Sarcasm and subtle irony**: Reviews with sarcastic praise containing positive words are sometimes misclassified as Positive.
  - **Double negation & mixed reviews**: Long reviews containing both praise for actors and criticism for the plot cause ambiguous predictions.
  - **Text length correlation**: Incorrect predictions had a slightly higher mean length (254.8 words vs 231.2 words for correct predictions), showing truncation at 256 tokens occasionally excludes concluding sentiment.

**Confidence Analysis**:
- Mean confidence on correct predictions is significantly higher (~0.98) than on misclassified samples (~0.79).
- The model displays calibrated uncertainty on difficult/ambiguous examples.

---

### Conclusions

**What Worked**:
- `distilbert-base-uncased` achieved >91.4% accuracy with fast convergence and low memory footprint.
- Stratified 70/15/15 splits provided balanced, reliable validation metrics across classes.
- Mixed precision (`fp16`) enabled fast throughput (135+ samples/sec).

---

## Best Model Summary

- **Saved Location**: `./outputs/exercise2/model/best_model/`
- **Architecture**: `DistilBertForSequenceClassification`
- **Test Accuracy**: 91.47%
- **Artifacts**:
  - `outputs/exercise2/plots/training_curves.png`
  - `outputs/exercise2/plots/confusion_matrix.png`
  - `outputs/exercise2/plots/error_analysis.png`
  - `outputs/exercise2/metrics/training_history.csv`
  - `outputs/exercise2/metrics/all_metrics.json`
  - `outputs/exercise2/metrics/error_analysis.json`
