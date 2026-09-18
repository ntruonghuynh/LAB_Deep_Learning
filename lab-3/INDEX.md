# Lab 3 Documentation Index

Quick navigation for all Lab 3 documentation.

## 🎯 Start Here

**New to this lab?**
1. Read [GETTING_STARTED.md](GETTING_STARTED.md) - Step-by-step setup and exercise walkthroughs
2. Skim [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Understand what you'll build
3. Keep [QUICK_REFERENCE.md](QUICK_REFERENCE.md) open - Copy-paste code snippets

**Already started?**
- Jump to [notebooks/README.md](notebooks/README.md) for notebook-specific guidance
- Check [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) for workflow rules

## 📚 Documentation Map

### Quick Start & Overview
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Setup + complete exercise walkthroughs (START HERE!)
- **[README.md](README.md)** - Project overview, quick start, tips
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Comprehensive guide with concepts and deliverables
- **[INDEX.md](INDEX.md)** - This file (navigation hub)

### Reference & Rules
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Code snippets cheat sheet
- **[LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md)** - 10 workflow rules (R1-R10)
- **[requirements.txt](requirements.txt)** - Python dependencies

### Implementation Guides
- **[notebooks/README.md](notebooks/README.md)** - Notebook best practices and troubleshooting
- **[experiments/Exercise2_Finetuning_Log.md](experiments/Exercise2_Finetuning_Log.md)** - Experiment tracking template

### Source Code Documentation
- **[src/data/](src/data/)** - Dataset utilities (splits, tokenization, stats)
- **[src/models/](src/models/)** - Model loading and tokenization
- **[src/training/](src/training/)** - Training configuration and monitoring
- **[src/evaluation/](src/evaluation/)** - Metrics computation and error analysis
- **[src/utils/](src/utils/)** - Visualization utilities

### Legacy/Reference
- **[RULES.md](RULES.md)** - Original general deep learning rules
- **[LAB3_WORKFLOW_RULES.md](LAB3_WORKFLOW_RULES.md)** - CNN vs MLP workflow (not for this lab)
- **[SUMMARY.md](SUMMARY.md)** - Setup summary and checklist

## 🗺️ Learning Path

### Exercise 1: Sentiment Analysis (30 min)
```
GETTING_STARTED.md (Exercise 1 section)
→ Create notebook 01_exercise1_sentiment_analysis.ipynb
→ Use code from QUICK_REFERENCE.md (Exercise 1 section)
→ Check notebooks/README.md for tips
```

### Exercise 2: Finetuning (2-3 hours)
```
GETTING_STARTED.md (Exercise 2 sections)
→ Part 1: Data Exploration
   └─ Create notebook 02_exercise2_data_exploration.ipynb
   └─ Reference: src/data/dataset_utils.py
   
→ Part 2: Training
   └─ Create notebook 03_exercise2_training.ipynb
   └─ Reference: src/training/trainer_utils.py
   └─ Follow LAB3_HUGGINGFACE_RULES.md (R2: Monitor training)
   
→ Part 3: Evaluation
   └─ Create notebook 04_exercise2_evaluation.ipynb
   └─ Reference: src/evaluation/eval_metrics.py
   └─ Follow LAB3_HUGGINGFACE_RULES.md (R4-R5: CM and metrics)
   
→ Part 4: Error Analysis
   └─ Create notebook 05_exercise2_error_analysis.ipynb
   └─ Reference: src/evaluation/error_analysis.py
   └─ Follow LAB3_HUGGINGFACE_RULES.md (R6: Error analysis)

→ Documentation
   └─ Fill out experiments/Exercise2_Finetuning_Log.md
   └─ Follow LAB3_HUGGINGFACE_RULES.md (R9-R10)
```

## 📖 By Task

### "I want to..."

**...understand the lab requirements**
→ [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (Objectives & Exercises sections)

**...set up my environment**
→ [GETTING_STARTED.md](GETTING_STARTED.md) (Setup section)

**...start Exercise 1**
→ [GETTING_STARTED.md](GETTING_STARTED.md) (Exercise 1 section)

**...start Exercise 2**
→ [GETTING_STARTED.md](GETTING_STARTED.md) (Exercise 2 sections)

**...find code snippets**
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**...understand the workflow rules**
→ [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md)

**...create reproducible splits**
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (Data Management section)
→ [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) (R1)

**...understand confusion matrix**
→ [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (Key Concepts section)
→ [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) (R4)

**...analyze errors**
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (Error Analysis section)
→ [src/evaluation/error_analysis.py](src/evaluation/error_analysis.py)

**...document my experiments**
→ [experiments/Exercise2_Finetuning_Log.md](experiments/Exercise2_Finetuning_Log.md)

**...troubleshoot notebook issues**
→ [notebooks/README.md](notebooks/README.md) (Common Issues section)
→ [GETTING_STARTED.md](GETTING_STARTED.md) (Common Issues section)

**...understand the project structure**
→ [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (Project Structure section)

**...see example notebook code**
→ [GETTING_STARTED.md](GETTING_STARTED.md) (all exercise sections)

## 🎓 By Learning Goal

### Understanding Pre-trained Models
- [GETTING_STARTED.md](GETTING_STARTED.md) - Exercise 1 section
- [src/models/model_utils.py](src/models/model_utils.py) - `load_sentiment_pipeline()`

### Understanding Tokenization
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Key Concepts
- [src/models/model_utils.py](src/models/model_utils.py) - `demonstrate_tokenization()`
- [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) - R7

### Creating Reproducible Experiments
- [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) - R1
- [src/data/dataset_utils.py](src/data/dataset_utils.py) - Split management

### Training with Monitoring
- [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) - R2
- [src/training/trainer_utils.py](src/training/trainer_utils.py)

### Comprehensive Evaluation
- [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) - R4, R5
- [src/evaluation/eval_metrics.py](src/evaluation/eval_metrics.py)
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Metrics Explained

### Error Analysis
- [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) - R6
- [src/evaluation/error_analysis.py](src/evaluation/error_analysis.py)

### Documentation
- [experiments/Exercise2_Finetuning_Log.md](experiments/Exercise2_Finetuning_Log.md)
- [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md) - R9, R10

## 🔧 By Component

### Data Pipeline
```
LAB3_HUGGINGFACE_RULES.md (R1)
→ src/data/dataset_utils.py
  ├─ create_stratified_splits()
  ├─ save_splits() / load_splits()
  ├─ get_dataset_statistics()
  └─ prepare_dataset_for_training()
→ QUICK_REFERENCE.md (Data Management section)
```

### Model & Tokenization
```
LAB3_HUGGINGFACE_RULES.md (R7, R8)
→ src/models/model_utils.py
  ├─ load_sentiment_pipeline()
  ├─ demonstrate_tokenization()
  └─ load_model_and_tokenizer()
→ QUICK_REFERENCE.md (Model & Tokenization section)
```

### Training Pipeline
```
LAB3_HUGGINGFACE_RULES.md (R2, R3)
→ src/training/trainer_utils.py
  ├─ create_training_arguments()
  ├─ create_compute_metrics_fn()
  └─ save_training_history()
→ QUICK_REFERENCE.md (Training section)
```

### Evaluation & Analysis
```
LAB3_HUGGINGFACE_RULES.md (R4, R5, R6)
→ src/evaluation/eval_metrics.py
  ├─ compute_classification_metrics()
  ├─ compute_confusion_matrix()
  └─ explain_confusion_matrix()
→ src/evaluation/error_analysis.py
  ├─ analyze_errors()
  ├─ analyze_error_patterns()
  └─ save_error_analysis()
→ QUICK_REFERENCE.md (Evaluation section)
```

### Visualization
```
src/utils/visualization.py
  ├─ plot_training_history()
  ├─ plot_confusion_matrix()
  ├─ plot_error_distribution()
  └─ plot_confidence_distribution()
→ QUICK_REFERENCE.md (Visualization section)
```

## 📱 Quick Links

### Must Read (Everyone)
1. [GETTING_STARTED.md](GETTING_STARTED.md)
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. [LAB3_HUGGINGFACE_RULES.md](LAB3_HUGGINGFACE_RULES.md)

### Keep Open While Working
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - For copy-paste
- [notebooks/README.md](notebooks/README.md) - For troubleshooting

### Reference When Needed
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - For concepts
- [experiments/Exercise2_Finetuning_Log.md](experiments/Exercise2_Finetuning_Log.md) - For documentation

## ✅ Completion Checklist

Use this to track your progress:

### Setup
- [ ] Read GETTING_STARTED.md
- [ ] Installed dependencies
- [ ] Verified installation
- [ ] Understand project structure

### Exercise 1
- [ ] Created notebook 01
- [ ] Completed sentiment analysis
- [ ] Understood tokenization

### Exercise 2 - Data
- [ ] Created notebook 02
- [ ] Explored dataset
- [ ] Created and saved splits

### Exercise 2 - Training
- [ ] Created notebook 03
- [ ] Trained model
- [ ] Monitored training
- [ ] Saved artifacts

### Exercise 2 - Evaluation
- [ ] Created notebook 04
- [ ] Computed metrics
- [ ] Generated confusion matrix
- [ ] Plotted training curves

### Exercise 2 - Error Analysis
- [ ] Created notebook 05
- [ ] Analyzed errors
- [ ] Identified patterns
- [ ] Saved analysis

### Documentation
- [ ] Filled experiment log
- [ ] All artifacts saved
- [ ] Ready for submission

---

**Need help?** Start with [GETTING_STARTED.md](GETTING_STARTED.md) and work through step-by-step!

**Ready to start?** → [GETTING_STARTED.md](GETTING_STARTED.md)

Good luck! 🚀
