# Deep Learning Lab Rules - Rule-Based System

## Core Principles

### R1: Pipeline-Driven Presentation
**RULE**: Present work following the Machine Learning pipeline, NOT as a list of code files.
**WHY**: Focus on methodology and experimentation flow, not implementation details.
**APPLY**: Structure reports, notebooks, and documentation around: data → preprocessing → model → training → evaluation → analysis.

### R2: Separation of Concerns
**RULE**: `src/` contains reusable logic; notebooks are for observation, monitoring, analysis, and storytelling.
**WHY**: Maintain clean boundaries between production code and experimental exploration.
**APPLY**: 
- Place functions, classes, utilities in `src/`
- Use notebooks to call `src/` code and visualize results
- Never duplicate logic between notebooks

### R3: Experiment Immutability
**RULE**: Each significant training session is a separate run; NEVER overwrite previous run history.
**WHY**: Preserve full experimental lineage for comparison and debugging.
**APPLY**: 
- Create unique run IDs or timestamps for each experiment
- Never reuse output directories across runs
- Keep all historical runs accessible

### R4: Run Traceability
**RULE**: Every run must be traceable with: purpose, config, data/split, metrics, artifacts/checkpoints, and conclusions.
**WHY**: Enable reproducibility and reasoning about experiment outcomes.
**APPLY**: Log or document:
- What question this run answers
- Exact configuration/hyperparameters used
- Dataset version and split strategy
- All metrics computed
- Model checkpoints if applicable
- Insights and conclusions drawn

### R5: Raw Metrics Preservation
**RULE**: Always save raw metrics (e.g., history.csv/json), not just visualization images.
**WHY**: Enable re-analysis and different visualizations without re-running experiments.
**APPLY**: 
- Export training history to CSV/JSON
- Save confusion matrices as data, not just images
- Store per-class metrics in structured format

### R6: Deep Error Analysis
**RULE**: Evaluation cannot stop at a single accuracy number; analyze where and why the model fails.
**WHY**: Understand model behavior and identify improvement opportunities.
**APPLY**:
- Examine failure cases by category
- Visualize misclassifications
- Investigate performance across different data subgroups
- Document error patterns and hypotheses

### R7: Demand-Driven Structure
**RULE**: Do not create folders because "the template requires it". Optional folders only appear when the Lab actually produces corresponding artifacts.
**WHY**: Avoid empty directories and cargo-cult structure.
**APPLY**: Create folders only when you have content to put in them.

### R8: Report Traceability
**RULE**: Every figure/number in the report must be traceable back to a specific run.
**WHY**: Ensure reproducibility and prevent cherry-picking results.
**APPLY**: 
- Include run IDs in figure captions
- Link metrics to specific experiment configurations
- Maintain mapping from report claims to experiment evidence

---

## Pre-Lab Design: 7 Critical Questions

Before creating ANY files or folders for a new Lab, the team must answer these questions. The answers determine the required structure.

### Q1: Problem Definition
**What is the problem? What are the input, output, and primary metrics that truly reflect the objective?**

### Q2: Data Pipeline
**What states does the data go through? Do raw, preprocessed, augmented, and split data need to be saved?**

### Q3: Experimental Factors
**Which decisions need to be compared through experiments? Architecture, optimizer, learning rate, augmentation, loss function, threshold…?**

### Q4: Training Monitoring
**What needs to be monitored during training to know if the model is learning correctly or has issues?**

### Q5: Result Artifacts
**What raw artifacts need to be saved to reconstruct tables/charts?**

### Q6: Error Analysis Strategy
**What does error analysis look like for this problem type? Classification, detection, segmentation, sequence… each differs.**

### Q7: Reproducibility Requirements
**What does someone cloning the repo need to know to re-run or verify conclusions?**

---

## Standard Architecture for All Labs

This is the shared framework. Each Lab has "core" (mandatory) and "optional" (as-needed) components.

### Application Principle
This convention standardizes thinking and experimental evidence preservation, NOT to force identical structure across all Labs. When starting a new Lab, return to the 7 questions in Section 2 to decide what to keep, add, or remove. However, the principles of pipeline, traceability, raw metrics, monitoring, and experiment reasoning must be maintained throughout.

---

## Implementation Guidelines

### When Starting a New Lab
1. Answer the 7 critical questions as a team
2. Design the minimal structure needed based on answers
3. Create only the folders/files that will be immediately used
4. Establish naming conventions for runs (timestamps, sequential IDs)
5. Set up logging and metric tracking infrastructure first

### During Experimentation
1. Never reuse run identifiers
2. Log configuration before training starts
3. Save metrics continuously, not just at the end
4. Document observations and hypotheses in notebooks
5. Create checkpoints at meaningful milestones

### During Analysis
1. Load raw metrics from saved files, don't rely on memory
2. Compare runs systematically using their IDs
3. Investigate outliers and failure modes
4. Document insights that inform next experiments

### Before Reporting
1. Verify every claim traces to a specific run
2. Check that all figures can be regenerated from saved data
3. Ensure someone else could reproduce key results
4. Clean up only the structure, never delete run history

---

## Anti-Patterns to Avoid

❌ Creating empty folders "just in case"  
❌ Overwriting previous experiment results  
❌ Saving only final accuracy without intermediate metrics  
❌ Storing only plots without underlying data  
❌ Duplicating code between notebooks and src/  
❌ Presenting results without experiment IDs  
❌ Using fixed paths that break when structure changes  
❌ Committing large model files without versioning strategy  

---

## Checklist for Lab Completion

- [ ] All 7 design questions have documented answers
- [ ] Every significant run has unique identifier
- [ ] Raw metrics saved for all experiments
- [ ] Error analysis completed, not just accuracy
- [ ] All report figures link to specific runs
- [ ] README explains how to reproduce key results
- [ ] Only necessary folders exist (no empty templates)
- [ ] Notebooks tell clear experimental story
- [ ] src/ contains clean, reusable code
- [ ] Future team members can understand and extend work
