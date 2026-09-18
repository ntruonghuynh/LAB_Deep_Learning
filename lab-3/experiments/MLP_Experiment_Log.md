# MLP Experiment Log

## Run ID: mlp_run_001
**Date**: 2026-08-18  
**Purpose**: Baseline MLP training on image classification task (flatten input)

### Configuration
- Architecture: [To be filled - e.g., Flatten → FC(512) → ReLU → Dropout(0.5) → FC(256) → ReLU → FC(num_classes)]
- Optimizer: [e.g., Adam, lr=0.001, beta1=0.9, beta2=0.999]
- Batch size: [e.g., 32]
- Epochs: [e.g., 50]
- Loss function: [e.g., CrossEntropyLoss]
- Data split: train=[70%], val=[15%], test=[15%] (same as CNN)
- Input size: [e.g., 224x224x3 → flattened to 150528]

### Training Observations
- Convergence behavior: [Smooth? Unstable? Overfitting after epoch X?]
- Training time per epoch: [X seconds]
- Total training time: [Y minutes]
- Comparison to CNN: [Faster/slower convergence? More/less overfitting?]

### Results

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | - | - | - |
| Loss | - | - | - |

#### Per-Class Performance (Test Set)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Class 0 | - | - | - | - |
| Class 1 | - | - | - | - |
| ... | - | - | - | - |

### Confusion Matrix Analysis
- Most confused pairs: [e.g., Class A ↔ Class B (X samples), Class C ↔ Class D (Y samples)]
- Patterns observed: [e.g., struggles with spatial patterns that CNN handles well]
- Comparison to CNN: [Does MLP confuse different class pairs?]

### Error Analysis
- **Failure Mode 1**: [Description + hypothesis]
  - Example images: [Reference to saved error analysis plots]
  - Why MLP fails: [e.g., no spatial awareness, treats pixels independently]
- **Failure Mode 2**: [Description + hypothesis]

### Performance Metrics
- Parameters count: [X million]
- FLOPs: [Y million - likely less than CNN due to no convolutions]
- Training time/epoch: [X seconds - may be slower despite fewer FLOPs]
- Inference time/batch: [Y milliseconds - compare to CNN]

### Computation vs Latency Analysis
**Key Finding**: MLP has [X%] fewer FLOPs than CNN but [Y%] longer training time.

**Explanation**:
- **FLOPs**: MLP = [calculations], CNN = [calculations]
- **Memory access**: MLP has large FC weight matrices → poor cache locality
- **Hardware optimization**: CNN uses optimized conv kernels (cuDNN/MKL), better memory patterns
- **Parallelization**: CNN exploits spatial parallelism more efficiently

**Evidence**:
- Measured on [hardware spec]
- Profiling results: [memory bandwidth, cache hits, etc.]

### Conclusions
- [Key insight 1]
- [Comparison to CNN performance]
- [What MLP does well/poorly]
- [Why certain patterns emerge]

### Next Steps
- [Potential improvements to try]
- [Architectural variations]
- [Hypotheses to test]

---

## Run ID: mlp_run_002
**Date**: [Date]  
**Purpose**: [e.g., Experiment with deeper architecture]

[Same structure as above...]
