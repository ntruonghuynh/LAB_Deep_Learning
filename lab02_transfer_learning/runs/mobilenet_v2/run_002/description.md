# mobilenet_v2_pretrained

## Purpose
Test a lightweight MobileNetV2 pretrained on ImageNet for FashionMNIST transfer learning.

## Hypothesis
MobileNetV2 should train faster than heavier architectures while still achieving strong validation accuracy due to pretrained features.

## Changed from baseline
Use torchvision.models.mobilenet_v2 pretrained weights, replace the classifier with a 10-class output layer, and freeze the feature extractor.

## Kept constant
Use the same FashionMNIST split, seed, 224x224 RGB-style preprocessing, and evaluation workflow as other pretrained experiments.

## Expected observation
MobileNetV2 should be a good speed/accuracy tradeoff, especially on CPU.

## Observed result
- Best epoch: 4 (lowest validation loss)
- Best validation loss: 0.4019
- Best validation accuracy: 0.8528
- Final train accuracy: 0.8299
- Final validation accuracy: 0.8535
- Final train/validation accuracy gap: -0.0236

## Interpretation
See notebooks/02_training_and_comparison.ipynb for comparison with other runs.
