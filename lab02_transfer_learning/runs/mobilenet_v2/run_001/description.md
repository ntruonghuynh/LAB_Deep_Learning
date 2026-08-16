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
