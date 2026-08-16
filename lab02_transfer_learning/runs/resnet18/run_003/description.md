# resnet18_finetune

## Purpose
Fine-tune the last convolutional block of an ImageNet-pretrained ResNet18 for FashionMNIST classification.

## Hypothesis
Unfreezing the final ResNet block should allow the model to adapt better to FashionMNIST while still benefiting from pretrained visual features.

## Changed from baseline
Start from the frozen ResNet18 pretrained baseline, then unfreeze layer4 and use a smaller learning rate to fine-tune the final feature block and classifier.

## Kept constant
Keep the same dataset, image size, ImageNet normalization, batch size, seed, and evaluation protocol as the ResNet18 pretrained baseline.

## Expected observation
Fine-tuning may improve validation accuracy over the classifier-only baseline, but may also overfit faster if trained for too many epochs.

## Observed result
- Best epoch: 4 (lowest validation loss)
- Best validation loss: 0.1929
- Best validation accuracy: 0.9320
- Final train accuracy: 0.9715
- Final validation accuracy: 0.9275
- Final train/validation accuracy gap: 0.0440

## Interpretation
See notebooks/02_training_and_comparison.ipynb for comparison with other runs.
