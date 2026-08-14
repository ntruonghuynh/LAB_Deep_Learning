# resnet18_pretrained

## Purpose
Use a ResNet18 model pretrained on ImageNet as a transfer learning baseline for FashionMNIST classification.

## Hypothesis
A pretrained ResNet18 with a frozen backbone should learn the FashionMNIST classification head quickly, even with a small number of epochs.

## Changed from baseline
Replace the custom CNN with torchvision.models.resnet18 pretrained weights, resize FashionMNIST images to 224x224, convert grayscale images to 3 channels, freeze the feature extractor, and train only the final classifier layer.

## Kept constant
Use the same FashionMNIST dataset, validation ratio, seed, loss function, and train/validation/test evaluation protocol as the previous experiments.

## Expected observation
Validation accuracy should improve quickly compared with a randomly initialized model, while training remains relatively stable because only the classifier is trained.

## Observed result
- Best epoch: 5 (lowest validation loss)
- Best validation loss: 0.3755
- Best validation accuracy: 0.8638
- Final train accuracy: 0.8587
- Final validation accuracy: 0.8638
- Final train/validation accuracy gap: -0.0051

## Interpretation
See notebooks/02_training_and_comparison.ipynb for comparison with other runs.
