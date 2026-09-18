"""
CNN and MLP model architectures for Lab 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for image classification.

    Architecture:
        Conv2D(3, 32, 3x3) → ReLU → MaxPool(2x2)
        Conv2D(32, 64, 3x3) → ReLU → MaxPool(2x2)
        Conv2D(64, 128, 3x3) → ReLU → MaxPool(2x2)
        Flatten → FC(512) → ReLU → Dropout(0.5) → FC(num_classes)
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Note: Adjust input size based on your image dimensions after pooling
        # For 32x32 input: after 3 poolings → 4x4x128 = 2048
        # For 224x224 input: after 3 poolings → 28x28x128 = 100352
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjust for your input size
        self.fc2 = nn.Linear(512, num_classes)

        # Regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))

        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleMLP(nn.Module):
    """
    Simple MLP architecture for image classification.

    Architecture:
        Flatten → FC(512) → ReLU → Dropout(0.5)
        FC(512, 256) → ReLU → Dropout(0.5)
        FC(256, num_classes)
    """

    def __init__(self, num_classes: int = 10, input_size: int = 32*32*3):
        super(SimpleMLP, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn(num_classes: int, input_channels: int = 3) -> SimpleCNN:
    """Factory function to create CNN model."""
    model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    print(f"CNN created with {model.count_parameters():,} parameters")
    return model


def create_mlp(num_classes: int, input_size: int) -> SimpleMLP:
    """Factory function to create MLP model."""
    model = SimpleMLP(num_classes=num_classes, input_size=input_size)
    print(f"MLP created with {model.count_parameters():,} parameters")
    return model
