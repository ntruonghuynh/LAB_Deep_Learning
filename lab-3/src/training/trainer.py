"""
Training utilities for Lab 3: CNN vs MLP
Includes training loop with monitoring and checkpoint saving.
"""

import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import json


class Trainer:
    """
    Training loop with monitoring for loss and accuracy.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 device: torch.device,
                 output_dir: str):
        """
        Args:
            model: Neural network model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on (cuda/cpu)
            output_dir: Directory to save checkpoints and metrics
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)

        # Create output directories
        (self.output_dir / 'model').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metrics').mkdir(parents=True, exist_ok=True)

        # History tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': []
        }

        self.best_val_acc = 0.0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs with monitoring.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        print(f"Training for {num_epochs} epochs...")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            epoch_time = time.time() - start_time

            # Update history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')

            # Print progress
            if verbose:
                print(f"Epoch [{epoch}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                      f"Time: {epoch_time:.2f}s")

        # Save last model
        self.save_checkpoint('last_model.pth')

        # Save history
        self.save_history()

        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / 'model' / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)

    def save_history(self):
        """Save training history to JSON and CSV."""
        import pandas as pd

        # Save as CSV
        csv_path = self.output_dir / 'metrics' / 'history.csv'
        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)
        print(f"History saved to {csv_path}")

        # Save as JSON
        json_path = self.output_dir / 'metrics' / 'history.json'
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / 'model' / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded checkpoint from {path}")
