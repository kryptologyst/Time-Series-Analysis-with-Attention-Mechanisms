"""
Training and evaluation module for time series models.

This module provides comprehensive training, validation, and evaluation
functionality for attention-based time series models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsCalculator:
    """Calculate various evaluation metrics for time series forecasting."""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all available metrics."""
        return {
            'mse': MetricsCalculator.mse(y_true, y_pred),
            'mae': MetricsCalculator.mae(y_true, y_pred),
            'rmse': MetricsCalculator.rmse(y_true, y_pred),
            'mape': MetricsCalculator.mape(y_true, y_pred),
            'smape': MetricsCalculator.smape(y_true, y_pred)
        }


class TimeSeriesTrainer:
    """Trainer class for time series models with attention mechanisms."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.loss_fn = nn.MSELoss()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('min_delta', 0.0)
        )
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions, _ = self.model(batch_x)
            loss = self.loss_fn(predictions, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.metrics_calculator.calculate_all_metrics(
            np.array(all_targets), np.array(all_predictions)
        )
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions, _ = self.model(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self.metrics_calculator.calculate_all_metrics(
            np.array(all_targets), np.array(all_predictions)
        )
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_metrics'].append(train_metrics)
            self.train_history['val_metrics'].append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Val RMSE: {val_metrics['rmse']:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.train_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions, attention_weights = self.model(batch_x)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                attention_weights_list.append(attention_weights.cpu().numpy())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            np.array(all_targets), np.array(all_predictions)
        )
        
        logger.info("Test Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.6f}")
        
        return {
            'metrics': metrics,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'attention_weights': np.concatenate(attention_weights_list, axis=0)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_history': self.train_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.train_history = checkpoint['train_history']
        logger.info(f"Model loaded from {filepath}")


class ModelComparator:
    """Compare multiple models and their performance."""
    
    def __init__(self):
        """Initialize the model comparator."""
        self.results = {}
        self.metrics_calculator = MetricsCalculator()
    
    def add_model_result(self, model_name: str, predictions: np.ndarray, 
                         targets: np.ndarray, attention_weights: Optional[np.ndarray] = None):
        """
        Add model results for comparison.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions
            targets: True targets
            attention_weights: Optional attention weights
        """
        metrics = self.metrics_calculator.calculate_all_metrics(targets, predictions)
        
        self.results[model_name] = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics,
            'attention_weights': attention_weights
        }
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get a comparison table of all models."""
        comparison_data = []
        
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot comparison of all models."""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Predictions vs Targets
        ax1 = axes[0, 0]
        for model_name, result in self.results.items():
            ax1.plot(result['targets'][:100], label=f'{model_name} - True', alpha=0.7)
            ax1.plot(result['predictions'][:100], label=f'{model_name} - Pred', alpha=0.7)
        ax1.set_title('Predictions vs Targets (First 100 samples)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Metrics comparison
        ax2 = axes[0, 1]
        metrics_df = self.get_comparison_table()
        metrics_to_plot = ['mse', 'mae', 'rmse', 'mape']
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / n_models
        
        for i, (model_name, result) in enumerate(self.results.items()):
            values = [result['metrics'][metric] for metric in metrics_to_plot]
            ax2.bar(x + i * width, values, width, label=model_name)
        
        ax2.set_title('Metrics Comparison')
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x + width * (n_models - 1) / 2)
        ax2.set_xticklabels(metrics_to_plot)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax3 = axes[1, 0]
        for model_name, result in self.results.items():
            errors = result['predictions'] - result['targets']
            ax3.hist(errors, bins=30, alpha=0.7, label=model_name)
        ax3.set_title('Error Distribution')
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Attention weights (if available)
        ax4 = axes[1, 1]
        for model_name, result in self.results.items():
            if result['attention_weights'] is not None:
                avg_attention = np.mean(result['attention_weights'], axis=0)
                ax4.plot(avg_attention, label=f'{model_name} Attention')
        ax4.set_title('Average Attention Weights')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Attention Weight')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()


def create_sequences(data: np.ndarray, targets: np.ndarray, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sequences for time series training.
    
    Args:
        data: Input time series data
        targets: Target values
        seq_len: Sequence length
        
    Returns:
        Tuple of (X, Y) tensors
    """
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(targets[i+seq_len])
    
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(Y)


def split_data(data: np.ndarray, train_split: float = 0.7, val_split: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data
        train_split: Training set proportion
        val_split: Validation set proportion
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n_samples = len(data)
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    from src.attention_mechanisms import AttentionRNN
    from src.data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
    
    # Generate synthetic data
    config = TimeSeriesConfig(n_samples=1000)
    generator = SyntheticTimeSeriesGenerator(config)
    ts, _ = generator.generate_time_series("complex")
    
    # Create sequences
    X, Y = create_sequences(ts, ts, seq_len=20)
    
    # Split data
    train_data, val_data, test_data = split_data(X.numpy())
    train_targets, val_targets, test_targets = split_data(Y.numpy())
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_targets))
    val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.FloatTensor(val_targets))
    test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_targets))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and trainer
    model = AttentionRNN(input_dim=1, hidden_dim=64, attention_type="temporal")
    trainer_config = {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        'gradient_clip_norm': 1.0
    }
    
    trainer = TimeSeriesTrainer(model, trainer_config)
    
    # Train model
    history = trainer.train(train_loader, val_loader, epochs=50)
    
    # Evaluate model
    results = trainer.evaluate(test_loader)
    print("Test Results:", results['metrics'])
