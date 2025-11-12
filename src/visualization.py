"""
Comprehensive visualization module for time series analysis.

This module provides various plotting functions for time series data,
model predictions, attention weights, and evaluation metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TimeSeriesVisualizer:
    """Comprehensive visualization class for time series analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Default DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_time_series(self, data: np.ndarray, title: str = "Time Series", 
                        xlabel: str = "Time", ylabel: str = "Value",
                        save_path: Optional[str] = None) -> None:
        """
        Plot a simple time series.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_time_series_components(self, components: Dict[str, np.ndarray], 
                                  title: str = "Time Series Components",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot time series components (trend, seasonality, noise, etc.).
        
        Args:
            components: Dictionary of component names and data
            title: Plot title
            save_path: Optional path to save the plot
        """
        n_components = len(components)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=(self.figsize[0], self.figsize[1] * (n_components + 1) / 3))
        
        if n_components == 1:
            axes = [axes]
        
        # Plot individual components
        for i, (name, data) in enumerate(components.items()):
            axes[i].plot(data, label=name)
            axes[i].set_title(f"{name.title()} Component")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        # Plot total
        if 'total' in components:
            axes[-1].plot(components['total'], label='Total', color='black', linewidth=2)
        else:
            total = sum(components.values())
            axes[-1].plot(total, label='Total', color='black', linewidth=2)
        
        axes[-1].set_title("Total Time Series")
        axes[-1].set_xlabel("Time")
        axes[-1].set_ylabel("Value")
        axes[-1].grid(True, alpha=0.3)
        axes[-1].legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Components plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = "Predictions vs Actual", 
                        xlabel: str = "Time", ylabel: str = "Value",
                        save_path: Optional[str] = None) -> None:
        """
        Plot predictions against actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot actual and predicted values
        plt.plot(y_true, label='Actual', alpha=0.8, linewidth=2)
        plt.plot(y_pred, label='Predicted', alpha=0.8, linewidth=2)
        
        # Add error shading
        error = np.abs(y_true - y_pred)
        plt.fill_between(range(len(y_true)), y_true - error, y_true + error, 
                         alpha=0.3, label='Error Range')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_attention_weights(self, attention_weights: np.ndarray,
                              title: str = "Attention Weights Heatmap",
                              save_path: Optional[str] = None) -> None:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights array [batch_size, seq_len]
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Average attention weights across batch
        avg_attention = np.mean(attention_weights, axis=0)
        
        # Create heatmap
        sns.heatmap(attention_weights[:50], cmap='Blues', cbar=True)
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Sample")
        
        # Add average attention line
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Average')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Attention weights plot saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = "Training History",
                             save_path: Optional[str] = None) -> None:
        """
        Plot training history (loss and metrics).
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot training metrics
        if history['train_metrics']:
            train_metrics = pd.DataFrame(history['train_metrics'])
            for metric in ['mse', 'mae', 'rmse']:
                if metric in train_metrics.columns:
                    axes[0, 1].plot(train_metrics[metric], label=f'Train {metric.upper()}')
            
            axes[0, 1].set_title('Training Metrics')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Metric Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot validation metrics
        if history['val_metrics']:
            val_metrics = pd.DataFrame(history['val_metrics'])
            for metric in ['mse', 'mae', 'rmse']:
                if metric in val_metrics.columns:
                    axes[1, 0].plot(val_metrics[metric], label=f'Val {metric.upper()}')
            
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Metric Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot MAPE if available
        if history['val_metrics'] and 'mape' in val_metrics.columns:
            axes[1, 1].plot(val_metrics['mape'], label='Validation MAPE', color='red')
            axes[1, 1].set_title('Mean Absolute Percentage Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                           title: str = "Error Analysis",
                           save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive error analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save the plot
        """
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error distribution
        axes[0, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error over time
        axes[1, 0].plot(errors, alpha=0.7)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Error Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Prediction Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Error analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]],
                             title: str = "Model Comparison",
                             save_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            results: Dictionary of model results
            title: Plot title
            save_path: Optional path to save the plot
        """
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Predictions comparison
        ax1 = axes[0, 0]
        for model_name, result in results.items():
            ax1.plot(result['predictions'][:100], label=f'{model_name} - Pred', alpha=0.8)
        ax1.plot(result['targets'][:100], label='True', color='black', linewidth=2)
        ax1.set_title('Predictions Comparison (First 100 samples)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Metrics comparison
        ax2 = axes[0, 1]
        metrics_data = []
        model_names = []
        for model_name, result in results.items():
            metrics_data.append(list(result['metrics'].values()))
            model_names.append(model_name)
        
        metrics_df = pd.DataFrame(metrics_data, index=model_names, 
                                 columns=list(results[list(results.keys())[0]]['metrics'].keys()))
        
        metrics_df.plot(kind='bar', ax=ax2)
        ax2.set_title('Metrics Comparison')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Metric Value')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error comparison
        ax3 = axes[1, 0]
        for model_name, result in results.items():
            errors = result['predictions'] - result['targets']
            ax3.hist(errors, bins=30, alpha=0.7, label=model_name)
        ax3.set_title('Error Distribution Comparison')
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Attention weights comparison
        ax4 = axes[1, 1]
        for model_name, result in results.items():
            if 'attention_weights' in result and result['attention_weights'] is not None:
                avg_attention = np.mean(result['attention_weights'], axis=0)
                ax4.plot(avg_attention, label=f'{model_name} Attention')
        ax4.set_title('Average Attention Weights Comparison')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Attention Weight')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()


class InteractiveVisualizer:
    """Interactive visualization using Plotly."""
    
    def __init__(self):
        """Initialize the interactive visualizer."""
        pass
    
    def create_interactive_time_series(self, data: np.ndarray, 
                                      title: str = "Interactive Time Series") -> go.Figure:
        """
        Create an interactive time series plot.
        
        Args:
            data: Time series data
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data,
            mode='lines',
            name='Time Series',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    
    def create_interactive_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      title: str = "Interactive Predictions") -> go.Figure:
        """
        Create an interactive predictions plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(y_true))),
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    
    def create_interactive_attention(self, attention_weights: np.ndarray,
                                    title: str = "Interactive Attention Weights") -> go.Figure:
        """
        Create an interactive attention weights heatmap.
        
        Args:
            attention_weights: Attention weights array
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights[:50],  # Show first 50 samples
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time Step',
            yaxis_title='Sample'
        )
        
        return fig


def create_dashboard_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create data structure for dashboard visualization.
    
    Args:
        results: Model results dictionary
        
    Returns:
        Dashboard data dictionary
    """
    dashboard_data = {
        'metrics': results.get('metrics', {}),
        'predictions': results.get('predictions', []),
        'targets': results.get('targets', []),
        'attention_weights': results.get('attention_weights', []),
        'training_history': results.get('training_history', {}),
        'model_info': results.get('model_info', {})
    }
    
    return dashboard_data


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    data = np.cumsum(np.random.randn(1000))
    
    # Initialize visualizer
    visualizer = TimeSeriesVisualizer()
    
    # Plot time series
    visualizer.plot_time_series(data, title="Sample Time Series")
    
    # Plot components
    components = {
        'trend': np.linspace(0, 10, 1000),
        'seasonality': 5 * np.sin(2 * np.pi * np.arange(1000) / 100),
        'noise': np.random.normal(0, 0.5, 1000),
        'total': data
    }
    visualizer.plot_time_series_components(components)
    
    # Generate sample predictions
    y_true = data[-100:]
    y_pred = y_true + np.random.normal(0, 1, 100)
    
    # Plot predictions
    visualizer.plot_predictions(y_true, y_pred)
    
    # Plot error analysis
    visualizer.plot_error_analysis(y_true, y_pred)
