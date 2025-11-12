#!/usr/bin/env python3
"""
Main execution script for Time Series Analysis with Attention Mechanisms.

This script provides a command-line interface for running the time series analysis
with various attention mechanisms and models.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig, create_energy_demand_series, create_stock_price_series
from attention_mechanisms import AttentionRNN, TransformerTimeSeries
from trainer import TimeSeriesTrainer, create_sequences, split_data, ModelComparator
from visualization import TimeSeriesVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/timeseries_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        'data': {
            'synthetic': {
                'n_samples': 1000,
                'noise_level': 0.1,
                'trend_strength': 0.5,
                'seasonality_periods': [12, 24, 168],
                'anomaly_probability': 0.05
            },
            'preprocessing': {
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'sequence_length': 20,
                'normalize': True
            }
        },
        'models': {
            'attention_rnn': {
                'input_dim': 1,
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'attention_type': 'temporal'
            },
            'transformer': {
                'input_dim': 1,
                'd_model': 64,
                'n_heads': 4,
                'num_layers': 3,
                'dropout': 0.1
            }
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'early_stopping_patience': 10,
            'weight_decay': 1e-5,
            'gradient_clip_norm': 1.0
        },
        'visualization': {
            'figure_size': [12, 8],
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300
        }
    }


def generate_data(data_type: str, config: dict) -> tuple:
    """Generate synthetic time series data."""
    logger.info(f"Generating {data_type} time series data")
    
    if data_type == "complex":
        generator = SyntheticTimeSeriesGenerator(
            TimeSeriesConfig(**config['data']['synthetic'])
        )
        ts, components = generator.generate_time_series("complex")
    elif data_type == "energy":
        ts, components = create_energy_demand_series(config['data']['synthetic']['n_samples'])
    elif data_type == "stock":
        ts, components = create_stock_price_series(config['data']['synthetic']['n_samples'])
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    logger.info(f"Generated time series with {len(ts)} samples")
    return ts, components


def create_model(model_type: str, config: dict) -> torch.nn.Module:
    """Create a model based on the selected type."""
    logger.info(f"Creating {model_type} model")
    
    if model_type == "attention_rnn":
        model_config = config['models']['attention_rnn']
        return AttentionRNN(**model_config)
    elif model_type == "transformer":
        model_config = config['models']['transformer']
        return TransformerTimeSeries(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(model, train_loader, val_loader, test_loader, config: dict) -> dict:
    """Train and evaluate a model."""
    logger.info("Starting model training")
    
    # Create trainer
    trainer_config = config['training']
    trainer = TimeSeriesTrainer(model, trainer_config)
    
    # Train model
    history = trainer.train(train_loader, val_loader, trainer_config['epochs'])
    
    # Evaluate model
    results = trainer.evaluate(test_loader)
    
    logger.info("Training and evaluation completed")
    return {
        'trainer': trainer,
        'history': history,
        'results': results
    }


def run_single_model(data_type: str, model_type: str, config: dict) -> dict:
    """Run a single model experiment."""
    logger.info(f"Running experiment: {data_type} data with {model_type} model")
    
    # Generate data
    ts, components = generate_data(data_type, config)
    
    # Create sequences
    X, Y = create_sequences(ts, ts, config['data']['preprocessing']['sequence_length'])
    
    # Split data
    train_data, val_data, test_data = split_data(
        X.numpy(),
        config['data']['preprocessing']['train_split'],
        config['data']['preprocessing']['val_split']
    )
    train_targets, val_targets, test_targets = split_data(
        Y.numpy(),
        config['data']['preprocessing']['train_split'],
        config['data']['preprocessing']['val_split']
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_targets))
    val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.FloatTensor(val_targets))
    test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_targets))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Create and train model
    model = create_model(model_type, config)
    results = train_and_evaluate(model, train_loader, val_loader, test_loader, config)
    
    return {
        'data': {'ts': ts, 'components': components},
        'model_type': model_type,
        'results': results
    }


def run_model_comparison(data_type: str, config: dict) -> dict:
    """Run comparison of multiple models."""
    logger.info(f"Running model comparison on {data_type} data")
    
    model_types = ["attention_rnn", "transformer"]
    results = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        result = run_single_model(data_type, model_type, config)
        results[model_type] = result['results']['results']
    
    return results


def create_visualizations(results: dict, config: dict, output_dir: str = "plots"):
    """Create comprehensive visualizations."""
    logger.info("Creating visualizations")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    visualizer = TimeSeriesVisualizer(
        figsize=tuple(config['visualization']['figure_size']),
        dpi=config['visualization']['dpi']
    )
    
    # Plot time series and components
    if 'data' in results:
        ts = results['data']['ts']
        components = results['data']['components']
        
        visualizer.plot_time_series(
            ts, 
            title=f"{results['model_type'].title()} - Time Series",
            save_path=f"{output_dir}/time_series.png"
        )
        
        visualizer.plot_time_series_components(
            components,
            title=f"{results['model_type'].title()} - Components",
            save_path=f"{output_dir}/components.png"
        )
    
    # Plot training history
    if 'results' in results:
        history = results['results']['history']
        visualizer.plot_training_history(
            history,
            title=f"{results['model_type'].title()} - Training History",
            save_path=f"{output_dir}/training_history.png"
        )
        
        # Plot predictions
        eval_results = results['results']['results']
        visualizer.plot_predictions(
            eval_results['targets'],
            eval_results['predictions'],
            title=f"{results['model_type'].title()} - Predictions",
            save_path=f"{output_dir}/predictions.png"
        )
        
        # Plot attention weights
        if 'attention_weights' in eval_results and eval_results['attention_weights'] is not None:
            visualizer.plot_attention_weights(
                eval_results['attention_weights'],
                title=f"{results['model_type'].title()} - Attention Weights",
                save_path=f"{output_dir}/attention_weights.png"
            )
        
        # Plot error analysis
        visualizer.plot_error_analysis(
            eval_results['targets'],
            eval_results['predictions'],
            title=f"{results['model_type'].title()} - Error Analysis",
            save_path=f"{output_dir}/error_analysis.png"
        )


def save_results(results: dict, output_dir: str = "results"):
    """Save results to files."""
    logger.info("Saving results")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save metrics
    if 'results' in results and 'results' in results['results']:
        metrics = results['results']['results']['metrics']
        
        import json
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_dir}/metrics.json")
    
    # Save model
    if 'results' in results and 'trainer' in results['results']:
        trainer = results['results']['trainer']
        trainer.save_model(f"{output_dir}/model.pth")
        logger.info(f"Model saved to {output_dir}/model.pth")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Time Series Analysis with Attention Mechanisms")
    parser.add_argument("--data-type", choices=["complex", "energy", "stock"], 
                       default="complex", help="Type of data to generate")
    parser.add_argument("--model-type", choices=["attention_rnn", "transformer"], 
                       default="attention_rnn", help="Type of model to use")
    parser.add_argument("--config", default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare multiple models")
    parser.add_argument("--output-dir", default="output", 
                       help="Output directory for results")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip generating plots")
    parser.add_argument("--epochs", type=int, 
                       help="Number of training epochs (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    try:
        if args.compare:
            # Run model comparison
            results = run_model_comparison(args.data_type, config)
            
            # Create comparison visualizations
            if not args.no_plots:
                visualizer = TimeSeriesVisualizer()
                visualizer.plot_model_comparison(
                    results,
                    title=f"Model Comparison on {args.data_type.title()} Data",
                    save_path=f"{args.output_dir}/model_comparison.png"
                )
            
            # Print comparison results
            print("\nModel Comparison Results:")
            print("=" * 50)
            for model_name, result in results.items():
                print(f"\n{model_name.upper()}:")
                for metric, value in result['metrics'].items():
                    print(f"  {metric.upper()}: {value:.6f}")
        
        else:
            # Run single model
            results = run_single_model(args.data_type, args.model_type, config)
            
            # Create visualizations
            if not args.no_plots:
                create_visualizations(results, config, f"{args.output_dir}/plots")
            
            # Save results
            save_results(results, f"{args.output_dir}/results")
            
            # Print results
            print(f"\n{args.model_type.upper()} Results:")
            print("=" * 50)
            metrics = results['results']['results']['metrics']
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value:.6f}")
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
