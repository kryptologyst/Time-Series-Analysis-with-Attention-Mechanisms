#!/usr/bin/env python3
"""
Setup script for Time Series Analysis with Attention Mechanisms.

This script sets up the project environment, creates necessary directories,
and provides initial configuration.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'data',
        'models',
        'logs',
        'plots',
        'output',
        'results',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def install_dependencies():
    """Install required dependencies."""
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    return True


def create_sample_config():
    """Create a sample configuration file if it doesn't exist."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.info("Creating sample configuration file...")
        config_content = """# Configuration file for Time Series Analysis with Attention Mechanisms

# Data configuration
data:
  synthetic:
    n_samples: 1000
    noise_level: 0.1
    trend_strength: 0.5
    seasonality_periods: [12, 24, 168]  # daily, weekly patterns
    anomaly_probability: 0.05
  
  preprocessing:
    train_split: 0.7
    val_split: 0.15
    test_split: 0.15
    sequence_length: 20
    normalize: true

# Model configuration
models:
  attention_rnn:
    input_dim: 1
    hidden_dim: 64
    num_layers: 2
    dropout: 0.2
    attention_type: "temporal"  # temporal, dot_product, additive, multi_head
  
  transformer:
    d_model: 64
    nhead: 4
    num_layers: 3
    dropout: 0.1
    sequence_length: 20

# Training configuration
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  weight_decay: 1e-5
  gradient_clip_norm: 1.0

# Evaluation configuration
evaluation:
  metrics: ["mse", "mae", "rmse", "mape", "smape"]
  forecast_horizon: 10
  confidence_intervals: [0.8, 0.95]

# Visualization configuration
visualization:
  figure_size: [12, 8]
  style: "seaborn-v0_8"
  color_palette: "husl"
  save_plots: true
  plot_format: "png"
  dpi: 300

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/timeseries_analysis.log"
  max_file_size: "10MB"
  backup_count: 5

# Paths
paths:
  data_dir: "data"
  models_dir: "models"
  logs_dir: "logs"
  plots_dir: "plots"
"""
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        logger.info("Sample configuration file created!")


def run_tests():
    """Run the test suite to verify installation."""
    try:
        logger.info("Running tests...")
        result = subprocess.run([sys.executable, 'tests/test_timeseries.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("All tests passed!")
            return True
        else:
            logger.warning(f"Some tests failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return False


def create_run_script():
    """Create a convenient run script."""
    run_script_content = """#!/bin/bash
# Convenient run script for Time Series Analysis

echo "Time Series Analysis with Attention Mechanisms"
echo "=============================================="
echo ""
echo "Available commands:"
echo "1. Run single model: python main.py --data-type complex --model-type attention_rnn --epochs 50"
echo "2. Compare models: python main.py --data-type energy --compare --epochs 30"
echo "3. Launch web interface: streamlit run streamlit_app.py"
echo "4. Run tests: python tests/test_timeseries.py"
echo "5. Open notebook: jupyter notebook notebooks/demo_notebook.ipynb"
echo ""
echo "For more options, run: python main.py --help"
"""
    
    with open("run.sh", 'w') as f:
        f.write(run_script_content)
    
    # Make it executable
    os.chmod("run.sh", 0o755)
    logger.info("Run script created!")


def main():
    """Main setup function."""
    logger.info("Setting up Time Series Analysis with Attention Mechanisms...")
    
    # Create directories
    create_directories()
    
    # Create sample configuration
    create_sample_config()
    
    # Create run script
    create_run_script()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Setup failed due to dependency installation issues")
        return False
    
    # Run tests
    if not run_tests():
        logger.warning("Some tests failed, but setup completed")
    
    logger.info("Setup completed successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run a quick test: python main.py --data-type complex --epochs 10")
    logger.info("2. Launch web interface: streamlit run streamlit_app.py")
    logger.info("3. Open the demo notebook: jupyter notebook notebooks/demo_notebook.ipynb")
    logger.info("4. Check the README.md for detailed usage instructions")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
