# Time Series Analysis with Attention Mechanisms

A comprehensive Python project for time series analysis using state-of-the-art attention mechanisms. This project implements various attention-based models for time series forecasting, including RNNs with attention, Transformers, and multiple attention mechanisms.

## Features

- **Multiple Attention Mechanisms**: Dot-product, additive, multi-head, temporal, self-attention, and cross-attention
- **Advanced Models**: Attention-based RNNs (GRU/LSTM), Transformer models
- **Synthetic Data Generation**: Realistic time series with trends, seasonality, noise, and anomalies
- **Comprehensive Evaluation**: Multiple metrics (MSE, MAE, RMSE, MAPE, SMAPE)
- **Interactive Visualization**: Streamlit web interface and comprehensive plotting
- **Model Comparison**: Side-by-side comparison of different models
- **Configuration Management**: YAML-based configuration system
- **Logging and Checkpointing**: Full logging and model saving/loading
- **Unit Tests**: Comprehensive test suite for all components

## Project Structure

```
├── src/                          # Source code
│   ├── data_generator.py         # Synthetic data generation
│   ├── attention_mechanisms.py   # Attention mechanisms and models
│   ├── trainer.py               # Training and evaluation
│   └── visualization.py         # Plotting and visualization
├── config/                      # Configuration files
│   └── config.yaml              # Main configuration
├── tests/                       # Unit tests
│   └── test_timeseries.py       # Test suite
├── data/                        # Data storage
├── models/                      # Saved models
├── logs/                        # Log files
├── plots/                       # Generated plots
├── notebooks/                   # Jupyter notebooks
├── requirements.txt             # Dependencies
├── main.py                     # Command-line interface
├── streamlit_app.py            # Web interface
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Time-Series-Analysis-with-Attention-Mechanisms.git
cd Time-Series-Analysis-with-Attention-Mechanisms
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

Run a single model experiment:
```bash
python main.py --data-type complex --model-type attention_rnn --epochs 50
```

Compare multiple models:
```bash
python main.py --data-type energy --compare --epochs 30
```

Generate energy demand data with transformer model:
```bash
python main.py --data-type energy --model-type transformer --epochs 100
```

### Web Interface

Launch the Streamlit web interface:
```bash
streamlit run streamlit_app.py
```

The web interface provides:
- Interactive data generation and visualization
- Model configuration and training
- Real-time results and metrics
- Attention weight visualization
- Error analysis and model comparison

## Usage Examples

### Basic Usage

```python
from src.data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig
from src.attention_mechanisms import AttentionRNN
from src.trainer import TimeSeriesTrainer, create_sequences, split_data

# Generate synthetic data
config = TimeSeriesConfig(n_samples=1000, noise_level=0.1)
generator = SyntheticTimeSeriesGenerator(config)
ts, components = generator.generate_time_series("complex")

# Create sequences
X, Y = create_sequences(ts, ts, seq_len=20)

# Split data
train_data, val_data, test_data = split_data(X.numpy(), 0.7, 0.15)
train_targets, val_targets, test_targets = split_data(Y.numpy(), 0.7, 0.15)

# Create model
model = AttentionRNN(input_dim=1, hidden_dim=64, attention_type="temporal")

# Train model
trainer_config = {
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 10
}
trainer = TimeSeriesTrainer(model, trainer_config)

# Train and evaluate
history = trainer.train(train_loader, val_loader, epochs=50)
results = trainer.evaluate(test_loader)
```

### Advanced Usage

```python
from src.attention_mechanisms import TransformerTimeSeries, MultiHeadAttention
from src.visualization import TimeSeriesVisualizer

# Create transformer model
transformer = TransformerTimeSeries(
    input_dim=1,
    d_model=64,
    n_heads=4,
    num_layers=3
)

# Train transformer
trainer = TimeSeriesTrainer(transformer, trainer_config)
history = trainer.train(train_loader, val_loader, epochs=100)

# Visualize results
visualizer = TimeSeriesVisualizer()
visualizer.plot_training_history(history)
visualizer.plot_predictions(results['targets'], results['predictions'])
visualizer.plot_attention_weights(results['attention_weights'])
```

## Configuration

The project uses YAML configuration files for easy customization. Key configuration options:

```yaml
# Data configuration
data:
  synthetic:
    n_samples: 1000
    noise_level: 0.1
    trend_strength: 0.5
    seasonality_periods: [12, 24, 168]
    anomaly_probability: 0.05

# Model configuration
models:
  attention_rnn:
    input_dim: 1
    hidden_dim: 64
    num_layers: 2
    dropout: 0.2
    attention_type: "temporal"

# Training configuration
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  weight_decay: 1e-5
```

## Available Models

### Attention Mechanisms
- **Dot-Product Attention**: Standard scaled dot-product attention
- **Additive Attention**: Bahdanau-style additive attention
- **Multi-Head Attention**: Multi-head self-attention mechanism
- **Temporal Attention**: Time-series specific attention
- **Self-Attention**: Self-attention for sequence modeling
- **Cross-Attention**: Cross-attention between sequences

### Model Architectures
- **AttentionRNN**: RNN (GRU/LSTM) with attention mechanism
- **TransformerTimeSeries**: Transformer model for time series

## Data Types

### Synthetic Data
- **Complex Pattern**: Multi-frequency patterns with trends and seasonality
- **Energy Demand**: Realistic energy consumption patterns
- **Stock Price**: Random walk with drift and anomalies
- **Custom Patterns**: Configurable synthetic data generation

## Evaluation Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/ -v
```

Or run specific test modules:

```bash
python tests/test_timeseries.py
```

## Logging

The project includes comprehensive logging:
- Training progress and metrics
- Model performance
- Error tracking
- Configuration logging

Logs are saved to `logs/timeseries_analysis.log` and displayed in the console.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit
- PyYAML

See `requirements.txt` for complete dependency list.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{timeseries_attention,
  title={Time Series Analysis with Attention Mechanisms},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Time-Series-Analysis-with-Attention-Mechanisms}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework
- The time series analysis community for inspiration and best practices

## Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the maintainers

## Changelog

### Version 1.0.0
- Initial release
- Multiple attention mechanisms
- RNN and Transformer models
- Synthetic data generation
- Web interface
- Comprehensive testing
- Documentation
# Time-Series-Analysis-with-Attention-Mechanisms
