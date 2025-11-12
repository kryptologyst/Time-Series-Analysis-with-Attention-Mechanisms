"""
Unit tests for time series analysis with attention mechanisms.

This module contains comprehensive tests for all components of the time series
analysis system including data generation, attention mechanisms, training, and evaluation.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import (
    SyntheticTimeSeriesGenerator, TimeSeriesConfig, 
    create_energy_demand_series, create_stock_price_series
)
from attention_mechanisms import (
    AttentionRNN, TransformerTimeSeries, DotProductAttention,
    AdditiveAttention, MultiHeadAttention, TemporalAttention
)
from trainer import (
    TimeSeriesTrainer, MetricsCalculator, EarlyStopping,
    create_sequences, split_data, ModelComparator
)
from visualization import TimeSeriesVisualizer


class TestDataGenerator(unittest.TestCase):
    """Test cases for data generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TimeSeriesConfig(
            n_samples=100,
            noise_level=0.1,
            trend_strength=0.5,
            seasonality_periods=[12, 24],
            anomaly_probability=0.05
        )
        self.generator = SyntheticTimeSeriesGenerator(self.config)
    
    def test_time_series_config(self):
        """Test TimeSeriesConfig initialization."""
        config = TimeSeriesConfig()
        self.assertEqual(config.n_samples, 1000)
        self.assertEqual(config.noise_level, 0.1)
        self.assertEqual(config.trend_strength, 0.5)
        self.assertEqual(config.seasonality_periods, [12, 24, 168])
    
    def test_synthetic_generator_initialization(self):
        """Test SyntheticTimeSeriesGenerator initialization."""
        self.assertIsInstance(self.generator, SyntheticTimeSeriesGenerator)
        self.assertEqual(self.generator.config.n_samples, 100)
    
    def test_generate_trend(self):
        """Test trend generation."""
        trend = self.generator.generate_trend(50)
        self.assertEqual(len(trend), 50)
        self.assertGreater(trend[-1], trend[0])  # Should be increasing
    
    def test_generate_seasonality(self):
        """Test seasonality generation."""
        seasonality = self.generator.generate_seasonality(100)
        self.assertEqual(len(seasonality), 100)
        self.assertIsInstance(seasonality, np.ndarray)
    
    def test_generate_noise(self):
        """Test noise generation."""
        noise = self.generator.generate_noise(100)
        self.assertEqual(len(noise), 100)
        self.assertAlmostEqual(np.mean(noise), 0, delta=0.1)
    
    def test_generate_anomalies(self):
        """Test anomaly generation."""
        anomalies = self.generator.generate_anomalies(100)
        self.assertEqual(len(anomalies), 100)
        self.assertGreaterEqual(np.sum(anomalies != 0), 0)
    
    def test_generate_time_series(self):
        """Test complete time series generation."""
        ts, components = self.generator.generate_time_series("complex")
        
        self.assertEqual(len(ts), 100)
        self.assertIsInstance(ts, np.ndarray)
        self.assertIsInstance(components, dict)
        self.assertIn('total', components)
        self.assertIn('trend', components)
        self.assertIn('seasonality', components)
        self.assertIn('noise', components)
    
    def test_generate_multiple_series(self):
        """Test multiple series generation."""
        series_dict = self.generator.generate_multiple_series(n_series=3)
        
        self.assertEqual(len(series_dict), 3)
        for series_name, (ts, components) in series_dict.items():
            self.assertIsInstance(ts, np.ndarray)
            self.assertIsInstance(components, dict)
    
    def test_create_energy_demand_series(self):
        """Test energy demand series generation."""
        ts, components = create_energy_demand_series(100)
        
        self.assertEqual(len(ts), 100)
        self.assertIsInstance(ts, np.ndarray)
        self.assertIsInstance(components, dict)
        self.assertIn('base_demand', components)
    
    def test_create_stock_price_series(self):
        """Test stock price series generation."""
        ts, components = create_stock_price_series(100)
        
        self.assertEqual(len(ts), 100)
        self.assertIsInstance(ts, np.ndarray)
        self.assertIsInstance(components, dict)
        self.assertIn('base_price', components)


class TestAttentionMechanisms(unittest.TestCase):
    """Test cases for attention mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 10
        self.hidden_dim = 16
        self.d_model = 16
        self.n_heads = 2
        
        # Create sample data
        self.x = torch.randn(self.batch_size, self.seq_len, 1)
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
    
    def test_dot_product_attention(self):
        """Test dot-product attention mechanism."""
        attention = DotProductAttention(self.hidden_dim)
        
        output, weights = attention(self.hidden_states, self.hidden_states, self.hidden_states)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        
        # Check that attention weights sum to 1
        weights_sum = torch.sum(weights, dim=-1)
        self.assertTrue(torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6))
    
    def test_additive_attention(self):
        """Test additive attention mechanism."""
        attention = AdditiveAttention(self.hidden_dim)
        
        output, weights = attention(self.hidden_states, self.hidden_states, self.hidden_states)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        attention = MultiHeadAttention(self.d_model, self.n_heads)
        
        output, weights = attention(self.hidden_states, self.hidden_states, self.hidden_states)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
    
    def test_temporal_attention(self):
        """Test temporal attention mechanism."""
        attention = TemporalAttention(self.hidden_dim)
        
        context_vector, weights = attention(self.hidden_states)
        
        self.assertEqual(context_vector.shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len))
        
        # Check that attention weights sum to 1
        weights_sum = torch.sum(weights, dim=-1)
        self.assertTrue(torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6))
    
    def test_attention_rnn(self):
        """Test AttentionRNN model."""
        model = AttentionRNN(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            attention_type="temporal"
        )
        
        output, attention_weights = model(self.x)
        
        self.assertEqual(output.shape, (self.batch_size,))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
    
    def test_transformer_time_series(self):
        """Test TransformerTimeSeries model."""
        model = TransformerTimeSeries(
            input_dim=1,
            d_model=self.d_model,
            n_heads=self.n_heads
        )
        
        output, attention_weights = model(self.x)
        
        self.assertEqual(output.shape, (self.batch_size,))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))


class TestTrainer(unittest.TestCase):
    """Test cases for trainer module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 10
        self.n_samples = 100
        
        # Create sample data
        self.data = np.random.randn(self.n_samples)
        self.X, self.Y = create_sequences(self.data, self.data, self.seq_len)
        
        # Create data loaders
        dataset = TensorDataset(self.X, self.Y)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create model
        self.model = AttentionRNN(input_dim=1, hidden_dim=16)
        
        # Create trainer
        self.trainer_config = {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'gradient_clip_norm': 1.0
        }
        self.trainer = TimeSeriesTrainer(self.model, self.trainer_config)
    
    def test_create_sequences(self):
        """Test sequence creation."""
        X, Y = create_sequences(self.data, self.data, self.seq_len)
        
        self.assertEqual(X.shape, (self.n_samples - self.seq_len, self.seq_len, 1))
        self.assertEqual(Y.shape, (self.n_samples - self.seq_len,))
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(Y, torch.Tensor)
    
    def test_split_data(self):
        """Test data splitting."""
        train_data, val_data, test_data = split_data(self.data, 0.6, 0.2)
        
        total_len = len(train_data) + len(val_data) + len(test_data)
        self.assertEqual(total_len, len(self.data))
        self.assertGreater(len(train_data), len(val_data))
        self.assertGreater(len(train_data), len(test_data))
    
    def test_metrics_calculator(self):
        """Test metrics calculator."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('smape', metrics)
        
        # Check that all metrics are positive
        for metric, value in metrics.items():
            self.assertGreaterEqual(value, 0)
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        early_stopping = EarlyStopping(patience=3)
        
        # Simulate decreasing loss
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        
        for i, loss in enumerate(losses):
            should_stop = early_stopping(loss, self.model)
            if i < len(losses) - 1:  # Not the last iteration
                self.assertFalse(should_stop)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, TimeSeriesTrainer)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.trainer.loss_fn, nn.MSELoss)
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        train_loss, train_metrics = self.trainer.train_epoch(self.loader)
        
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_metrics, dict)
        self.assertGreaterEqual(train_loss, 0)
    
    def test_validate_epoch(self):
        """Test validation for one epoch."""
        val_loss, val_metrics = self.trainer.validate_epoch(self.loader)
        
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_metrics, dict)
        self.assertGreaterEqual(val_loss, 0)
    
    def test_evaluate(self):
        """Test model evaluation."""
        results = self.trainer.evaluate(self.loader)
        
        self.assertIn('metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('targets', results)
        self.assertIn('attention_weights', results)
        
        self.assertEqual(len(results['predictions']), len(results['targets']))


class TestVisualization(unittest.TestCase):
    """Test cases for visualization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = TimeSeriesVisualizer()
        self.n_samples = 100
        
        # Create sample data
        self.data = np.random.randn(self.n_samples)
        self.components = {
            'trend': np.linspace(0, 1, self.n_samples),
            'seasonality': np.sin(np.linspace(0, 4*np.pi, self.n_samples)),
            'noise': np.random.normal(0, 0.1, self.n_samples),
            'total': self.data
        }
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        self.assertIsInstance(self.visualizer, TimeSeriesVisualizer)
        self.assertEqual(self.visualizer.figsize, (12, 8))
        self.assertEqual(self.visualizer.dpi, 300)
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        # This test just checks that the method doesn't raise an exception
        # In a real test environment, you might want to check the plot output
        try:
            self.visualizer.plot_time_series(self.data, title="Test Plot")
        except Exception as e:
            self.fail(f"plot_time_series raised an exception: {e}")
    
    def test_plot_time_series_components(self):
        """Test time series components plotting."""
        try:
            self.visualizer.plot_time_series_components(self.components, title="Test Components")
        except Exception as e:
            self.fail(f"plot_time_series_components raised an exception: {e}")
    
    def test_plot_predictions(self):
        """Test predictions plotting."""
        y_true = self.data[:50]
        y_pred = y_true + np.random.normal(0, 0.1, 50)
        
        try:
            self.visualizer.plot_predictions(y_true, y_pred, title="Test Predictions")
        except Exception as e:
            self.fail(f"plot_predictions raised an exception: {e}")
    
    def test_plot_attention_weights(self):
        """Test attention weights plotting."""
        attention_weights = np.random.rand(10, 20)  # 10 samples, 20 time steps
        
        try:
            self.visualizer.plot_attention_weights(attention_weights, title="Test Attention")
        except Exception as e:
            self.fail(f"plot_attention_weights raised an exception: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TimeSeriesConfig(n_samples=200, noise_level=0.1)
        self.generator = SyntheticTimeSeriesGenerator(self.config)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        ts, components = self.generator.generate_time_series("complex")
        
        # Create sequences
        X, Y = create_sequences(ts, ts, seq_len=10)
        
        # Split data
        train_data, val_data, test_data = split_data(X.numpy(), 0.7, 0.15)
        train_targets, val_targets, test_targets = split_data(Y.numpy(), 0.7, 0.15)
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_targets))
        val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.FloatTensor(val_targets))
        test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_targets))
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Create and train model
        model = AttentionRNN(input_dim=1, hidden_dim=16, attention_type="temporal")
        trainer_config = {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5
        }
        trainer = TimeSeriesTrainer(model, trainer_config)
        
        # Train for a few epochs
        history = trainer.train(train_loader, val_loader, epochs=5)
        
        # Evaluate
        results = trainer.evaluate(test_loader)
        
        # Check results
        self.assertIn('metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('targets', results)
        self.assertIn('attention_weights', results)
        
        # Check that metrics are reasonable
        metrics = results['metrics']
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Generate data
        ts, _ = self.generator.generate_time_series("complex")
        
        # Create sequences
        X, Y = create_sequences(ts, ts, seq_len=10)
        
        # Split data
        train_data, val_data, test_data = split_data(X.numpy(), 0.7, 0.15)
        train_targets, val_targets, test_targets = split_data(Y.numpy(), 0.7, 0.15)
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.FloatTensor(train_targets))
        val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.FloatTensor(val_targets))
        test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.FloatTensor(test_targets))
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Create models
        models = {
            'attention_rnn': AttentionRNN(input_dim=1, hidden_dim=16, attention_type="temporal"),
            'transformer': TransformerTimeSeries(input_dim=1, d_model=16, n_heads=2)
        }
        
        # Train and compare models
        comparator = ModelComparator()
        
        for model_name, model in models.items():
            trainer_config = {
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'early_stopping_patience': 5
            }
            trainer = TimeSeriesTrainer(model, trainer_config)
            
            # Train for a few epochs
            trainer.train(train_loader, val_loader, epochs=5)
            
            # Evaluate
            results = trainer.evaluate(test_loader)
            
            # Add to comparator
            comparator.add_model_result(
                model_name,
                results['predictions'],
                results['targets'],
                results['attention_weights']
            )
        
        # Get comparison table
        comparison_table = comparator.get_comparison_table()
        
        self.assertIsInstance(comparison_table, pd.DataFrame)
        self.assertEqual(len(comparison_table), 2)  # Two models
        self.assertIn('Model', comparison_table.columns)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataGenerator,
        TestAttentionMechanisms,
        TestTrainer,
        TestVisualization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
