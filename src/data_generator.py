"""
Synthetic time series data generation module.

This module provides functions to generate realistic synthetic time series data
with various characteristics including trends, seasonality, noise, and anomalies.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """Configuration for synthetic time series generation."""
    n_samples: int = 1000
    noise_level: float = 0.1
    trend_strength: float = 0.5
    seasonality_periods: List[int] = None
    anomaly_probability: float = 0.05
    anomaly_magnitude: float = 3.0
    
    def __post_init__(self):
        if self.seasonality_periods is None:
            self.seasonality_periods = [12, 24, 168]  # daily, weekly patterns


class SyntheticTimeSeriesGenerator:
    """Generator for synthetic time series data with various patterns."""
    
    def __init__(self, config: TimeSeriesConfig, random_seed: int = 42):
        """
        Initialize the synthetic time series generator.
        
        Args:
            config: Configuration for time series generation
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_trend(self, n_samples: int) -> np.ndarray:
        """Generate a linear trend component."""
        return np.linspace(0, self.config.trend_strength, n_samples)
    
    def generate_seasonality(self, n_samples: int) -> np.ndarray:
        """Generate seasonal components."""
        seasonal_component = np.zeros(n_samples)
        
        for period in self.config.seasonality_periods:
            amplitude = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2 * np.pi)
            seasonal_component += amplitude * np.sin(2 * np.pi * np.arange(n_samples) / period + phase)
            
        return seasonal_component
    
    def generate_noise(self, n_samples: int) -> np.ndarray:
        """Generate Gaussian noise."""
        return np.random.normal(0, self.config.noise_level, n_samples)
    
    def generate_anomalies(self, n_samples: int) -> np.ndarray:
        """Generate anomaly points."""
        anomalies = np.zeros(n_samples)
        n_anomalies = int(n_samples * self.config.anomaly_probability)
        
        if n_anomalies > 0:
            anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
            anomaly_values = np.random.normal(0, self.config.anomaly_magnitude, n_anomalies)
            anomalies[anomaly_indices] = anomaly_values
            
        return anomalies
    
    def generate_complex_pattern(self, n_samples: int) -> np.ndarray:
        """Generate a complex pattern with multiple frequencies."""
        t = np.arange(n_samples)
        
        # Multiple frequency components
        pattern = (
            0.5 * np.sin(2 * np.pi * t / 50) +  # Long-term cycle
            0.3 * np.sin(2 * np.pi * t / 20) +  # Medium-term cycle
            0.2 * np.sin(2 * np.pi * t / 5)     # Short-term cycle
        )
        
        return pattern
    
    def generate_time_series(self, pattern_type: str = "complex") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate a complete time series with specified pattern.
        
        Args:
            pattern_type: Type of pattern to generate ("complex", "trend", "seasonal", "random")
            
        Returns:
            Tuple of (complete_time_series, components_dict)
        """
        n_samples = self.config.n_samples
        
        # Generate base components
        trend = self.generate_trend(n_samples)
        seasonality = self.generate_seasonality(n_samples)
        noise = self.generate_noise(n_samples)
        anomalies = self.generate_anomalies(n_samples)
        
        # Generate pattern-specific component
        if pattern_type == "complex":
            pattern = self.generate_complex_pattern(n_samples)
        elif pattern_type == "trend":
            pattern = trend
        elif pattern_type == "seasonal":
            pattern = seasonality
        elif pattern_type == "random":
            pattern = np.random.normal(0, 1, n_samples)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Combine all components
        time_series = pattern + seasonality + trend + noise + anomalies
        
        components = {
            "trend": trend,
            "seasonality": seasonality,
            "noise": noise,
            "anomalies": anomalies,
            "pattern": pattern,
            "total": time_series
        }
        
        logger.info(f"Generated {pattern_type} time series with {n_samples} samples")
        
        return time_series, components
    
    def generate_multiple_series(self, n_series: int = 5, pattern_types: Optional[List[str]] = None) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Generate multiple time series with different patterns.
        
        Args:
            n_series: Number of series to generate
            pattern_types: List of pattern types (if None, uses random selection)
            
        Returns:
            Dictionary mapping series names to (time_series, components) tuples
        """
        if pattern_types is None:
            pattern_types = ["complex", "trend", "seasonal", "random"]
        
        series_dict = {}
        
        for i in range(n_series):
            pattern_type = np.random.choice(pattern_types)
            series_name = f"series_{i+1}_{pattern_type}"
            
            # Use different random seeds for each series
            self.random_seed += 1
            np.random.seed(self.random_seed)
            
            time_series, components = self.generate_time_series(pattern_type)
            series_dict[series_name] = (time_series, components)
        
        logger.info(f"Generated {n_series} time series")
        return series_dict


def create_energy_demand_series(n_samples: int = 1000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate a realistic energy demand time series.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (time_series, components_dict)
    """
    config = TimeSeriesConfig(
        n_samples=n_samples,
        noise_level=0.05,
        trend_strength=0.3,
        seasonality_periods=[24, 168],  # daily and weekly patterns
        anomaly_probability=0.02
    )
    
    generator = SyntheticTimeSeriesGenerator(config)
    
    # Generate base components
    trend = generator.generate_trend(n_samples)
    seasonality = generator.generate_seasonality(n_samples)
    noise = generator.generate_noise(n_samples)
    anomalies = generator.generate_anomalies(n_samples)
    
    # Create realistic energy demand pattern
    t = np.arange(n_samples)
    base_demand = (
        50 +  # Base load
        20 * np.sin(2 * np.pi * t / 24) +  # Daily cycle
        10 * np.sin(2 * np.pi * t / 168) +  # Weekly cycle
        5 * np.sin(2 * np.pi * t / 8760)    # Yearly cycle (if hourly data)
    )
    
    time_series = base_demand + trend + seasonality + noise + anomalies
    
    components = {
        "base_demand": base_demand,
        "trend": trend,
        "seasonality": seasonality,
        "noise": noise,
        "anomalies": anomalies,
        "total": time_series
    }
    
    return time_series, components


def create_stock_price_series(n_samples: int = 1000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate a realistic stock price time series with random walk characteristics.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (time_series, components_dict)
    """
    config = TimeSeriesConfig(
        n_samples=n_samples,
        noise_level=0.02,
        trend_strength=0.1,
        seasonality_periods=[252],  # yearly pattern for daily data
        anomaly_probability=0.01
    )
    
    generator = SyntheticTimeSeriesGenerator(config)
    
    # Generate random walk with drift
    returns = np.random.normal(0.001, 0.02, n_samples)  # Daily returns
    price_series = 100 * np.exp(np.cumsum(returns))  # Starting price = 100
    
    # Add trend and anomalies
    trend = generator.generate_trend(n_samples) * 0.1
    anomalies = generator.generate_anomalies(n_samples) * 5
    
    time_series = price_series + trend + anomalies
    
    components = {
        "base_price": price_series,
        "returns": returns,
        "trend": trend,
        "anomalies": anomalies,
        "total": time_series
    }
    
    return time_series, components


if __name__ == "__main__":
    # Example usage
    config = TimeSeriesConfig(n_samples=500)
    generator = SyntheticTimeSeriesGenerator(config)
    
    # Generate a complex time series
    ts, components = generator.generate_time_series("complex")
    
    print(f"Generated time series with {len(ts)} samples")
    print(f"Mean: {np.mean(ts):.3f}, Std: {np.std(ts):.3f}")
    print(f"Min: {np.min(ts):.3f}, Max: {np.max(ts):.3f}")
