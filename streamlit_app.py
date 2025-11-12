"""
Streamlit web interface for time series analysis with attention mechanisms.

This module provides an interactive web interface for exploring time series data,
training models, and visualizing results.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import yaml
import logging
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import SyntheticTimeSeriesGenerator, TimeSeriesConfig, create_energy_demand_series, create_stock_price_series
from attention_mechanisms import AttentionRNN, TransformerTimeSeries
from trainer import TimeSeriesTrainer, create_sequences, split_data, MetricsCalculator
from visualization import TimeSeriesVisualizer, InteractiveVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis with Attention Mechanisms",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """Load configuration from YAML file."""
    try:
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
    
    # Default configuration
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
            }
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'early_stopping_patience': 10
        }
    }

def generate_synthetic_data(data_type: str, config: dict) -> tuple:
    """Generate synthetic time series data."""
    if data_type == "Complex Pattern":
        generator = SyntheticTimeSeriesGenerator(
            TimeSeriesConfig(**config['data']['synthetic'])
        )
        ts, components = generator.generate_time_series("complex")
    elif data_type == "Energy Demand":
        ts, components = create_energy_demand_series(config['data']['synthetic']['n_samples'])
    elif data_type == "Stock Price":
        ts, components = create_stock_price_series(config['data']['synthetic']['n_samples'])
    else:
        # Random walk
        np.random.seed(42)
        ts = np.cumsum(np.random.randn(config['data']['synthetic']['n_samples']))
        components = {'total': ts}
    
    return ts, components

def create_model(model_type: str, config: dict) -> torch.nn.Module:
    """Create a model based on the selected type."""
    if model_type == "Attention RNN":
        model_config = config['models']['attention_rnn']
        return AttentionRNN(**model_config)
    elif model_type == "Transformer":
        return TransformerTimeSeries(
            input_dim=1,
            d_model=64,
            n_heads=4,
            num_layers=3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model, train_loader, val_loader, config: dict, progress_bar):
    """Train the model with progress tracking."""
    trainer_config = config['training']
    trainer = TimeSeriesTrainer(model, trainer_config)
    
    epochs = trainer_config['epochs']
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        train_loss, _ = trainer.train_epoch(train_loader)
        val_loss, _ = trainer.validate_epoch(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        progress_bar.progress((epoch + 1) / epochs)
        
        # Early stopping check
        if epoch > 10 and val_loss > min(history['val_loss'][-10:]):
            break
    
    return trainer, history

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Time Series Analysis with Attention Mechanisms</h1>', unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
    
    # Data selection
    st.sidebar.subheader("Data Configuration")
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["Complex Pattern", "Energy Demand", "Stock Price", "Random Walk"]
    )
    
    n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
    noise_level = st.sidebar.slider("Noise Level", 0.01, 0.5, 0.1)
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 2.0, 0.5)
    
    # Model selection
    st.sidebar.subheader("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Attention RNN", "Transformer"]
    )
    
    attention_type = st.sidebar.selectbox(
        "Attention Type",
        ["temporal", "dot_product", "additive", "multi_head"]
    )
    
    hidden_dim = st.sidebar.slider("Hidden Dimension", 16, 128, 64)
    num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2)
    
    # Training configuration
    st.sidebar.subheader("Training Configuration")
    epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
    
    # Update config with sidebar values
    config['data']['synthetic']['n_samples'] = n_samples
    config['data']['synthetic']['noise_level'] = noise_level
    config['data']['synthetic']['trend_strength'] = trend_strength
    config['models']['attention_rnn']['hidden_dim'] = hidden_dim
    config['models']['attention_rnn']['num_layers'] = num_layers
    config['models']['attention_rnn']['attention_type'] = attention_type
    config['training']['epochs'] = epochs
    config['training']['learning_rate'] = learning_rate
    config['training']['batch_size'] = batch_size
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Data Generation", "Model Training", "Results", "Analysis"])
    
    with tab1:
        st.header("Data Generation and Exploration")
        
        if st.button("Generate New Data"):
            with st.spinner("Generating synthetic time series data..."):
                ts, components = generate_synthetic_data(data_type, config)
                
                # Store in session state
                st.session_state['time_series'] = ts
                st.session_state['components'] = components
                st.session_state['data_type'] = data_type
        
        if 'time_series' in st.session_state:
            ts = st.session_state['time_series']
            components = st.session_state['components']
            
            # Display data statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Length", len(ts))
            with col2:
                st.metric("Mean", f"{np.mean(ts):.3f}")
            with col3:
                st.metric("Std", f"{np.std(ts):.3f}")
            with col4:
                st.metric("Min", f"{np.min(ts):.3f}")
            
            # Plot time series
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(ts))),
                y=ts,
                mode='lines',
                name='Time Series',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"{data_type} Time Series",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot components if available
            if len(components) > 1:
                st.subheader("Time Series Components")
                
                component_names = [k for k in components.keys() if k != 'total']
                selected_components = st.multiselect(
                    "Select components to display",
                    component_names,
                    default=component_names[:3]
                )
                
                if selected_components:
                    fig_components = make_subplots(
                        rows=len(selected_components) + 1,
                        cols=1,
                        subplot_titles=selected_components + ["Total"]
                    )
                    
                    for i, comp_name in enumerate(selected_components):
                        fig_components.add_trace(
                            go.Scatter(
                                x=list(range(len(components[comp_name]))),
                                y=components[comp_name],
                                mode='lines',
                                name=comp_name
                            ),
                            row=i+1, col=1
                        )
                    
                    fig_components.add_trace(
                        go.Scatter(
                            x=list(range(len(ts))),
                            y=ts,
                            mode='lines',
                            name='Total',
                            line=dict(color='black', width=2)
                        ),
                        row=len(selected_components)+1, col=1
                    )
                    
                    fig_components.update_layout(height=300 * (len(selected_components) + 1))
                    st.plotly_chart(fig_components, use_container_width=True)
    
    with tab2:
        st.header("Model Training")
        
        if 'time_series' not in st.session_state:
            st.warning("Please generate data first in the Data Generation tab.")
        else:
            ts = st.session_state['time_series']
            
            if st.button("Start Training"):
                with st.spinner("Training model..."):
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
                    
                    train_dataset = TensorDataset(
                        torch.FloatTensor(train_data), 
                        torch.FloatTensor(train_targets)
                    )
                    val_dataset = TensorDataset(
                        torch.FloatTensor(val_data), 
                        torch.FloatTensor(val_targets)
                    )
                    test_dataset = TensorDataset(
                        torch.FloatTensor(test_data), 
                        torch.FloatTensor(test_targets)
                    )
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    
                    # Create and train model
                    model = create_model(model_type, config)
                    
                    progress_bar = st.progress(0)
                    trainer, history = train_model(model, train_loader, val_loader, config, progress_bar)
                    
                    # Evaluate model
                    results = trainer.evaluate(test_loader)
                    
                    # Store results in session state
                    st.session_state['model'] = model
                    st.session_state['trainer'] = trainer
                    st.session_state['history'] = history
                    st.session_state['results'] = results
                    st.session_state['test_loader'] = test_loader
                    
                    st.success("Training completed!")
                    
                    # Display training metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Train Loss", f"{history['train_loss'][-1]:.6f}")
                    with col2:
                        st.metric("Final Val Loss", f"{history['val_loss'][-1]:.6f}")
                    with col3:
                        st.metric("Test RMSE", f"{results['metrics']['rmse']:.6f}")
                    with col4:
                        st.metric("Test MAE", f"{results['metrics']['mae']:.6f}")
    
    with tab3:
        st.header("Results and Visualizations")
        
        if 'results' not in st.session_state:
            st.warning("Please train a model first in the Model Training tab.")
        else:
            results = st.session_state['results']
            history = st.session_state['history']
            
            # Training history plot
            st.subheader("Training History")
            
            fig_history = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Loss", "Validation Loss"]
            )
            
            fig_history.add_trace(
                go.Scatter(
                    x=list(range(len(history['train_loss']))),
                    y=history['train_loss'],
                    mode='lines',
                    name='Train Loss',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig_history.add_trace(
                go.Scatter(
                    x=list(range(len(history['val_loss']))),
                    y=history['val_loss'],
                    mode='lines',
                    name='Val Loss',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            fig_history.update_layout(height=400)
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Predictions plot
            st.subheader("Predictions vs Actual")
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=list(range(len(results['targets']))),
                y=results['targets'],
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            fig_pred.add_trace(go.Scatter(
                x=list(range(len(results['predictions']))),
                y=results['predictions'],
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2)
            ))
            
            fig_pred.update_layout(
                title="Predictions vs Actual Values",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Attention weights heatmap
            if 'attention_weights' in results and results['attention_weights'] is not None:
                st.subheader("Attention Weights")
                
                attention_weights = results['attention_weights']
                avg_attention = np.mean(attention_weights, axis=0)
                
                fig_attention = go.Figure(data=go.Heatmap(
                    z=attention_weights[:50],  # Show first 50 samples
                    colorscale='Blues',
                    showscale=True
                ))
                
                fig_attention.update_layout(
                    title="Attention Weights Heatmap",
                    xaxis_title="Time Step",
                    yaxis_title="Sample",
                    height=400
                )
                
                st.plotly_chart(fig_attention, use_container_width=True)
                
                # Average attention weights
                fig_avg_attention = go.Figure()
                fig_avg_attention.add_trace(go.Scatter(
                    x=list(range(len(avg_attention))),
                    y=avg_attention,
                    mode='lines+markers',
                    name='Average Attention',
                    line=dict(color='green', width=2)
                ))
                
                fig_avg_attention.update_layout(
                    title="Average Attention Weights",
                    xaxis_title="Time Step",
                    yaxis_title="Attention Weight",
                    height=300
                )
                
                st.plotly_chart(fig_avg_attention, use_container_width=True)
    
    with tab4:
        st.header("Analysis and Metrics")
        
        if 'results' not in st.session_state:
            st.warning("Please train a model first in the Model Training tab.")
        else:
            results = st.session_state['results']
            
            # Metrics table
            st.subheader("Evaluation Metrics")
            
            metrics_df = pd.DataFrame([results['metrics']])
            st.dataframe(metrics_df, use_container_width=True)
            
            # Error analysis
            st.subheader("Error Analysis")
            
            errors = results['predictions'] - results['targets']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Error distribution
                fig_error_dist = go.Figure()
                fig_error_dist.add_trace(go.Histogram(
                    x=errors,
                    nbinsx=50,
                    name='Error Distribution'
                ))
                
                fig_error_dist.update_layout(
                    title="Error Distribution",
                    xaxis_title="Prediction Error",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig_error_dist, use_container_width=True)
            
            with col2:
                # Actual vs Predicted scatter
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=results['targets'],
                    y=results['predictions'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', opacity=0.6)
                ))
                
                # Perfect prediction line
                min_val = min(np.min(results['targets']), np.min(results['predictions']))
                max_val = max(np.max(results['targets']), np.max(results['predictions']))
                
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_scatter.update_layout(
                    title="Actual vs Predicted",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values",
                    height=300
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Model information
            st.subheader("Model Information")
            
            if 'model' in st.session_state:
                model = st.session_state['model']
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Parameters", f"{total_params:,}")
                with col2:
                    st.metric("Trainable Parameters", f"{trainable_params:,}")
                with col3:
                    st.metric("Model Type", model_type)

if __name__ == "__main__":
    main()
