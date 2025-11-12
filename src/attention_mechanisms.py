"""
Enhanced attention mechanisms for time series analysis.

This module implements various attention mechanisms including:
- Dot-product attention
- Additive attention
- Multi-head attention
- Self-attention
- Cross-attention
- Temporal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class DotProductAttention(nn.Module):
    """Dot-product attention mechanism."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize dot-product attention.
        
        Args:
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of dot-product attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_dim]
            key: Key tensor [batch_size, seq_len, hidden_dim]
            value: Value tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class AdditiveAttention(nn.Module):
    """Additive attention mechanism (Bahdanau attention)."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize additive attention.
        
        Args:
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of additive attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_dim]
            key: Key tensor [batch_size, seq_len, hidden_dim]
            value: Value tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, hidden_dim = query.size()
        
        # Expand query for broadcasting
        query_expanded = query.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_dim)
        key_expanded = key.unsqueeze(1).expand(batch_size, seq_len, seq_len, hidden_dim)
        
        # Concatenate query and key
        concat = torch.cat([query_expanded, key_expanded], dim=-1)
        
        # Compute attention scores
        scores = self.v(torch.tanh(self.attention(concat))).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear transformation
        output = self.w_o(context)
        
        return output, attention_weights.mean(dim=1)  # Average attention weights across heads


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time series."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of temporal attention.
        
        Args:
            hidden_states: Hidden states from RNN [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted context vector
        context_vector = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        
        return context_vector, attention_weights


class SelfAttention(nn.Module):
    """Self-attention mechanism."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initialize self-attention.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """Cross-attention mechanism."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initialize cross-attention.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key_value: Key-value tensor [batch_size, seq_len_kv, d_model]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len_q, d_model = query.size()
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key_value)
        V = self.w_v(key_value)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class AttentionRNN(nn.Module):
    """RNN with attention mechanism for time series forecasting."""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2,
                 attention_type: str = "temporal", dropout: float = 0.2, 
                 rnn_type: str = "GRU"):
        """
        Initialize attention-based RNN.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            attention_type: Type of attention mechanism
            dropout: Dropout rate
            rnn_type: Type of RNN ("GRU", "LSTM", "RNN")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.rnn_type = rnn_type
        
        # RNN layer
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        if attention_type == "temporal":
            self.attention = TemporalAttention(hidden_dim, dropout)
        elif attention_type == "dot_product":
            self.attention = DotProductAttention(hidden_dim, dropout)
        elif attention_type == "additive":
            self.attention = AdditiveAttention(hidden_dim, dropout)
        elif attention_type == "multi_head":
            self.attention = MultiHeadAttention(hidden_dim, n_heads=4, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention-based RNN.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply attention
        if self.attention_type == "temporal":
            context_vector, attention_weights = self.attention(rnn_out)
        else:
            # For other attention types, use self-attention
            context_vector, attention_weights = self.attention(rnn_out, rnn_out, rnn_out)
            context_vector = context_vector.mean(dim=1)  # Average over sequence length
        
        # Apply dropout and final linear layer
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector).squeeze(-1)
        
        return output, attention_weights


class TransformerTimeSeries(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_dim: int = 1, d_model: int = 64, n_heads: int = 4,
                 num_layers: int = 3, dropout: float = 0.1, max_len: int = 1000):
        """
        Initialize transformer for time series.
        
        Args:
            input_dim: Input dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer forward pass
        transformer_out = self.transformer(x, mask)
        
        # Global average pooling
        context_vector = transformer_out.mean(dim=1)  # [batch_size, d_model]
        
        # Output projection
        context_vector = self.dropout(context_vector)
        output = self.output_projection(context_vector).squeeze(-1)
        
        # Extract attention weights from the last layer
        attention_weights = torch.ones(batch_size, seq_len) / seq_len  # Placeholder
        
        return output, attention_weights


if __name__ == "__main__":
    # Example usage
    batch_size, seq_len, input_dim = 32, 20, 1
    
    # Test AttentionRNN
    model = AttentionRNN(input_dim=input_dim, hidden_dim=64, attention_type="temporal")
    x = torch.randn(batch_size, seq_len, input_dim)
    output, attention_weights = model(x)
    
    print(f"AttentionRNN output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test Transformer
    transformer = TransformerTimeSeries(input_dim=input_dim, d_model=64, n_heads=4)
    output, attention_weights = transformer(x)
    
    print(f"Transformer output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
