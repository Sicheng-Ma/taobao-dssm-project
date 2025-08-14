"""
Sequence Transformer for user behavior sequence encoding.

This module implements a Transformer-based encoder for processing user behavior
sequences to extract rich temporal patterns and user preferences.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention mechanism for sequence encoding."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        output = self.dropout(output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        
        return output, attention_weights


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for sequence positions."""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and feed forward network."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
    
    def call(self, x, training, mask=None):
        attn_output, _ = self.att(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2


class SequenceTransformer(tf.keras.layers.Layer):
    """
    Transformer-based encoder for user behavior sequences.
    
    This layer processes user behavior sequences (e.g., item interactions, 
    search queries, category browsing) and outputs a fixed-dimensional 
    representation vector.
    """
    
    def __init__(self, 
                 max_seq_length: int = 50,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dff: int = 512,
                 dropout: float = 0.1,
                 output_dim: int = 64,
                 feature_dim: int = 16):
        """
        Initialize the SequenceTransformer.
        
        Args:
            max_seq_length: Maximum sequence length
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dff: Feed forward network dimension
            dropout: Dropout rate
            output_dim: Output embedding dimension
            feature_dim: Input feature dimension
        """
        super(SequenceTransformer, self).__init__()
        
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = tf.keras.layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_length, d_model)
        
        # Transformer layers
        self.transformer_layers = [
            TransformerBlock(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ]
        
        # Output projection
        self.output_projection = tf.keras.layers.Dense(output_dim)
        
        # Global pooling
        self.global_pooling = tf.keras.layers.GlobalAveragePooling1D()
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        logger.info(f"Initialized SequenceTransformer with {num_layers} layers, "
                   f"d_model={d_model}, num_heads={num_heads}")
    
    def create_padding_mask(self, seq):
        """Create padding mask for sequences."""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def call(self, sequence_features, training=None):
        """
        Process user behavior sequence.
        
        Args:
            sequence_features: Input sequence features of shape (batch_size, seq_len, feature_dim)
            training: Whether in training mode
            
        Returns:
            sequence_embedding: Fixed-dimensional sequence representation
        """
        batch_size = tf.shape(sequence_features)[0]
        seq_len = tf.shape(sequence_features)[1]
        
        # Input projection
        x = self.input_projection(sequence_features)  # (batch_size, seq_len, d_model)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Create padding mask
        padding_mask = self.create_padding_mask(sequence_features[:, :, 0])
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training, padding_mask)
        
        # Global pooling
        pooled = self.global_pooling(x)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_projection(pooled)  # (batch_size, output_dim)
        
        return output
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model,
            'output_dim': self.output_dim
        })
        return config


class UserBehaviorProcessor:
    """
    Utility class for processing user behavior data into sequences.
    """
    
    def __init__(self, max_seq_length: int = 50):
        self.max_seq_length = max_seq_length
    
    def create_behavior_sequence(self, 
                                user_behaviors: List[Dict],
                                behavior_types: List[str] = None) -> np.ndarray:
        """
        Create behavior sequence from user interaction data.
        
        Args:
            user_behaviors: List of user behavior dictionaries
            behavior_types: Types of behaviors to include
            
        Returns:
            sequence: Padded sequence array
        """
        if behavior_types is None:
            behavior_types = ['item_id', 'category_id', 'brand_id', 'price']
        
        sequence = []
        for behavior in user_behaviors[-self.max_seq_length:]:  # Keep recent behaviors
            features = []
            for behavior_type in behavior_types:
                if behavior_type in behavior:
                    features.append(float(behavior[behavior_type]))
                else:
                    features.append(0.0)
            sequence.append(features)
        
        # Pad sequence
        while len(sequence) < self.max_seq_length:
            sequence.append([0.0] * len(behavior_types))
        
        return np.array(sequence)
    
    def batch_create_sequences(self, 
                              user_behaviors_batch: List[List[Dict]],
                              behavior_types: List[str] = None) -> np.ndarray:
        """
        Create sequences for a batch of users.
        
        Args:
            user_behaviors_batch: List of user behavior lists
            behavior_types: Types of behaviors to include
            
        Returns:
            sequences: Batch of padded sequences
        """
        sequences = []
        for user_behaviors in user_behaviors_batch:
            sequence = self.create_behavior_sequence(user_behaviors, behavior_types)
            sequences.append(sequence)
        
        return np.array(sequences)


# Example usage and testing
def test_sequence_transformer():
    """Test the SequenceTransformer implementation."""
    # Create sample data
    batch_size = 32
    seq_len = 50
    feature_dim = 16
    
    # Random sequence features
    sequence_features = tf.random.normal((batch_size, seq_len, feature_dim))
    
    # Initialize transformer
    transformer = SequenceTransformer(
        max_seq_length=seq_len,
        d_model=128,
        num_heads=8,
        num_layers=4,
        output_dim=64
    )
    
    # Forward pass
    output = transformer(sequence_features, training=True)
    
    print(f"Input shape: {sequence_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Transformer test passed!")
    
    return transformer, output


if __name__ == "__main__":
    # Run test
    transformer, output = test_sequence_transformer()
