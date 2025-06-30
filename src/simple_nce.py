"""
Simplified NCE Loss Implementation

This module provides a simplified NCE loss that can be used like binary_crossentropy.
"""

import tensorflow as tf
import numpy as np
from typing import Optional


class SimpleNCELoss(tf.keras.losses.Loss):
    """
    Simplified NCE Loss that can be used like binary_crossentropy.
    
    This version automatically handles negative sampling internally,
    so you can use it with standard binary classification data.
    """
    
    def __init__(self, num_negatives: int = 5, temperature: float = 0.1, name: str = 'simple_nce_loss', **kwargs):
        super(SimpleNCELoss, self).__init__(name=name, **kwargs)
        self.num_negatives = num_negatives
        self.temperature = temperature
    
    def call(self, y_true, y_pred):
        """
        Compute simplified NCE loss.
        
        Args:
            y_true: True labels [batch_size]
            y_pred: Predicted similarities [batch_size]
        
        Returns:
            NCE loss tensor
        """
        batch_size = tf.shape(y_true)[0]
        
        # For positive samples (y_true == 1), use the predicted similarity
        # For negative samples (y_true == 0), generate random negative similarities
        positive_mask = tf.cast(y_true, tf.bool)
        
        # Positive similarities
        positive_sim = tf.where(positive_mask, y_pred, tf.zeros_like(y_pred))
        
        # Generate negative similarities for all samples
        # In a real implementation, you'd want proper negative sampling
        negative_sim = tf.random.uniform(
            shape=[batch_size, self.num_negatives], 
            minval=-1, 
            maxval=1
        )
        
        # Apply temperature scaling——温度系数调整
        positive_sim_scaled = positive_sim / self.temperature
        negative_sim_scaled = negative_sim / self.temperature
        
        # For positive samples, compute NCE loss
        # For negative samples, use standard binary crossentropy
        nce_loss = tf.where(
            positive_mask,
            self._compute_nce_loss(positive_sim_scaled, negative_sim_scaled),
            tf.zeros_like(positive_sim)
        )
        
        # For negative samples, use binary crossentropy
        bce_loss = tf.where(
            tf.logical_not(positive_mask),
            tf.keras.losses.binary_crossentropy(y_true, tf.sigmoid(y_pred)),
            tf.zeros_like(y_pred)
        )
        
        # Combine losses
        total_loss = nce_loss + bce_loss
        
        return tf.reduce_mean(total_loss)
    
    def _compute_nce_loss(self, positive_sim, negative_sim):
        """Compute NCE loss for positive samples."""
        # Concatenate positive and negative similarities
        all_logits = tf.concat([positive_sim[:, None], negative_sim], axis=1)
        
        # Create labels: first element is positive (1), rest are negative (0)
        labels = tf.zeros_like(all_logits)
        labels = tf.tensor_scatter_nd_update(
            labels, 
            tf.constant([[i, 0] for i in range(tf.shape(labels)[0])]), 
            tf.ones(tf.shape(labels)[0])
        )
        
        # Compute cross-entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=all_logits)
        
        return loss
    
    def get_config(self):
        config = super(SimpleNCELoss, self).get_config()
        config.update({
            'num_negatives': self.num_negatives,
            'temperature': self.temperature
        })
        return config


# Convenience function to use NCE loss like binary_crossentropy
def nce_loss(num_negatives: int = 5, temperature: float = 0.1):
    """
    Create a simple NCE loss function that can be used like binary_crossentropy.
    
    Args:
        num_negatives: Number of negative samples per positive sample
        temperature: Temperature parameter for similarity scaling
        
    Returns:
        SimpleNCELoss instance
    """
    return SimpleNCELoss(num_negatives=num_negatives, temperature=temperature)


# Pre-configured NCE loss functions
def nce_loss_5():
    """NCE loss with 5 negative samples."""
    return nce_loss(num_negatives=5, temperature=0.1)

def nce_loss_10():
    """NCE loss with 10 negative samples."""
    return nce_loss(num_negatives=10, temperature=0.1)

def nce_loss_20():
    """NCE loss with 20 negative samples."""
    return nce_loss(num_negatives=20, temperature=0.1) 