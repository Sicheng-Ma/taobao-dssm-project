"""
InfoNCE Loss for DSSM Recommendation Model

Simple implementation of InfoNCE loss for contrastive learning in recommendation systems.
"""

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class InfoNCELoss(tf.keras.losses.Loss):
    """
    InfoNCE (Info Noise Contrastive Estimation) Loss for contrastive learning.
    
    Args:
        temperature: Temperature parameter for scaling logits (default: 0.1)
        reduction: Reduction type for the loss
        name: Name of the loss function
    """
    
    def __init__(self, temperature: float = 0.1, 
                 reduction: str = 'auto', 
                 name: str = 'info_nce_loss', **kwargs):
        super(InfoNCELoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.temperature = temperature
        logger.info(f"Initialized InfoNCE loss with temperature: {temperature}")
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            y_true: Ground truth labels (not used in InfoNCE, but required by Keras)
            y_pred: Predicted similarity scores [batch_size, num_negatives + 1]
                   where the first element is the positive pair score
                   
        Returns:
            InfoNCE loss value
        """
        # y_pred should be [batch_size, num_negatives + 1] where first column is positive
        # and remaining columns are negative samples
        
        # Extract positive and negative scores
        positive_scores = y_pred[:, 0:1]  # [batch_size, 1]
        negative_scores = y_pred[:, 1:]   # [batch_size, num_negatives]
        
        # Apply temperature scaling
        positive_scores = positive_scores / self.temperature
        negative_scores = negative_scores / self.temperature
        
        # Concatenate positive and negative scores
        all_scores = tf.concat([positive_scores, negative_scores], axis=1)
        
        # Compute log-softmax
        log_softmax = tf.nn.log_softmax(all_scores, axis=1)
        
        # InfoNCE loss is negative log-likelihood of positive sample
        # The positive sample is at index 0
        loss = -log_softmax[:, 0]
        
        return loss
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super(InfoNCELoss, self).get_config()
        config.update({
            'temperature': self.temperature
        })
        return config


def create_info_nce_loss(temperature: float = 0.1) -> InfoNCELoss:
    """
    Factory function to create InfoNCE loss with specified temperature.
    
    Args:
        temperature: Temperature parameter for scaling logits
        
    Returns:
        Configured InfoNCE loss function
    """
    return InfoNCELoss(temperature=temperature) 