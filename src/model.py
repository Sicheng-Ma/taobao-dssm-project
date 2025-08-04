"""
Taobao DSSM Model Implementation

Implements the Deep Structured Semantic Model (DSSM) for recommendation systems.
Provides user and item embedding towers with cosine similarity matching.
"""

import tensorflow as tf
from collections import namedtuple
from typing import List, Tuple, Optional
import logging

# Import configuration
from src.config import (
    EMBEDDING_DIM,
    DNN_UNITS,
    TEMP
)

# Configure logging
logger = logging.getLogger(__name__)

# Feature column data structures
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])


class DNN(tf.keras.layers.Layer):
    """
    Deep Neural Network layer for feature transformation.
    
    Args:
        hidden_units: List of hidden layer dimensions
        activation: Activation function for hidden layers
        l2_reg: L2 regularization coefficient
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(self, hidden_units: List[int], activation: str = 'relu', 
                 l2_reg: float = 1e-4, dropout_rate: float = 0.3, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.dnn_layers = [
            tf.keras.layers.Dense(
                units, 
                activation=activation, 
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ) for units in hidden_units
        ]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout_rate) for _ in hidden_units]
    
    def call(self, inputs, training=None):
        """Forward pass through the DNN."""
        x = inputs
        for i in range(len(self.dnn_layers)):
            x = self.dnn_layers[i](x)
            x = self.dropout_layers[i](x, training=training) 
        return x


class CosinSimilarity(tf.keras.layers.Layer):
    """
    Cosine similarity layer for computing similarity between user and item embeddings.
    
    Args:
        temp: Temperature parameter for scaling similarity scores
    """
    
    def __init__(self, temp: float = 1.0, **kwargs):
        super(CosinSimilarity, self).__init__(**kwargs)
        self.temp = temp

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Compute cosine similarity between user and item embeddings."""
        user_emb, item_emb = inputs
        user_emb_norm = tf.nn.l2_normalize(user_emb, axis=1)
        item_emb_norm = tf.nn.l2_normalize(item_emb, axis=1)
        dot_product = tf.reduce_sum(
            tf.multiply(user_emb_norm, item_emb_norm), 
            axis=1, 
            keepdims=True
        )
        return dot_product * self.temp


class PredictLayer(tf.keras.layers.Layer):
    """Prediction layer that outputs probability scores."""
    
    def __init__(self, **kwargs):
        super(PredictLayer, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Generate prediction probabilities."""
        return self.dense(inputs)
        
class FeatureEncoder(tf.keras.layers.Layer):
    """
    Feature encoder that handles both sparse and dense features.
    
    Args:
        feature_columns: List of feature column definitions
    """
    
    def __init__(self, feature_columns: List, **kwargs):
        super(FeatureEncoder, self).__init__(**kwargs)
        self.feature_columns = feature_columns
        
        # Create input layers for all features
        self.feature_input_layer_dict = {
            feat.name: tf.keras.layers.Input(
                shape=(feat.dimension if isinstance(feat, DenseFeat) else 1,), 
                name=feat.name
            )
            for feat in self.feature_columns
        }
        
        # Create embedding layers for sparse features
        self.embedding_layer_dict = {
            feat.name: tf.keras.layers.Embedding(
                feat.vocabulary_size, 
                feat.embedding_dim, 
                name=f"emb_{feat.name}"
            )
            for feat in self.feature_columns if isinstance(feat, SparseFeat)
        }


def process_feature(feature_columns: List, feature_encoder: FeatureEncoder) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    Process features through the encoder.
    
    Args:
        feature_columns: List of feature column definitions
        feature_encoder: Feature encoder instance
        
    Returns:
        Tuple of (sparse_embeddings, dense_inputs)
    """
    sparse_embedding_list = []
    dense_input_list = []

    for feat in feature_columns:
        input_layer = feature_encoder.feature_input_layer_dict[feat.name]
        if isinstance(feat, SparseFeat):
            embedding_layer = feature_encoder.embedding_layer_dict[feat.name]
            sparse_embedding_list.append(embedding_layer(input_layer))
        elif isinstance(feat, DenseFeat):
            dense_input_list.append(input_layer)
            
    return sparse_embedding_list, dense_input_list


def DSSM(user_feature_columns: List, item_feature_columns: List, 
         dnn_units: Optional[List[int]] = None, temp: Optional[float] = None) -> tf.keras.Model:
    """
    Build DSSM model with user and item towers.
    
    Args:
        user_feature_columns: List of user feature column definitions
        item_feature_columns: List of item feature column definitions
        dnn_units: Hidden layer dimensions for DNN towers
        temp: Temperature parameter for cosine similarity
        
    Returns:
        Compiled DSSM model with user and item embedding submodels
    """
    # Use config defaults if not provided
    dnn_units = dnn_units or DNN_UNITS
    temp = temp or TEMP
    
    logger.info(f"Building DSSM model with DNN units: {dnn_units}, temperature: {temp}")
    
    # Create feature encoder
    feature_encode = FeatureEncoder(user_feature_columns + item_feature_columns)
    feature_input_layers_list = list(feature_encode.feature_input_layer_dict.values())
    
    # Process user features
    user_sparse_embs, _ = process_feature(user_feature_columns, feature_encode)
    user_dnn_input = tf.keras.layers.Flatten()(
        tf.keras.layers.Concatenate(axis=1)(user_sparse_embs)
    )
    user_dnn_out = DNN(dnn_units, name="user_dnn")(user_dnn_input)
    
    # Process item features
    item_sparse_embs, item_dense_inputs = process_feature(item_feature_columns, feature_encode)
    item_sparse_dnn_input = tf.keras.layers.Flatten()(
        tf.keras.layers.Concatenate(axis=1)(item_sparse_embs)
    )
    item_all_features_input = tf.keras.layers.Concatenate(axis=1)(
        [item_sparse_dnn_input] + item_dense_inputs
    )
    item_dnn_out = DNN(dnn_units, name="item_dnn")(item_all_features_input)
    
    # Compute similarity and predictions
    scores = CosinSimilarity(temp)([user_dnn_out, item_dnn_out])
    output = PredictLayer()(scores)
    
    # Build main model
    model = tf.keras.models.Model(feature_input_layers_list, output)
    
    # Store input layers for submodels
    model.user_input_layers = [
        feature_encode.feature_input_layer_dict[feat.name] 
        for feat in user_feature_columns
    ]
    model.item_input_layers = [
        feature_encode.feature_input_layer_dict[feat.name] 
        for feat in item_feature_columns
    ]
    
    # Create embedding submodels
    model.user_embedding_model = tf.keras.models.Model(
        inputs=model.user_input_layers, 
        outputs=user_dnn_out
    )
    model.item_embedding_model = tf.keras.models.Model(
        inputs=model.item_input_layers, 
        outputs=item_dnn_out
    )
    
    logger.info("DSSM model built successfully")
    return model