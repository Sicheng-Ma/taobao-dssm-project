# src/feature_engineering.py

"""
Feature Engineering Module

Handles feature encoding and scaling for the DSSM model.
"""

import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)


def perform_feature_engineering(train_df, test_df, sparse_features, dense_features, output_dir):
    """
    Perform feature engineering on training and test datasets.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        sparse_features: List of sparse feature names
        dense_features: List of dense feature names
        output_dir: Output directory for saving encoders
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    logger.info("Starting feature engineering")
    
    # Make copies to avoid modifying original data
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1. Process sparse features: Label Encoding
    feature_encoders = {}
    for feat in tqdm(sparse_features, desc="Encoding Sparse Features"):
        encoder = LabelEncoder()
        
        # Handle missing values for sparse features
        train_df[feat] = train_df[feat].fillna('UNKNOWN')
        test_df[feat] = test_df[feat].fillna('UNKNOWN')
        
        # Convert to string and fit encoder
        combined_data = pd.concat([train_df[feat], test_df[feat]], axis=0).astype(str)
        encoder.fit(combined_data)
        
        # Transform data
        train_df[feat] = encoder.transform(train_df[feat].astype(str))
        test_df[feat] = encoder.transform(test_df[feat].astype(str))
        
        feature_encoders[feat] = encoder
        logger.info(f"Encoded feature {feat}: {len(encoder.classes_) if encoder.classes_ is not None else 0} unique values")
    
    logger.info("Sparse feature encoding completed")

    # 2. Process dense features: MinMax Scaling
    if dense_features:
        mms = MinMaxScaler(feature_range=(0, 1))
        
        # Handle missing values for dense features
        for feat in dense_features:
            train_df[feat] = train_df[feat].fillna(train_df[feat].median())
            test_df[feat] = test_df[feat].fillna(test_df[feat].median())
        
        # Fit and transform
        train_df[dense_features] = mms.fit_transform(train_df[dense_features])
        test_df[dense_features] = mms.transform(test_df[dense_features])
        
        logger.info("Dense feature scaling completed")
    else:
        mms = None
        logger.info("No dense features to process")

    # 3. Save encoders and scalers
    with open(os.path.join(output_dir, 'feature_encoders.pkl'), 'wb') as f:
        pickle.dump(feature_encoders, f)
    
    if mms is not None:
        with open(os.path.join(output_dir, 'mms.pkl'), 'wb') as f:
            pickle.dump(mms, f)
    
    logger.info(f"Feature processors saved to {output_dir}")
    logger.info("Feature engineering completed")
    
    return train_df, test_df