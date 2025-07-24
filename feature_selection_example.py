#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM Feature Selection Example

This script demonstrates how to use the LightGBM feature selection module
to analyze feature importance and select the most relevant features.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.feature_selection import perform_feature_selection, load_feature_selection_results
from src.config import SPARSE_FEATURES, DENSE_FEATURES, OUTPUT_DIR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate feature selection."""
    
    # Check if processed data exists
    train_data_path = os.path.join(OUTPUT_DIR, 'train_data.csv')
    test_data_path = os.path.join(OUTPUT_DIR, 'test_data.csv')
    
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        logger.error("Processed data not found. Please run data processing first.")
        logger.info("Expected files:")
        logger.info(f"  - {train_data_path}")
        logger.info(f"  - {test_data_path}")
        return
    
    # Load data
    logger.info("Loading processed data...")
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Create output directory for feature selection results
    feature_selection_dir = os.path.join(OUTPUT_DIR, 'feature_selection')
    os.makedirs(feature_selection_dir, exist_ok=True)
    
    # Perform feature selection with different strategies
    logger.info("=" * 50)
    logger.info("STRATEGY 1: Threshold-based feature selection")
    logger.info("=" * 50)
    
    # Strategy 1: Use importance threshold
    selected_train_1, selected_test_1, selected_features_1 = perform_feature_selection(
        train_df=train_df,
        test_df=test_df,
        target_col='label',
        sparse_features=SPARSE_FEATURES,
        dense_features=DENSE_FEATURES,
        output_dir=os.path.join(feature_selection_dir, 'threshold_based'),
        importance_threshold=0.01,  # Keep features with >= 1% importance
        top_k_features=None,
        num_leaves=31,
        learning_rate=0.05,
        num_boost_round=100
    )
    
    logger.info(f"Selected features (threshold): {selected_features_1}")
    
    logger.info("=" * 50)
    logger.info("STRATEGY 2: Top-K feature selection")
    logger.info("=" * 50)
    
    # Strategy 2: Select top K features
    selected_train_2, selected_test_2, selected_features_2 = perform_feature_selection(
        train_df=train_df,
        test_df=test_df,
        target_col='label',
        sparse_features=SPARSE_FEATURES,
        dense_features=DENSE_FEATURES,
        output_dir=os.path.join(feature_selection_dir, 'top_k_based'),
        importance_threshold=0.0,  # Set to 0 when using top_k_features
        top_k_features=10,  # Select top 10 features
        num_leaves=31,
        learning_rate=0.05,
        num_boost_round=100
    )
    
    logger.info(f"Selected features (top-k): {selected_features_2}")
    
    # Compare strategies
    logger.info("=" * 50)
    logger.info("COMPARISON OF STRATEGIES")
    logger.info("=" * 50)
    
    logger.info(f"Threshold-based selection: {len(selected_features_1)} features")
    logger.info(f"Top-K selection: {len(selected_features_2)} features")
    
    # Find common features
    common_features = set(selected_features_1) & set(selected_features_2)
    logger.info(f"Common features: {len(common_features)}")
    logger.info(f"Common feature names: {list(common_features)}")
    
    # Save comparison results
    comparison_results = {
        'threshold_based': selected_features_1,
        'top_k_based': selected_features_2,
        'common_features': list(common_features)
    }
    
    import json
    comparison_path = os.path.join(feature_selection_dir, 'comparison_results.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"Comparison results saved to {comparison_path}")
    
    # Demonstrate loading saved results
    logger.info("=" * 50)
    logger.info("DEMONSTRATING LOADING SAVED RESULTS")
    logger.info("=" * 50)
    
    # Load threshold-based results
    threshold_dir = os.path.join(feature_selection_dir, 'threshold_based')
    loaded_features, loaded_importance = load_feature_selection_results(threshold_dir)
    
    logger.info(f"Loaded {len(loaded_features)} features from saved results")
    logger.info(f"Top 5 most important features:")
    for i, (_, row) in enumerate(loaded_importance.head(5).iterrows()):
        logger.info(f"  {i+1}. {row['feature']}: {row['importance_normalized']:.4f}")
    
    logger.info("=" * 50)
    logger.info("FEATURE SELECTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 50)
    logger.info(f"Results saved to: {feature_selection_dir}")
    logger.info("Files created:")
    logger.info("  - feature_importance.csv: Detailed importance scores")
    logger.info("  - selected_features.pkl: List of selected features")
    logger.info("  - lightgbm_feature_selector.pkl: Trained LightGBM model")
    logger.info("  - feature_importance.png: Visualization of top features")
    logger.info("  - comparison_results.json: Comparison between strategies")


if __name__ == "__main__":
    main() 