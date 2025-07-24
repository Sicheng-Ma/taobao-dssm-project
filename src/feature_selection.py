# src/feature_selection.py

"""
Feature Selection Module using LightGBM

Performs feature importance analysis using LightGBM and selects features based on importance scores.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import logging
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class LightGBMFeatureSelector:
    """
    Feature selector using LightGBM for importance analysis.
    """
    
    def __init__(self, 
                 importance_threshold: float = 0.001,
                 top_k_features: Optional[int] = None,
                 random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            importance_threshold: Minimum importance score to keep a feature
            top_k_features: Number of top features to select (if None, use threshold)
            random_state: Random seed for reproducibility
        """
        self.importance_threshold = importance_threshold
        self.top_k_features = top_k_features
        self.random_state = random_state
        self.feature_importance = None
        self.selected_features = None
        self.model = None
        
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            categorical_features: Optional[List[str]] = None,
            **lgb_params) -> 'LightGBMFeatureSelector':
        """
        Fit LightGBM model and calculate feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            categorical_features: List of categorical feature names
            **lgb_params: Additional LightGBM parameters
            
        Returns:
            Self for method chaining
        """
        logger.info("Starting LightGBM feature importance analysis")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Default LightGBM parameters
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # Update with user parameters
        default_params.update(lgb_params)
        
        # Prepare dataset
        if categorical_features:
            # Convert categorical features to category dtype
            for col in categorical_features:
                if col in X_train.columns:
                    X_train[col] = X_train[col].astype('category')
                    X_val[col] = X_val[col].astype('category')
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        logger.info("Training LightGBM model for feature importance")
        self.model = lgb.train(
            default_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Get feature importance
        self.feature_importance = self._get_feature_importance()
        
        # Select features
        self.selected_features = self._select_features()
        
        logger.info(f"Feature importance analysis completed. Selected {len(self.selected_features)} features")
        
        return self
    
    def _get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from the trained model."""
        importance_dict = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_dict
        }).sort_values('importance', ascending=False)
        
        # Normalize importance scores
        importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
        
        return importance_df
    
    def _select_features(self) -> List[str]:
        """Select features based on importance criteria."""
        if self.top_k_features is not None:
            # Select top K features
            selected = self.feature_importance.head(self.top_k_features)['feature'].tolist()
        else:
            # Select features above threshold
            selected = self.feature_importance[
                self.feature_importance['importance_normalized'] >= self.importance_threshold
            ]['feature'].tolist()
        
        return selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to include only selected features."""
        if self.selected_features is None:
            raise ValueError("Model must be fitted before transforming data")
        
        missing_features = set(self.selected_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        return X[self.selected_features]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.feature_importance.copy()
    
    def plot_feature_importance(self, 
                               top_n: int = 20, 
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> None:
        """Plot feature importance."""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted before plotting feature importance")
        
        # Get top N features
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=figsize)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance_normalized'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Normalized Importance')
        plt.title(f'Top {top_n} Feature Importance (LightGBM)')
        plt.gca().invert_yaxis()
        
        # Add importance values on bars
        for i, (_, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance_normalized'] + 0.001, i, 
                    f'{row["importance_normalized"]:.4f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str) -> None:
        """Save feature selection results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = os.path.join(output_dir, 'feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
        
        # Save selected features
        if self.selected_features is not None:
            selected_path = os.path.join(output_dir, 'selected_features.pkl')
            with open(selected_path, 'wb') as f:
                pickle.dump(self.selected_features, f)
            logger.info(f"Selected features saved to {selected_path}")
        
        # Save model
        if self.model is not None:
            model_path = os.path.join(output_dir, 'lightgbm_feature_selector.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"LightGBM model saved to {model_path}")


def perform_feature_selection(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            target_col: str,
                            sparse_features: List[str],
                            dense_features: List[str],
                            output_dir: str,
                            importance_threshold: float = 0.001,
                            top_k_features: Optional[int] = None,
                            **lgb_params) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Perform feature selection using LightGBM.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        target_col: Target column name
        sparse_features: List of sparse feature names
        dense_features: List of dense feature names
        output_dir: Output directory for saving results
        importance_threshold: Minimum importance score to keep a feature
        top_k_features: Number of top features to select
        **lgb_params: Additional LightGBM parameters
        
    Returns:
        Tuple of (selected_train_df, selected_test_df, selected_feature_names)
    """
    logger.info("Starting feature selection process")
    
    # Prepare features
    all_features = sparse_features + dense_features
    X_train = train_df[all_features].copy()
    X_test = test_df[all_features].copy()
    y_train = train_df[target_col]
    
    # Initialize feature selector
    selector = LightGBMFeatureSelector(
        importance_threshold=importance_threshold,
        top_k_features=top_k_features
    )
    
    # Fit and select features
    selector.fit(X_train, y_train, categorical_features=sparse_features, **lgb_params)
    
    # Get selected features
    selected_features = selector.selected_features
    logger.info(f"Selected {len(selected_features)} features out of {len(all_features)}")
    
    # Transform datasets
    selected_train_df = selector.transform(X_train)
    selected_test_df = selector.transform(X_test)
    
    # Add target column back to training data
    selected_train_df[target_col] = y_train
    
    # Save results
    selector.save_results(output_dir)
    
    # Plot feature importance
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    selector.plot_feature_importance(save_path=plot_path)
    
    # Print summary
    logger.info("Feature selection summary:")
    logger.info(f"Original features: {len(all_features)}")
    logger.info(f"Selected features: {len(selected_features)}")
    logger.info(f"Reduction: {((len(all_features) - len(selected_features)) / len(all_features) * 100):.1f}%")
    
    return selected_train_df, selected_test_df, selected_features


def load_feature_selection_results(output_dir: str) -> Tuple[List[str], pd.DataFrame]:
    """
    Load previously saved feature selection results.
    
    Args:
        output_dir: Directory containing saved results
        
    Returns:
        Tuple of (selected_features, feature_importance_df)
    """
    # Load selected features
    selected_path = os.path.join(output_dir, 'selected_features.pkl')
    with open(selected_path, 'rb') as f:
        selected_features = pickle.load(f)
    
    # Load feature importance
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance = pd.read_csv(importance_path)
    
    return selected_features, feature_importance


if __name__ == "__main__":
    # Example usage
    logger.info("Feature selection module loaded successfully") 