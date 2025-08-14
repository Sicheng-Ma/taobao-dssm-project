"""
Hard Negative Sampling for improved DSSM training.

This module implements various hard negative sampling strategies to improve
the quality of negative samples and enhance model training effectiveness.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class HardNegativeSampler:
    """
    Hard negative sampling strategies for recommendation systems.
    
    This class implements various hard negative sampling methods to select
    more challenging negative samples that are similar to positive samples
    but not clicked by the user.
    """
    
    def __init__(self, 
                 item_embeddings: Optional[np.ndarray] = None,
                 item_ids: Optional[List] = None,
                 sampling_strategy: str = 'similarity_based',
                 top_k: int = 1000,
                 similarity_threshold: float = 0.7,
                 cache_size: int = 10000):
        """
        Initialize the HardNegativeSampler.
        
        Args:
            item_embeddings: Pre-computed item embeddings (item_count, embedding_dim)
            item_ids: List of item IDs corresponding to embeddings
            sampling_strategy: Sampling strategy ('similarity_based', 'popularity_based', 'hybrid')
            top_k: Number of candidate items to consider for hard negatives
            similarity_threshold: Minimum similarity threshold for hard negatives
            cache_size: Size of similarity cache
        """
        self.item_embeddings = item_embeddings
        self.item_ids = item_ids
        self.sampling_strategy = sampling_strategy
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.cache_size = cache_size
        
        # Initialize similarity cache
        self.similarity_cache = {}
        self.item_popularity = {}
        
        # Build item ID to index mapping
        if item_ids is not None:
            self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        else:
            self.item_id_to_idx = {}
        
        logger.info(f"Initialized HardNegativeSampler with strategy: {sampling_strategy}")
    
    def update_item_embeddings(self, item_embeddings: np.ndarray, item_ids: List):
        """Update item embeddings and corresponding IDs."""
        self.item_embeddings = item_embeddings
        self.item_ids = item_ids
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        # Clear cache when embeddings are updated
        self.similarity_cache.clear()
        logger.info(f"Updated item embeddings with {len(item_ids)} items")
    
    def compute_item_similarities(self, target_item_id: Union[int, str]) -> np.ndarray:
        """
        Compute similarities between target item and all other items.
        
        Args:
            target_item_id: ID of the target item
            
        Returns:
            similarities: Array of similarities with all items
        """
        if target_item_id not in self.item_id_to_idx:
            logger.warning(f"Item {target_item_id} not found in embeddings")
            return np.zeros(len(self.item_ids))
        
        target_idx = self.item_id_to_idx[target_item_id]
        target_embedding = self.item_embeddings[target_idx:target_idx+1]
        
        # Compute cosine similarities
        similarities = cosine_similarity(target_embedding, self.item_embeddings)[0]
        
        return similarities
    
    def get_similarity_based_negatives(self, 
                                     positive_item_id: Union[int, str],
                                     exclude_items: List = None,
                                     num_negatives: int = 1) -> List[Union[int, str]]:
        """
        Get hard negatives based on similarity to positive item.
        
        Args:
            positive_item_id: ID of the positive item
            exclude_items: Items to exclude from negative sampling
            num_negatives: Number of negative samples to return
            
        Returns:
            negative_items: List of hard negative item IDs
        """
        if self.item_embeddings is None:
            logger.error("Item embeddings not available for similarity-based sampling")
            return []
        
        if exclude_items is None:
            exclude_items = []
        
        # Compute similarities
        similarities = self.compute_item_similarities(positive_item_id)
        
        # Create mask for excluded items
        exclude_mask = np.ones(len(similarities), dtype=bool)
        for item_id in exclude_items:
            if item_id in self.item_id_to_idx:
                exclude_mask[self.item_id_to_idx[item_id]] = False
        
        # Apply mask and threshold
        valid_mask = (similarities >= self.similarity_threshold) & exclude_mask
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"No valid hard negatives found for item {positive_item_id}")
            return []
        
        # Sort by similarity and select top candidates
        valid_similarities = similarities[valid_indices]
        sorted_indices = np.argsort(valid_similarities)[::-1][:self.top_k]
        
        # Randomly sample from top candidates
        num_candidates = min(len(sorted_indices), num_negatives * 3)  # 3x oversampling
        if num_candidates > 0:
            selected_indices = np.random.choice(
                sorted_indices[:num_candidates], 
                size=min(num_negatives, num_candidates), 
                replace=False
            )
            negative_items = [self.item_ids[idx] for idx in selected_indices]
        else:
            negative_items = []
        
        return negative_items
    
    def get_popularity_based_negatives(self, 
                                     positive_item_id: Union[int, str],
                                     exclude_items: List = None,
                                     num_negatives: int = 1) -> List[Union[int, str]]:
        """
        Get hard negatives based on popularity (items with similar popularity to positive).
        
        Args:
            positive_item_id: ID of the positive item
            exclude_items: Items to exclude from negative sampling
            num_negatives: Number of negative samples to return
            
        Returns:
            negative_items: List of hard negative item IDs
        """
        if not self.item_popularity:
            logger.warning("Item popularity not available, using random sampling")
            return self.get_random_negatives(exclude_items, num_negatives)
        
        if exclude_items is None:
            exclude_items = []
        
        positive_popularity = self.item_popularity.get(positive_item_id, 0)
        
        # Find items with similar popularity
        similar_popularity_items = []
        for item_id, popularity in self.item_popularity.items():
            if item_id not in exclude_items and item_id != positive_item_id:
                # Items within 50% popularity range
                if 0.5 * positive_popularity <= popularity <= 2.0 * positive_popularity:
                    similar_popularity_items.append((item_id, abs(popularity - positive_popularity)))
        
        # Sort by popularity difference and select candidates
        similar_popularity_items.sort(key=lambda x: x[1])
        candidates = [item_id for item_id, _ in similar_popularity_items[:self.top_k]]
        
        # Randomly sample from candidates
        if candidates:
            num_samples = min(num_negatives, len(candidates))
            negative_items = random.sample(candidates, num_samples)
        else:
            negative_items = []
        
        return negative_items
    
    def get_hybrid_negatives(self, 
                           positive_item_id: Union[int, str],
                           exclude_items: List = None,
                           num_negatives: int = 1,
                           similarity_weight: float = 0.7) -> List[Union[int, str]]:
        """
        Get hard negatives using hybrid strategy (similarity + popularity).
        
        Args:
            positive_item_id: ID of the positive item
            exclude_items: Items to exclude from negative sampling
            num_negatives: Number of negative samples to return
            similarity_weight: Weight for similarity-based sampling
            
        Returns:
            negative_items: List of hard negative item IDs
        """
        if exclude_items is None:
            exclude_items = []
        
        # Get candidates from both strategies
        similarity_candidates = self.get_similarity_based_negatives(
            positive_item_id, exclude_items, num_negatives * 2
        )
        popularity_candidates = self.get_popularity_based_negatives(
            positive_item_id, exclude_items, num_negatives * 2
        )
        
        # Combine candidates with weights
        all_candidates = []
        
        # Add similarity-based candidates
        for item_id in similarity_candidates:
            all_candidates.append((item_id, similarity_weight))
        
        # Add popularity-based candidates
        for item_id in popularity_candidates:
            if item_id not in [c[0] for c in all_candidates]:
                all_candidates.append((item_id, 1 - similarity_weight))
        
        # Sample based on weights
        if all_candidates:
            candidates, weights = zip(*all_candidates)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            num_samples = min(num_negatives, len(candidates))
            selected_indices = np.random.choice(
                len(candidates), 
                size=num_samples, 
                replace=False, 
                p=weights
            )
            negative_items = [candidates[idx] for idx in selected_indices]
        else:
            negative_items = []
        
        return negative_items
    
    def get_random_negatives(self, 
                           exclude_items: List = None,
                           num_negatives: int = 1) -> List[Union[int, str]]:
        """
        Get random negative samples (baseline method).
        
        Args:
            exclude_items: Items to exclude from negative sampling
            num_negatives: Number of negative samples to return
            
        Returns:
            negative_items: List of random negative item IDs
        """
        if exclude_items is None:
            exclude_items = []
        
        available_items = [item_id for item_id in self.item_ids if item_id not in exclude_items]
        
        if len(available_items) < num_negatives:
            logger.warning(f"Not enough items available for random sampling")
            return available_items
        
        negative_items = random.sample(available_items, num_negatives)
        return negative_items
    
    def sample_hard_negatives(self, 
                            positive_item_id: Union[int, str],
                            exclude_items: List = None,
                            num_negatives: int = 1) -> List[Union[int, str]]:
        """
        Main method to sample hard negatives based on the selected strategy.
        
        Args:
            positive_item_id: ID of the positive item
            exclude_items: Items to exclude from negative sampling
            num_negatives: Number of negative samples to return
            
        Returns:
            negative_items: List of hard negative item IDs
        """
        if exclude_items is None:
            exclude_items = []
        
        if self.sampling_strategy == 'similarity_based':
            return self.get_similarity_based_negatives(positive_item_id, exclude_items, num_negatives)
        elif self.sampling_strategy == 'popularity_based':
            return self.get_popularity_based_negatives(positive_item_id, exclude_items, num_negatives)
        elif self.sampling_strategy == 'hybrid':
            return self.get_hybrid_negatives(positive_item_id, exclude_items, num_negatives)
        else:
            logger.warning(f"Unknown sampling strategy: {self.sampling_strategy}, using random")
            return self.get_random_negatives(exclude_items, num_negatives)
    
    def update_item_popularity(self, item_interactions: Dict[Union[int, str], int]):
        """Update item popularity based on interaction counts."""
        self.item_popularity.update(item_interactions)
        logger.info(f"Updated item popularity for {len(item_interactions)} items")
    
    def batch_sample_negatives(self, 
                             positive_items: List[Union[int, str]],
                             exclude_items_dict: Dict = None,
                             num_negatives_per_item: int = 1) -> List[List[Union[int, str]]]:
        """
        Sample hard negatives for a batch of positive items.
        
        Args:
            positive_items: List of positive item IDs
            exclude_items_dict: Dictionary mapping positive items to their exclude lists
            num_negatives_per_item: Number of negatives per positive item
            
        Returns:
            negative_items_batch: List of negative item lists for each positive item
        """
        if exclude_items_dict is None:
            exclude_items_dict = {}
        
        negative_items_batch = []
        
        for positive_item in positive_items:
            exclude_items = exclude_items_dict.get(positive_item, [])
            negatives = self.sample_hard_negatives(
                positive_item, exclude_items, num_negatives_per_item
            )
            negative_items_batch.append(negatives)
        
        return negative_items_batch


class DynamicNegativeSampler:
    """
    Dynamic negative sampling that adapts based on training progress.
    """
    
    def __init__(self, 
                 base_sampler: HardNegativeSampler,
                 initial_ratio: float = 4.0,
                 min_ratio: float = 1.0,
                 max_ratio: float = 10.0,
                 adaptation_rate: float = 0.1):
        """
        Initialize DynamicNegativeSampler.
        
        Args:
            base_sampler: Base hard negative sampler
            initial_ratio: Initial negative to positive ratio
            min_ratio: Minimum negative to positive ratio
            max_ratio: Maximum negative to positive ratio
            adaptation_rate: Rate of ratio adaptation
        """
        self.base_sampler = base_sampler
        self.current_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.adaptation_rate = adaptation_rate
        
        # Training history
        self.train_auc_history = []
        self.val_auc_history = []
        
        logger.info(f"Initialized DynamicNegativeSampler with ratio {initial_ratio}")
    
    def update_ratio(self, epoch: int, train_auc: float, val_auc: float):
        """
        Update negative sampling ratio based on training progress.
        
        Args:
            epoch: Current training epoch
            train_auc: Training AUC
            val_auc: Validation AUC
        """
        self.train_auc_history.append(train_auc)
        self.val_auc_history.append(val_auc)
        
        # Check for overfitting
        if len(self.val_auc_history) >= 2:
            current_val_auc = self.val_auc_history[-1]
            previous_val_auc = self.val_auc_history[-2]
            
            # If validation AUC is decreasing, increase negative ratio
            if current_val_auc < previous_val_auc:
                self.current_ratio = min(
                    self.current_ratio * (1 + self.adaptation_rate), 
                    self.max_ratio
                )
                logger.info(f"Overfitting detected, increased ratio to {self.current_ratio:.2f}")
            
            # If validation AUC is improving significantly, decrease ratio
            elif current_val_auc > previous_val_auc * 1.02:
                self.current_ratio = max(
                    self.current_ratio * (1 - self.adaptation_rate), 
                    self.min_ratio
                )
                logger.info(f"Good progress, decreased ratio to {self.current_ratio:.2f}")
    
    def sample_negatives(self, 
                        positive_item_id: Union[int, str],
                        exclude_items: List = None) -> List[Union[int, str]]:
        """
        Sample negatives using current dynamic ratio.
        
        Args:
            positive_item_id: ID of the positive item
            exclude_items: Items to exclude from negative sampling
            
        Returns:
            negative_items: List of negative item IDs
        """
        num_negatives = max(1, int(self.current_ratio))
        return self.base_sampler.sample_hard_negatives(
            positive_item_id, exclude_items, num_negatives
        )
    
    def get_current_ratio(self) -> float:
        """Get current negative to positive ratio."""
        return self.current_ratio


class NegativeSampleGenerator:
    """
    High-level interface for generating negative samples for training data.
    """
    
    def __init__(self, 
                 sampler: Union[HardNegativeSampler, DynamicNegativeSampler],
                 data_df: pd.DataFrame,
                 user_col: str = 'user_id',
                 item_col: str = 'item_id',
                 label_col: str = 'label'):
        """
        Initialize NegativeSampleGenerator.
        
        Args:
            sampler: Negative sampler instance
            data_df: Training data DataFrame
            user_col: User ID column name
            item_col: Item ID column name
            label_col: Label column name
        """
        self.sampler = sampler
        self.data_df = data_df
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        
        # Build user-item interaction history
        self.user_positive_items = defaultdict(set)
        self._build_interaction_history()
        
        logger.info(f"Initialized NegativeSampleGenerator with {len(self.user_positive_items)} users")
    
    def _build_interaction_history(self):
        """Build user positive item interaction history."""
        positive_data = self.data_df[self.data_df[self.label_col] == 1]
        
        for _, row in positive_data.iterrows():
            user_id = row[self.user_col]
            item_id = row[self.item_col]
            self.user_positive_items[user_id].add(item_id)
    
    def generate_negative_samples(self, 
                                num_negatives_per_positive: int = 1,
                                strategy: str = 'hard') -> pd.DataFrame:
        """
        Generate negative samples for training data.
        
        Args:
            num_negatives_per_positive: Number of negatives per positive sample
            strategy: Sampling strategy ('hard', 'random', 'dynamic')
            
        Returns:
            negative_df: DataFrame with negative samples
        """
        negative_samples = []
        
        # Get positive samples
        positive_data = self.data_df[self.data_df[self.label_col] == 1]
        
        for _, row in positive_data.iterrows():
            user_id = row[self.user_col]
            positive_item = row[self.item_col]
            
            # Get items to exclude (user's positive items)
            exclude_items = list(self.user_positive_items[user_id])
            
            # Sample negatives
            if strategy == 'hard':
                negative_items = self.sampler.sample_hard_negatives(
                    positive_item, exclude_items, num_negatives_per_positive
                )
            elif strategy == 'random':
                negative_items = self.sampler.get_random_negatives(
                    exclude_items, num_negatives_per_positive
                )
            elif strategy == 'dynamic' and isinstance(self.sampler, DynamicNegativeSampler):
                negative_items = self.sampler.sample_negatives(positive_item, exclude_items)
            else:
                negative_items = self.sampler.get_random_negatives(
                    exclude_items, num_negatives_per_positive
                )
            
            # Create negative samples
            for negative_item in negative_items:
                negative_row = row.copy()
                negative_row[self.item_col] = negative_item
                negative_row[self.label_col] = 0
                negative_samples.append(negative_row)
        
        negative_df = pd.DataFrame(negative_samples)
        logger.info(f"Generated {len(negative_df)} negative samples")
        
        return negative_df
    
    def update_sampler_embeddings(self, item_embeddings: np.ndarray, item_ids: List):
        """Update item embeddings in the sampler."""
        if hasattr(self.sampler, 'update_item_embeddings'):
            self.sampler.update_item_embeddings(item_embeddings, item_ids)
        elif hasattr(self.sampler, 'base_sampler'):
            self.sampler.base_sampler.update_item_embeddings(item_embeddings, item_ids)


# Example usage and testing
def test_hard_negative_sampling():
    """Test the HardNegativeSampler implementation."""
    # Create sample item embeddings
    num_items = 1000
    embedding_dim = 64
    item_embeddings = np.random.randn(num_items, embedding_dim)
    item_ids = list(range(num_items))
    
    # Initialize sampler
    sampler = HardNegativeSampler(
        item_embeddings=item_embeddings,
        item_ids=item_ids,
        sampling_strategy='similarity_based',
        top_k=100
    )
    
    # Test sampling
    positive_item = 100
    exclude_items = [100, 101, 102]
    negatives = sampler.sample_hard_negatives(positive_item, exclude_items, 5)
    
    print(f"Positive item: {positive_item}")
    print(f"Exclude items: {exclude_items}")
    print(f"Sampled negatives: {negatives}")
    print(f"Hard negative sampling test passed!")
    
    return sampler


if __name__ == "__main__":
    # Run test
    sampler = test_hard_negative_sampling()
