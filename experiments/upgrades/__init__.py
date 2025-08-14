"""
Upgrades package for enhanced DSSM recommendation system.

This package contains advanced feature engineering and training techniques
to improve the performance of the DSSM model.
"""

from .sequence_transformer import SequenceTransformer
from .hard_negative_sampling import HardNegativeSampler

__all__ = [
    'SequenceTransformer',
    'HardNegativeSampler'
]
