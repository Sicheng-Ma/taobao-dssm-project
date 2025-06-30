#!/usr/bin/env python3
"""
Simple NCE Loss Usage Example

This example shows how to use the simplified NCE loss that works like binary_crossentropy.
"""

import sys
import os
sys.path.append('.')

# Import existing modules
from src import config
from src import model

# Import simplified NCE loss
from src.simple_nce import nce_loss, nce_loss_5, nce_loss_10

import tensorflow as tf
import pandas as pd
import numpy as np


def compare_loss_functions():
    """Compare different loss functions usage."""
    print("=== Loss Function Comparison ===\n")
    
    # 1. Binary Crossentropy (original)
    print("1. Binary Crossentropy (Original)")
    print("   Usage: loss='binary_crossentropy'")
    print("   Code: model.compile(loss='binary_crossentropy', metrics=['auc'])")
    print("   Complexity: Very Simple ✅")
    print()
    
    # 2. Simplified NCE Loss
    print("2. Simplified NCE Loss (New)")
    print("   Usage: loss=nce_loss()")
    print("   Code: model.compile(loss=nce_loss(), metrics=['auc'])")
    print("   Complexity: Simple ✅")
    print()
    
    # 3. Complex NCE Loss (Previous version)
    print("3. Complex NCE Loss (Previous)")
    print("   Usage: Multiple steps required")
    print("   Code: Multiple lines of setup code")
    print("   Complexity: Complex ❌")
    print()


def simple_usage_example():
    """Show simple usage of NCE loss."""
    print("=== Simple NCE Loss Usage ===\n")
    
    # Create a simple model (for demonstration)
    model_simple = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    print("Method 1: Using pre-configured functions")
    print("model.compile(loss=nce_loss_5(), metrics=['auc'])")
    model_simple.compile(loss=nce_loss_5(), metrics=['auc'])
    print("✅ Success!")
    print()
    
    print("Method 2: Using custom parameters")
    print("model.compile(loss=nce_loss(num_negatives=10, temperature=0.1), metrics=['auc'])")
    model_simple.compile(loss=nce_loss(num_negatives=10, temperature=0.1), metrics=['auc'])
    print("✅ Success!")
    print()
    
    print("Method 3: Using string-like interface")
    print("nce_loss_fn = nce_loss()")
    print("model.compile(loss=nce_loss_fn, metrics=['auc'])")
    nce_loss_fn = nce_loss()
    model_simple.compile(loss=nce_loss_fn, metrics=['auc'])
    print("✅ Success!")
    print()


def dssm_with_simple_nce():
    """Show how to use simplified NCE with DSSM model."""
    print("=== DSSM with Simple NCE Loss ===\n")
    
    # Create simple feature columns for demonstration
    user_feature_columns = [
        model.SparseFeat(name='user_id', vocabulary_size=1000, embedding_dim=16),
        model.SparseFeat(name='age_level', vocabulary_size=10, embedding_dim=16)
    ]
    
    item_feature_columns = [
        model.SparseFeat(name='item_id', vocabulary_size=2000, embedding_dim=16),
        model.DenseFeat(name='price', dimension=1)
    ]
    
    # Create DSSM model
    dssm_model = model.DSSM(
        user_feature_columns, 
        item_feature_columns, 
        dnn_units=[64, 32], 
        temp=1.0
    )
    
    print("Original DSSM model with Binary Crossentropy:")
    print("dssm_model.compile(loss='binary_crossentropy', metrics=['auc'])")
    print()
    
    print("DSSM model with Simple NCE Loss:")
    print("dssm_model.compile(loss=nce_loss_10(), metrics=['auc'])")
    
    # Compile with NCE loss
    dssm_model.compile(loss=nce_loss_10(), metrics=['auc'])
    print("✅ Success! DSSM model now uses NCE loss!")
    print()


def performance_comparison():
    """Show the performance implications."""
    print("=== Performance Comparison ===\n")
    
    print("Binary Crossentropy:")
    print("  ✅ Very fast")
    print("  ✅ Memory efficient")
    print("  ✅ Simple implementation")
    print("  ❌ May not be optimal for recommendation systems")
    print()
    
    print("Simple NCE Loss:")
    print("  ✅ Better for recommendation systems")
    print("  ✅ Handles negative sampling automatically")
    print("  ✅ Easy to use (like binary_crossentropy)")
    print("  ⚠️  Slightly more computation")
    print("  ⚠️  Requires tuning of num_negatives and temperature")
    print()
    
    print("Complex NCE Loss:")
    print("  ✅ Most flexible")
    print("  ✅ Best performance potential")
    print("  ❌ Complex to implement")
    print("  ❌ Requires significant code changes")
    print("  ❌ Hard to maintain")
    print()


def migration_guide():
    """Show how to migrate from binary_crossentropy to NCE loss."""
    print("=== Migration Guide ===\n")
    
    print("From Binary Crossentropy to Simple NCE Loss:")
    print()
    print("Before:")
    print("  model.compile(loss='binary_crossentropy', metrics=['auc'])")
    print()
    print("After:")
    print("  from src.simple_nce import nce_loss_10")
    print("  model.compile(loss=nce_loss_10(), metrics=['auc'])")
    print()
    print("That's it! No other changes needed! 🎉")
    print()


def main():
    """Main function to demonstrate simplified NCE loss."""
    print("Simplified NCE Loss Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    compare_loss_functions()
    simple_usage_example()
    dssm_with_simple_nce()
    performance_comparison()
    migration_guide()
    
    print("=" * 50)
    print("Summary:")
    print("✅ Simplified NCE loss is almost as easy to use as binary_crossentropy")
    print("✅ No need for complex model wrapping or data preprocessing")
    print("✅ Just replace 'binary_crossentropy' with nce_loss()")
    print("✅ Better performance for recommendation systems")
    print()
    print("The key insight: We can make NCE loss simple by handling")
    print("the complexity internally, just like TensorFlow does with")
    print("binary_crossentropy!")


if __name__ == "__main__":
    main() 