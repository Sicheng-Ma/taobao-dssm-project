"""
Taobao DSSM Data Processing Pipeline

Processes raw data files to create training and test datasets for deep learning models.
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import gc
from typing import Optional, Union
import logging

# Import configuration
from src.config import (
    DATASET_FOLDER_NAME, 
    OUTPUT_DIR, 
    CHUNKSIZE, 
    NEG_POS_RATIO
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TEST_DATE = 20170513
TOTAL_SAMPLES = 26557961
RANDOM_SEED = 42

# Column mappings
COLUMN_MAPPINGS = {
    'ad_feature': {'customer': 'customer_id'},
    'user_profile': {
        'userid': 'user_id',
        'new_user_class_level ': 'new_user_class_level'
    },
    'raw_sample': {'user': 'user_id', 'clk': 'label'}
}


def clean_small_tables(data_path: Path, output_dir: Path) -> None:
    """Clean and standardize small reference tables."""
    logger.info("Processing small reference tables")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process ad_feature
    ad_feature_path = data_path / 'ad_feature.csv'
    if not ad_feature_path.exists():
        raise FileNotFoundError(f"ad_feature.csv not found at {ad_feature_path}")
    
    ad_feature = pd.read_csv(ad_feature_path)
    ad_feature = _standardize_columns(ad_feature, COLUMN_MAPPINGS['ad_feature'])
    ad_feature.to_csv(output_dir / 'ad_feature_cleaned.csv', index=False)
    
    # Process user_profile
    user_profile_path = data_path / 'user_profile.csv'
    if not user_profile_path.exists():
        raise FileNotFoundError(f"user_profile.csv not found at {user_profile_path}")
    
    user_profile = pd.read_csv(user_profile_path)
    user_profile = _standardize_columns(user_profile, COLUMN_MAPPINGS['user_profile'])
    user_profile.to_csv(output_dir / 'user_profile_cleaned.csv', index=False)
    
    logger.info("Small tables processed successfully")


def _standardize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Standardize column names according to mapping."""
    df = df.copy()
    for old_name, new_name in mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    return df


def process_large_file(data_path: Path, output_dir: Path, chunk_size: int = CHUNKSIZE) -> None:
    """Process large raw_sample.csv file in chunks."""
    logger.info("Processing large file in chunks")
    
    # Load reference tables
    ad_feature = pd.read_csv(output_dir / 'ad_feature_cleaned.csv')
    user_profile = pd.read_csv(output_dir / 'user_profile_cleaned.csv')
    
    # Define output files
    output_files = {
        'test': output_dir / 'test_samples.csv',
        'positive': output_dir / 'positive_samples.csv',
        'negative': output_dir / 'all_negative_samples.csv'
    }
    
    # Process chunks
    raw_sample_path = data_path / 'raw_sample.csv'
    if not raw_sample_path.exists():
        raise FileNotFoundError(f"raw_sample.csv not found at {raw_sample_path}")
    
    reader = pd.read_csv(raw_sample_path, chunksize=chunk_size)
    total_chunks = TOTAL_SAMPLES // chunk_size + 1
    is_first_chunk = True
    
    for chunk in tqdm(reader, total=total_chunks, desc="Processing chunks"):
        processed_chunk = _process_chunk(chunk, user_profile, ad_feature)
        _write_chunk_to_files(processed_chunk, output_files, is_first_chunk)
        is_first_chunk = False
    
    logger.info("Large file processing completed")


def _process_chunk(chunk: pd.DataFrame, user_profile: pd.DataFrame, ad_feature: pd.DataFrame) -> pd.DataFrame:
    """Process a single chunk of data."""
    # Standardize columns
    chunk = _standardize_columns(chunk, COLUMN_MAPPINGS['raw_sample'])
    
    # Remove unnecessary columns
    if 'noclk' in chunk.columns:
        chunk = chunk.drop(columns=['noclk'])
    
    # Merge with reference tables
    chunk = pd.merge(chunk, user_profile, on='user_id', how='left')
    chunk = pd.merge(chunk, ad_feature, on='adgroup_id', how='left')
    
    # Handle missing values more carefully
    # For categorical features, fill with -1 (will be handled in feature engineering)
    categorical_features = ['cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 
                           'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
                           'cate_id', 'campaign_id', 'customer_id', 'brand', 'pid']
    
    for feat in categorical_features:
        if feat in chunk.columns:
            chunk[feat] = chunk[feat].fillna(-1)
    
    # For numeric features, fill with 0
    numeric_features = ['price']
    for feat in numeric_features:
        if feat in chunk.columns:
            chunk[feat] = chunk[feat].fillna(0)
    
    # Convert timestamp to date
    chunk['day'] = pd.to_datetime(chunk['time_stamp'], unit='s').dt.strftime('%Y%m%d').astype(int)
    
    return chunk


def _write_chunk_to_files(chunk: pd.DataFrame, output_files: dict, is_first_chunk: bool) -> None:
    """Write processed chunk to appropriate output files."""
    # Split test data
    test_data = pd.DataFrame(chunk[chunk['day'] == TEST_DATE])
    if not test_data.empty:
        test_data.drop(columns=['day', 'time_stamp']).to_csv(
            output_files['test'], mode='a', index=False, header=is_first_chunk
        )
    
    # Split training data
    train_data = chunk[chunk['day'] < TEST_DATE]
    
    # Positive samples
    positive_data = pd.DataFrame(train_data[train_data['label'] == 1])
    if not positive_data.empty:
        positive_data.drop(columns=['day', 'time_stamp']).to_csv(
            output_files['positive'], mode='a', index=False, header=is_first_chunk
        )
    
    # Negative samples
    negative_data = pd.DataFrame(train_data[train_data['label'] == 0])
    if not negative_data.empty:
        negative_data.drop(columns=['day', 'time_stamp']).to_csv(
            output_files['negative'], mode='a', index=False, header=is_first_chunk
        )


def create_final_datasets(output_dir: Path, negative_ratio: int = NEG_POS_RATIO) -> None:
    """Create final training and test datasets with balanced sampling."""
    logger.info(f"Creating final datasets with negative ratio 1:{negative_ratio}")
    
    # Load positive samples
    positive_samples = pd.read_csv(output_dir / 'positive_samples.csv')
    num_positive = len(positive_samples)
    logger.info(f"Found {num_positive} positive samples")
    
    # Sample negative samples
    all_negative_samples = pd.read_csv(output_dir / 'all_negative_samples.csv')
    num_negative_needed = num_positive * negative_ratio
    num_to_sample = min(num_negative_needed, len(all_negative_samples))
    
    sampled_negative = all_negative_samples.sample(n=num_to_sample, random_state=RANDOM_SEED)
    logger.info(f"Sampled {len(sampled_negative)} negative samples")
    
    # Free memory
    del all_negative_samples
    gc.collect()
    
    # Create training dataset
    training_data = pd.concat([positive_samples, sampled_negative])
    training_data = training_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    training_data.to_csv(output_dir / 'train_data.csv', index=False)
    logger.info(f"Training dataset saved: {training_data.shape}")
    
    # Process test dataset
    test_data = pd.read_csv(output_dir / 'test_samples.csv')
    test_data.to_csv(output_dir / 'test_data.csv', index=False)
    logger.info(f"Test dataset saved: {test_data.shape}")


def run_pipeline(
    data_path: Union[str, Path], 
    output_dir: Union[str, Path] = OUTPUT_DIR,
    chunk_size: int = CHUNKSIZE, 
    negative_ratio: int = NEG_POS_RATIO
) -> None:
    """
    Run the complete data processing pipeline.
    
    Args:
        data_path: Path to raw data directory
        output_dir: Path to output directory (defaults to config.OUTPUT_DIR)
        chunk_size: Number of rows per chunk (defaults to config.CHUNKSIZE)
        negative_ratio: Ratio of negative to positive samples (defaults to config.NEG_POS_RATIO)
    """
    logger.info("Starting Taobao DSSM data processing pipeline")
    
    try:
        data_path = Path(data_path)
        output_dir = Path(output_dir)
        
        # Execute pipeline steps
        clean_small_tables(data_path, output_dir)
        process_large_file(data_path, output_dir, chunk_size)
        create_final_datasets(output_dir, negative_ratio)
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Training data: {output_dir / 'train_data.csv'}")
        logger.info(f"Test data: {output_dir / 'test_data.csv'}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Use configuration from config.py
    # For Kaggle environment, construct the data path
    DATA_PATH = f'../input/{DATASET_FOLDER_NAME}/'
    
    # Run the pipeline with config settings
    run_pipeline(DATA_PATH) 