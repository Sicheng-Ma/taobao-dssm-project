"""
改进的Taobao DSSM数据采样管道

解决AUC=0.5问题的关键：确保训练集和测试集类别分布一致
"""

import pandas as pd
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import gc
import random
from typing import Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import sampling configuration
import sys
sys.path.append('..')
from configs.sampling_config import (
    DATASET_FOLDER_NAME, 
    SAMPLE_RATIO,
    SAMPLE_CHUNKSIZE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
TEST_DATE = 20170513
RANDOM_SEED = 42

# ==================== 路径配置 ====================
# 本地采样路径配置
LOCAL_DATA_PATH = '/Users/masicheng/Desktop/搜广推/taobao-dssm-project/data'
LOCAL_OUTPUT_DIR = '/Users/masicheng/Desktop/搜广推/taobao-dssm-project/outputs'

# 云端采样路径配置
CLOUD_DATA_PATH = './data'
CLOUD_OUTPUT_DIR = './outputs'

    # ==================== 路径切换说明 ====================
    # 本地采样（快速验证）：
    # DATA_PATH = LOCAL_DATA_PATH
    # OUTPUT_DIR = LOCAL_OUTPUT_DIR
    # 用途：快速生成采样数据用于本地训练验证
    
    # 云端采样（生产环境）：
    # DATA_PATH = CLOUD_DATA_PATH
    # OUTPUT_DIR = CLOUD_OUTPUT_DIR
    # 用途：在云端环境生成采样数据
    
# 当前使用的路径配置（手动切换）
DATA_PATH = LOCAL_DATA_PATH  # 切换为 CLOUD_DATA_PATH 用于云端
OUTPUT_DIR = LOCAL_OUTPUT_DIR  # 切换为 CLOUD_OUTPUT_DIR 用于云端

# Column mappings
COLUMN_MAPPINGS = {
    'ad_feature': {'customer': 'customer_id'},
    'user_profile': {
        'userid': 'user_id',
        'new_user_class_level ': 'new_user_class_level'
    },
    'raw_sample': {'user': 'user_id', 'clk': 'label'}
}


def improved_sample_data_pipeline(data_path: str, output_dir: str = OUTPUT_DIR) -> None:
    """
    改进的采样数据管道 - 解决类别分布不一致问题
    
    Args:
        data_path: 原始数据路径
        output_dir: 输出目录
    """
    logger.info(f"开始改进的采样数据管道，采样比例: {SAMPLE_RATIO}")
    
    data_path = Path(data_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 处理小表
    clean_small_tables_sampled(data_path, output_path)
    
    # 2. 采样大表
    sample_large_file_improved(data_path, output_path)
    
    # 3. 创建最终数据集 - 使用分层采样
    create_stratified_datasets(output_path)
    
    logger.info("改进的采样数据管道完成！")


def clean_small_tables_sampled(data_path: Path, output_dir: Path) -> None:
    """处理小表（不采样，保持完整）"""
    logger.info("处理小表...")
    
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
    
    logger.info("小表处理完成")


def _standardize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """标准化列名"""
    df = df.copy()
    for old_name, new_name in mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    return df


def sample_large_file_improved(data_path: Path, output_dir: Path) -> None:
    """改进的大文件采样"""
    logger.info("开始改进的大文件采样...")
    
    # 加载参考表
    ad_feature = pd.read_csv(output_dir / 'ad_feature_cleaned.csv')
    user_profile = pd.read_csv(output_dir / 'user_profile_cleaned.csv')
    
    # 定义输出文件
    output_files = {
        'test': output_dir / 'test_samples_improved.csv',
        'train': output_dir / 'train_samples_improved.csv'
    }
    
    # 处理大文件
    raw_sample_path = data_path / 'raw_sample.csv'
    if not raw_sample_path.exists():
        raise FileNotFoundError(f"raw_sample.csv not found at {raw_sample_path}")
    
    # 计算总行数（用于进度条）
    total_lines = sum(1 for _ in open(raw_sample_path)) - 1  # 减去标题行
    logger.info(f"原始文件总行数: {total_lines}")
    
    # 采样处理
    reader = pd.read_csv(raw_sample_path, chunksize=SAMPLE_CHUNKSIZE)
    total_chunks = total_lines // SAMPLE_CHUNKSIZE + 1
    is_first_chunk = True
    
    for chunk in tqdm(reader, total=total_chunks, desc="改进采样处理chunks"):
        # 对每个chunk进行采样
        if len(chunk) > 0:
            sampled_chunk = chunk.sample(frac=SAMPLE_RATIO, random_state=RANDOM_SEED)
            if len(sampled_chunk) > 0:
                processed_chunk = _process_chunk(sampled_chunk, user_profile, ad_feature)
                _write_chunk_to_files_improved(processed_chunk, output_files, is_first_chunk)
                is_first_chunk = False
    
    logger.info("改进的大文件采样处理完成")


def _process_chunk(chunk: pd.DataFrame, user_profile: pd.DataFrame, ad_feature: pd.DataFrame) -> pd.DataFrame:
    """处理单个chunk"""
    # 标准化列名
    chunk = _standardize_columns(chunk, COLUMN_MAPPINGS['raw_sample'])
    
    # 移除不必要的列
    if 'noclk' in chunk.columns:
        chunk = chunk.drop(columns=['noclk'])
    
    # 合并参考表
    chunk = pd.merge(chunk, user_profile, on='user_id', how='left')
    chunk = pd.merge(chunk, ad_feature, on='adgroup_id', how='left')
    
    # 处理缺失值
    categorical_features = ['cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 
                           'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
                           'cate_id', 'campaign_id', 'customer_id', 'brand', 'pid']
    
    for feat in categorical_features:
        if feat in chunk.columns:
            chunk[feat] = chunk[feat].fillna(-1)
    
    numeric_features = ['price']
    for feat in numeric_features:
        if feat in chunk.columns:
            chunk[feat] = chunk[feat].fillna(0)
    
    # 转换时间戳
    chunk['day'] = pd.to_datetime(chunk['time_stamp'], unit='s').dt.strftime('%Y%m%d').astype(int)
    
    return chunk


def _write_chunk_to_files_improved(chunk: pd.DataFrame, output_files: dict, is_first_chunk: bool) -> None:
    """写入文件 - 改进版本"""
    # 分离测试数据
    test_data = pd.DataFrame(chunk[chunk['day'] == TEST_DATE])
    if not test_data.empty:
        test_data.drop(columns=['day', 'time_stamp']).to_csv(
            output_files['test'], mode='a', index=False, header=is_first_chunk
        )
    
    # 分离训练数据
    train_data = chunk[chunk['day'] < TEST_DATE]
    if not train_data.empty:
        train_data.drop(columns=['day', 'time_stamp']).to_csv(
            output_files['train'], mode='a', index=False, header=is_first_chunk
        )


def create_stratified_datasets(output_dir: Path) -> None:
    """创建分层采样的数据集 - 确保类别分布一致"""
    logger.info("创建分层采样的数据集...")
    
    # 加载所有数据
    train_data = pd.read_csv(output_dir / 'train_samples_improved.csv')
    test_data = pd.read_csv(output_dir / 'test_samples_improved.csv')
    
    logger.info(f"原始训练数据: {train_data.shape}, 正样本比例: {train_data['label'].mean():.4f}")
    logger.info(f"原始测试数据: {test_data.shape}, 正样本比例: {test_data['label'].mean():.4f}")
    
    # 合并所有数据用于分层采样
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    logger.info(f"合并后总数据: {all_data.shape}, 正样本比例: {all_data['label'].mean():.4f}")
    
    # 使用分层采样重新划分训练集和测试集
    train_df, test_df = train_test_split(
        all_data, 
        test_size=0.2,  # 20%作为测试集
        stratify=all_data['label'],  # 按标签分层
        random_state=RANDOM_SEED
    )
    
    logger.info(f"分层采样后训练数据: {train_df.shape}, 正样本比例: {train_df['label'].mean():.4f}")
    logger.info(f"分层采样后测试数据: {test_df.shape}, 正样本比例: {test_df['label'].mean():.4f}")
    
    # 保存最终数据集
    train_df.to_csv(output_dir / 'train_data_improved.csv', index=False)
    test_df.to_csv(output_dir / 'test_data_improved.csv', index=False)
    
    # 清理临时文件
    for temp_file in ['train_samples_improved.csv', 'test_samples_improved.csv']:
        if (output_dir / temp_file).exists():
            (output_dir / temp_file).unlink()
    
    logger.info("分层采样数据集创建完成！")


def improved_feature_engineering(train_df, test_df, output_dir):
    """改进的特征工程"""
    logger.info("开始改进的特征工程...")
    
    # 1. 价格特征清理和标准化
    if 'price' in train_df.columns:
        # 清理异常值
        price_q99 = train_df['price'].quantile(0.99)
        train_df['price_cleaned'] = train_df['price'].clip(upper=price_q99)
        test_df['price_cleaned'] = test_df['price'].clip(upper=price_q99)
        
        # 标准化
        scaler = StandardScaler()
        train_df['price_normalized'] = scaler.fit_transform(train_df[['price_cleaned']])
        test_df['price_normalized'] = scaler.transform(test_df[['price_cleaned']])
        
        # 保存scaler
        with open(os.path.join(output_dir, 'price_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
    # 2. 处理高基数特征
    high_cardinality_features = ['user_id', 'adgroup_id', 'campaign_id', 'customer_id']
    
    for feat in high_cardinality_features:
        if feat in train_df.columns:
            # 使用哈希编码减少维度
            train_df[f'{feat}_hash'] = train_df[feat].astype(str).apply(hash) % 1000
            test_df[f'{feat}_hash'] = test_df[feat].astype(str).apply(hash) % 1000
    
    # 3. 创建交互特征
    if 'age_level' in train_df.columns and 'final_gender_code' in train_df.columns:
        train_df['age_gender'] = train_df['age_level'].astype(str) + '_' + train_df['final_gender_code'].astype(str)
        test_df['age_gender'] = test_df['age_level'].astype(str) + '_' + test_df['final_gender_code'].astype(str)
    
    if 'cate_id' in train_df.columns and 'brand' in train_df.columns:
        train_df['cate_brand'] = train_df['cate_id'].astype(str) + '_' + train_df['brand'].astype(str)
        test_df['cate_brand'] = test_df['cate_id'].astype(str) + '_' + test_df['brand'].astype(str)
    
    # 4. 重新编码所有分类特征
    categorical_features = [
        'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 
        'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
        'cate_id', 'brand', 'pid', 'age_gender', 'cate_brand'
    ] + [f'{feat}_hash' for feat in high_cardinality_features]
    
    feature_encoders = {}
    
    for feat in categorical_features:
        if feat in train_df.columns:
            le = LabelEncoder()
            # 合并训练集和测试集的所有唯一值
            all_values = pd.concat([train_df[feat], test_df[feat]]).unique()
            le.fit(all_values)
            
            train_df[feat] = le.transform(train_df[feat])
            test_df[feat] = le.transform(test_df[feat])
            
            feature_encoders[feat] = le
    
    # 保存编码器
    with open(os.path.join(output_dir, 'improved_feature_encoders.pkl'), 'wb') as f:
        pickle.dump(feature_encoders, f)
    
    logger.info("改进的特征工程完成")
    return train_df, test_df, feature_encoders


if __name__ == "__main__":
    # 示例用法
    improved_sample_data_pipeline("../data/")
