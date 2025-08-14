#!/usr/bin/env python3
"""
完整的数据处理脚本 - 包含behavior_log的处理（云端优化版本）

这是预测用户点击概率的核心特征！
云端处理版本，适合Google Colab、Kaggle等平台。
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import gc
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
TEST_DATE = 20170513
RANDOM_SEED = 42

# ==================== 路径配置 ====================
# 本地处理路径配置
LOCAL_DATA_PATH = '/Users/masicheng/Desktop/搜广推/taobao-dssm-project/data'
LOCAL_OUTPUT_DIR = '/Users/masicheng/Desktop/搜广推/taobao-dssm-project/outputs'

# 云端处理路径配置
CLOUD_DATA_PATH = './data'
CLOUD_OUTPUT_DIR = './outputs'

    # ==================== 路径切换说明 ====================
    # 本地处理（小内存环境）：
    # DATA_PATH = LOCAL_DATA_PATH
    # OUTPUT_DIR = LOCAL_OUTPUT_DIR
    # 同时需要启用本地处理配置：
    # BEHAVIOR_SAMPLE_RATIO = 0.1
    # CHUNK_SIZE = 100000
    
    # 云端处理（大内存环境）：
    # DATA_PATH = CLOUD_DATA_PATH
    # OUTPUT_DIR = CLOUD_OUTPUT_DIR
    # 同时使用云端处理配置：
    # BEHAVIOR_SAMPLE_RATIO = 1.0
    # CHUNK_SIZE = 2000000
    
# 当前使用的路径配置（手动切换）
DATA_PATH = LOCAL_DATA_PATH  # 切换为 CLOUD_DATA_PATH 用于云端
OUTPUT_DIR = LOCAL_OUTPUT_DIR  # 切换为 CLOUD_OUTPUT_DIR 用于云端

# ==================== 处理配置 ====================
# 云端处理配置（适合大内存环境）
BEHAVIOR_SAMPLE_RATIO = 1.0  # 处理100%的行为数据
CHUNK_SIZE = 2000000  # 200万行一批，适合云端内存

# 本地处理配置（适合小内存环境）
# BEHAVIOR_SAMPLE_RATIO = 0.1  # 处理10%的行为数据
# CHUNK_SIZE = 100000  # 10万行一批，适合本地内存

def process_behavior_log(data_path: str, output_dir: str) -> None:
    """处理用户行为日志 - 云端优化版本"""
    logger.info(f"开始处理用户行为日志（云端版本）...")
    
    behavior_path = Path(data_path) / 'behavior_log.csv'
    if not behavior_path.exists():
        logger.error(f"behavior_log.csv not found at {behavior_path}")
        return
    
    # 存储用户行为统计
    user_behavior_stats = defaultdict(lambda: {
        'ipv_count': 0, 'cart_count': 0, 'fav_count': 0, 'buy_count': 0,
        'total_actions': 0, 'unique_cates': set(), 'unique_brands': set(),
        'last_action_time': 0
    })
    
    # 存储用户-类目交互
    user_cate_interactions = defaultdict(lambda: defaultdict(int))
    user_brand_interactions = defaultdict(lambda: defaultdict(int))
    
    logger.info("开始读取behavior_log.csv（云端版本）...")
    
    # 分块读取大文件
    processed_lines = 0
    chunk_count = 0
    
    for chunk in pd.read_csv(behavior_path, chunksize=CHUNK_SIZE):
        chunk_count += 1
        logger.info(f"处理第{chunk_count}个chunk，形状: {chunk.shape}")
        
        # 标准化列名
        chunk.columns = [col.strip() for col in chunk.columns]
        
        # 处理每一行
        for _, row in chunk.iterrows():
            user_id = row['user']
            btag = row['btag']
            cate = row['cate']
            brand = row['brand']
            time_stamp = row['time_stamp']
            
            # 更新用户行为统计
            user_behavior_stats[user_id]['total_actions'] += 1
            user_behavior_stats[user_id]['last_action_time'] = max(
                user_behavior_stats[user_id]['last_action_time'], time_stamp
            )
            
            # 统计不同行为类型
            if btag == 'ipv':
                user_behavior_stats[user_id]['ipv_count'] += 1
            elif btag == 'cart':
                user_behavior_stats[user_id]['cart_count'] += 1
            elif btag == 'fav':
                user_behavior_stats[user_id]['fav_count'] += 1
            elif btag == 'buy':
                user_behavior_stats[user_id]['buy_count'] += 1
            
            # 记录交互的类目和品牌
            user_behavior_stats[user_id]['unique_cates'].add(cate)
            user_behavior_stats[user_id]['unique_brands'].add(brand)
            
            # 统计用户-类目交互
            user_cate_interactions[user_id][cate] += 1
            user_brand_interactions[user_id][brand] += 1
        
        processed_lines += len(chunk)
        logger.info(f"已处理 {processed_lines:,} 行")
        
        # 清理内存
        del chunk
        gc.collect()
    
    logger.info("行为日志处理完成，开始创建特征...")
    
    # 创建用户行为特征DataFrame
    user_behavior_features = []
    
    for user_id, stats in user_behavior_stats.items():
        features = {
            'user_id': user_id,
            'total_actions': stats['total_actions'],
            'ipv_count': stats['ipv_count'],
            'cart_count': stats['cart_count'],
            'fav_count': stats['fav_count'],
            'buy_count': stats['buy_count'],
            'unique_cates_count': len(stats['unique_cates']),
            'unique_brands_count': len(stats['unique_brands']),
            'last_action_time': stats['last_action_time'],
            'buy_rate': stats['buy_count'] / max(stats['total_actions'], 1),
            'cart_rate': stats['cart_count'] / max(stats['total_actions'], 1),
            'fav_rate': stats['fav_count'] / max(stats['total_actions'], 1)
        }
        user_behavior_features.append(features)
    
    # 转换为DataFrame
    behavior_df = pd.DataFrame(user_behavior_features)
    
    # 保存用户行为特征
    behavior_df.to_csv(Path(output_dir) / 'user_behavior_features.csv', index=False)
    logger.info(f"用户行为特征保存完成，形状: {behavior_df.shape}")
    
    # 保存用户-类目交互（用于后续特征工程）
    with open(Path(output_dir) / 'user_cate_interactions.pkl', 'wb') as f:
        pickle.dump(dict(user_cate_interactions), f)
    
    with open(Path(output_dir) / 'user_brand_interactions.pkl', 'wb') as f:
        pickle.dump(dict(user_brand_interactions), f)
    
    logger.info("用户行为日志处理完成！")

def create_complete_dataset(data_path: str, output_dir: str) -> None:
    """创建完整的数据集，包含所有特征（云端版本）"""
    logger.info("开始创建完整数据集（云端版本）...")
    
    data_path = Path(data_path)
    output_path = Path(output_dir)
    
    # 1. 处理基础表
    logger.info("处理基础表...")
    
    # 处理ad_feature
    ad_feature = pd.read_csv(data_path / 'ad_feature.csv')
    ad_feature.columns = [col.strip() for col in ad_feature.columns]
    ad_feature.to_csv(output_path / 'ad_feature_cleaned.csv', index=False)
    
    # 处理user_profile
    user_profile = pd.read_csv(data_path / 'user_profile.csv')
    user_profile.columns = [col.strip() for col in user_profile.columns]
    # 标准化列名
    user_profile = user_profile.rename(columns={
        'userid': 'user_id',
        'new_user_class_level ': 'new_user_class_level'
    })
    user_profile.to_csv(output_path / 'user_profile_cleaned.csv', index=False)
    
    # 2. 处理raw_sample（云端版本）
    logger.info("处理raw_sample（云端版本）...")
    
    # 分块读取raw_sample
    raw_sample_chunks = []
    
    for chunk in pd.read_csv(data_path / 'raw_sample.csv', chunksize=CHUNK_SIZE):
        logger.info(f"处理raw_sample chunk，形状: {chunk.shape}")
        
        chunk.columns = [col.strip() for col in chunk.columns]
        # 标准化列名
        chunk = chunk.rename(columns={
            'user': 'user_id',
            'clk': 'label'
        })
        
        raw_sample_chunks.append(chunk)
        
        # 清理内存
        del chunk
        gc.collect()
    
    raw_sample = pd.concat(raw_sample_chunks, ignore_index=True)
    logger.info(f"raw_sample形状: {raw_sample.shape}")
    
    # 3. 合并所有数据
    logger.info("合并数据...")
    
    # 合并用户特征
    merged_data = raw_sample.merge(user_profile, on='user_id', how='left')
    logger.info(f"合并用户特征后形状: {merged_data.shape}")
    
    # 合并广告特征
    merged_data = merged_data.merge(ad_feature, on='adgroup_id', how='left')
    logger.info(f"合并广告特征后形状: {merged_data.shape}")
    
    # 合并用户行为特征
    behavior_file = output_path / 'user_behavior_features.csv'
    if behavior_file.exists():
        behavior_features = pd.read_csv(behavior_file)
        merged_data = merged_data.merge(behavior_features, on='user_id', how='left')
        logger.info(f"合并行为特征后形状: {merged_data.shape}")
        
        # 填充缺失值
        behavior_cols = ['total_actions', 'ipv_count', 'cart_count', 'fav_count', 'buy_count',
                        'unique_cates_count', 'unique_brands_count', 'buy_rate', 'cart_rate', 'fav_rate']
        for col in behavior_cols:
            if col in merged_data.columns:
                merged_data[col] = merged_data[col].fillna(0)
    else:
        logger.warning("用户行为特征文件不存在，跳过行为特征合并")
    
    # 4. 时间分割
    logger.info("进行时间分割...")
    
    # 转换时间戳
    merged_data['date'] = pd.to_datetime(merged_data['time_stamp'], unit='s').dt.date
    merged_data['date'] = merged_data['date'].astype(str).str.replace('-', '').astype(int)
    
    # 分割训练集和测试集
    train_data = merged_data[merged_data['date'] < TEST_DATE].copy()
    test_data = merged_data[merged_data['date'] == TEST_DATE].copy()
    
    logger.info(f"训练集形状: {train_data.shape}")
    logger.info(f"测试集形状: {test_data.shape}")
    
    # 5. 分层采样确保类别分布一致
    logger.info("进行分层采样...")
    
    # 合并所有数据用于分层采样
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # 使用分层采样重新划分训练集和测试集
    train_df, test_df = train_test_split(
        all_data,
        test_size=0.2,  # 20%作为测试集
        stratify=all_data['label'],  # 按标签分层
        random_state=RANDOM_SEED
    )
    
    logger.info(f"分层采样后训练集形状: {train_df.shape}")
    logger.info(f"分层采样后测试集形状: {test_df.shape}")
    
    # 检查类别分布
    train_pos_ratio = train_df['label'].mean()
    test_pos_ratio = test_df['label'].mean()
    logger.info(f"训练集正样本比例: {train_pos_ratio:.4f}")
    logger.info(f"测试集正样本比例: {test_pos_ratio:.4f}")
    
    # 6. 保存最终数据集
    train_df.to_csv(output_path / 'train_data_complete.csv', index=False)
    test_df.to_csv(output_path / 'test_data_complete.csv', index=False)
    
    logger.info("完整数据集创建完成！")

def main():
    """主函数"""
    data_path = DATA_PATH
    output_dir = OUTPUT_DIR
    
    logger.info("=== 当前路径配置 ===")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"输出路径: {output_dir}")
    logger.info(f"行为数据处理比例: {BEHAVIOR_SAMPLE_RATIO}")
    logger.info(f"Chunk大小: {CHUNK_SIZE}")
    logger.info("=== 开始完整数据处理 ===")
    
    # 1. 处理用户行为日志
    process_behavior_log(data_path, output_dir)
    
    # 2. 创建完整数据集
    create_complete_dataset(data_path, output_dir)
    
    logger.info("=== 完整数据处理完成（云端版本） ===")

if __name__ == "__main__":
    main()
