# src/config_sampling.py
# 采样配置 - 用于本地快速训练

import os

# --- 采样配置 ---
SAMPLE_RATIO = 0.1  # 采样10%的数据
SAMPLE_CHUNKSIZE = 500_000  # 每次处理50万行（减少内存占用）
SAMPLE_NEG_POS_RATIO = 2  # 减少负样本比例

# --- 文件路径 ---
DATASET_FOLDER_NAME = 'ad-displayclick-data-on-taobao-com-cheneymaaa'
OUTPUT_DIR = '../outputs'  # 输出目录

# --- 超参数（适合快速训练） ---
EMBEDDING_DIM = 16
DNN_UNITS = [64, 32]  # 减少网络层数和维度
TEMP = 1.0
LEARNING_RATE = 0.001
BATCH_SIZE = 2048  # 减少批次大小
EPOCHS = 5  # 减少训练轮数

# --- 特征列表 ---
USER_SPARSE_FEATURES = ['user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 
                        'age_level', 'pvalue_level', 'shopping_level', 
                        'occupation', 'new_user_class_level']

ITEM_SPARSE_FEATURES = ['adgroup_id', 'cate_id', 'campaign_id', 'customer_id', 'brand', 'pid']
ITEM_DENSE_FEATURES = ['price']

SPARSE_FEATURES = USER_SPARSE_FEATURES + ITEM_SPARSE_FEATURES
DENSE_FEATURES = ITEM_DENSE_FEATURES
ALL_FEATURES = SPARSE_FEATURES + DENSE_FEATURES
