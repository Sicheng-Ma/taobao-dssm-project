# configs/sampling_config.py
# 采样配置 - 用于本地快速训练

import os

# ==================== 采样配置 ====================
SAMPLE_RATIO = 0.1  # 采样10%的数据
SAMPLE_CHUNKSIZE = 500_000  # 每次处理50万行（减少内存占用）
SAMPLE_NEG_POS_RATIO = 2  # 减少负样本比例

# ==================== 路径配置 ====================
DATASET_FOLDER_NAME = 'ad-displayclick-data-on-taobao-com-cheneymaaa'

# 本地路径配置
LOCAL_OUTPUT_DIR = '/Users/masicheng/Desktop/搜广推/taobao-dssm-project/outputs'

# 云端路径配置
CLOUD_OUTPUT_DIR = './outputs'

# 当前使用的路径配置（手动切换）
# 本地训练：使用 LOCAL_OUTPUT_DIR
# 云端训练：使用 CLOUD_OUTPUT_DIR
OUTPUT_DIR = LOCAL_OUTPUT_DIR  # 切换为 CLOUD_OUTPUT_DIR 用于云端

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
