# src/config.py

import os

# --- 常量定义 ---

# 1. 文件路径
# 在Kaggle环境中，输入路径通常是固定的
DATASET_FOLDER_NAME = 'ad-displayclick-data-on-taobao-com-cheneymaaa'
# 我们在主Notebook中动态生成这个路径，这里只定义文件夹名
# DATA_PATH = f'../input/{DATASET_FOLDER_NAME}/' 

# 所有中间和最终文件都将保存在Kaggle的输出目录
OUTPUT_DIR = '/kaggle/working/'

# 2. 超参数
CHUNKSIZE = 2_000_000  # 每次处理200万行
NEG_POS_RATIO = 4      # 负样本与正样本的比例
EMBEDDING_DIM = 16     # Embedding维度
DNN_UNITS = [128, 64, 32] # DNN层级
TEMP = 1.0             # 余弦相似度温度系数
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
EPOCHS = 10

# 3. 特征列表
# 用户塔特征
USER_SPARSE_FEATURES = ['user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 
                        'age_level', 'pvalue_level', 'shopping_level', 
                        'occupation', 'new_user_class_level']
# 物品塔特征
ITEM_SPARSE_FEATURES = ['adgroup_id', 'cate_id', 'campaign_id', 'customer_id', 'brand', 'pid']
ITEM_DENSE_FEATURES = ['price']

# 合并特征
SPARSE_FEATURES = USER_SPARSE_FEATURES + ITEM_SPARSE_FEATURES
DENSE_FEATURES = ITEM_DENSE_FEATURES
ALL_FEATURES = SPARSE_FEATURES + DENSE_FEATURES