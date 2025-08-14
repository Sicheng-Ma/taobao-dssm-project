#!/usr/bin/env python3
"""
过拟合修复训练脚本 - 解决严重过拟合问题

策略：
1. 简化模型架构
2. 增强正则化
3. 特征选择
4. 优化训练策略
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加src到路径
sys.path.append('..')

# 导入模块
from configs import sampling_config as config

def simplified_feature_engineering(train_df, test_df, output_dir):
    """简化的特征工程 - 减少复杂度"""
    logger.info("开始简化特征工程...")
    
    # 1. 基础价格特征处理
    if 'price' in train_df.columns:
        # 使用RobustScaler处理异常值
        robust_scaler = RobustScaler()
        train_df['price_robust'] = robust_scaler.fit_transform(train_df[['price']])
        test_df['price_robust'] = robust_scaler.transform(test_df[['price']])
        
        # 保存scalers
        with open(os.path.join(output_dir, 'simplified_robust_scaler.pkl'), 'wb') as f:
            pickle.dump(robust_scaler, f)
    
    # 2. 简化的交互特征 - 只保留最重要的
    if all(feat in train_df.columns for feat in ['age_level', 'final_gender_code']):
        train_df['age_gender'] = train_df['age_level'].astype(str) + '_' + train_df['final_gender_code'].astype(str)
        test_df['age_gender'] = test_df['age_level'].astype(str) + '_' + test_df['final_gender_code'].astype(str)
    
    # 3. 简化的统计特征
    if 'user_id' in train_df.columns:
        # 只保留最重要的用户统计
        user_stats = train_df.groupby('user_id').agg({
            'label': ['count', 'mean'],
            'price': ['mean']
        }).fillna(0)
        user_stats.columns = ['user_click_count', 'user_click_rate', 'user_avg_price']
        
        train_df = train_df.merge(user_stats, on='user_id', how='left')
        test_df = test_df.merge(user_stats, on='user_id', how='left')
        
        # 填充缺失值
        for col in user_stats.columns:
            train_df[col] = train_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)
    
    # 4. 重新编码核心分类特征
    categorical_features = [
        'cms_segid', 'final_gender_code', 'age_level', 'pvalue_level',
        'cate_id', 'brand', 'age_gender'
    ]
    
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
    
    # 5. 简化的特征选择
    numeric_features = ['price_robust', 'user_click_count', 'user_click_rate', 'user_avg_price']
    
    if len(numeric_features) > 0:
        # 选择最重要的数值特征
        selector = SelectKBest(score_func=f_classif, k=min(3, len(numeric_features)))
        X_train_numeric = train_df[numeric_features].fillna(0)
        X_test_numeric = test_df[numeric_features].fillna(0)
        
        X_train_selected = selector.fit_transform(X_train_numeric, train_df['label'])
        X_test_selected = selector.transform(X_test_numeric)
        
        selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
        
        # 将选择的特征添加到数据框
        for i, feat in enumerate(selected_features):
            train_df[f'{feat}_selected'] = X_train_selected[:, i]
            test_df[f'{feat}_selected'] = X_test_selected[:, i]
    
    # 保存编码器
    with open(os.path.join(output_dir, 'simplified_feature_encoders.pkl'), 'wb') as f:
        pickle.dump(feature_encoders, f)
    
    logger.info("简化特征工程完成")
    return train_df, test_df, feature_encoders

def create_simplified_model_config():
    """创建简化的模型配置"""
    return {
        'EMBEDDING_DIM': 32,  # 降低embedding维度
        'DNN_UNITS': [128, 64],  # 减少到2层
        'TEMP': 0.1,  # 适中的温度参数
        'LEARNING_RATE': 0.0001,  # 更小的学习率
        'BATCH_SIZE': 1024,  # 更大的批次大小
        'EPOCHS': 30,  # 减少训练轮数
        'DROPOUT_RATE': 0.5,  # 更强的dropout
        'L2_REG': 1e-3,  # 更强的L2正则化
        'BATCH_NORM': True,  # 保留批归一化
        'RESIDUAL': False  # 移除残差连接
    }

def simplified_dssm_model(user_feature_columns, item_feature_columns, config_dict):
    """简化的DSSM模型"""
    import tensorflow as tf
    from collections import namedtuple
    
    # 使用与src.model相同的namedtuple定义
    SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
    DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
    
    # 创建输入层
    input_layers = {}
    logger.info(f"创建简化输入层，特征列数量: {len(user_feature_columns + item_feature_columns)}")
    
    for feat in user_feature_columns + item_feature_columns:
        logger.info(f"处理特征: {feat.name}, 类型: {type(feat)}")
        if hasattr(feat, 'vocabulary_size'):  # SparseFeat
            input_layers[feat.name] = tf.keras.layers.Input(shape=(1,), name=feat.name)
            logger.info(f"创建SparseFeat输入层: {feat.name}")
        elif hasattr(feat, 'dimension'):  # DenseFeat
            input_layers[feat.name] = tf.keras.layers.Input(shape=(feat.dimension,), name=feat.name)
            logger.info(f"创建DenseFeat输入层: {feat.name}")
        else:
            logger.warning(f"未知特征类型: {type(feat)} for {feat}")
    
    logger.info(f"创建的输入层: {list(input_layers.keys())}")
    
    # 用户塔
    user_embeddings = []
    for feat in user_feature_columns:
        if hasattr(feat, 'vocabulary_size'):  # SparseFeat
            embedding = tf.keras.layers.Embedding(
                feat.vocabulary_size, 
                feat.embedding_dim, 
                name=f'user_emb_{feat.name}'
            )(input_layers[feat.name])
            user_embeddings.append(embedding)
    
    if user_embeddings:
        user_concat = tf.keras.layers.Concatenate(axis=-1)(user_embeddings)
        user_flat = tf.keras.layers.Flatten()(user_concat)
    else:
        user_flat = tf.keras.layers.Dense(32, activation='relu')(tf.keras.layers.Input(shape=(1,)))
    
    # 用户DNN - 简化版本
    user_dnn = user_flat
    for i, units in enumerate(config_dict['DNN_UNITS']):
        if config_dict['BATCH_NORM']:
            user_dnn = tf.keras.layers.BatchNormalization()(user_dnn)
        
        user_dnn = tf.keras.layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config_dict['L2_REG']),
            name=f'user_dnn_{i}'
        )(user_dnn)
        
        user_dnn = tf.keras.layers.Dropout(config_dict['DROPOUT_RATE'])(user_dnn)
    
    # 物品塔
    item_embeddings = []
    item_dense = []
    
    for feat in item_feature_columns:
        if hasattr(feat, 'vocabulary_size'):  # SparseFeat
            embedding = tf.keras.layers.Embedding(
                feat.vocabulary_size, 
                feat.embedding_dim, 
                name=f'item_emb_{feat.name}'
            )(input_layers[feat.name])
            item_embeddings.append(embedding)
        elif hasattr(feat, 'dimension'):  # DenseFeat
            item_dense.append(input_layers[feat.name])
    
    if item_embeddings:
        item_concat = tf.keras.layers.Concatenate(axis=-1)(item_embeddings)
        item_flat = tf.keras.layers.Flatten()(item_concat)
    else:
        item_flat = tf.keras.layers.Dense(32, activation='relu')(tf.keras.layers.Input(shape=(1,)))
    
    if item_dense:
        item_dense_concat = tf.keras.layers.Concatenate(axis=-1)(item_dense)
        item_all = tf.keras.layers.Concatenate(axis=-1)([item_flat, item_dense_concat])
    else:
        item_all = item_flat
    
    # 物品DNN - 简化版本
    item_dnn = item_all
    for i, units in enumerate(config_dict['DNN_UNITS']):
        if config_dict['BATCH_NORM']:
            item_dnn = tf.keras.layers.BatchNormalization()(item_dnn)
        
        item_dnn = tf.keras.layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config_dict['L2_REG']),
            name=f'item_dnn_{i}'
        )(item_dnn)
        
        item_dnn = tf.keras.layers.Dropout(config_dict['DROPOUT_RATE'])(item_dnn)
    
    # 计算相似度 - 使用Keras层
    user_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(user_dnn)
    item_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item_dnn)
    
    similarity = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True))([user_norm, item_norm])
    similarity = tf.keras.layers.Lambda(lambda x: x * config_dict['TEMP'])(similarity)
    
    # 输出层
    output = tf.keras.layers.Dense(1, activation='sigmoid')(similarity)
    
    # 构建模型
    model = tf.keras.Model(inputs=list(input_layers.values()), outputs=output)
    
    return model

def main():
    """主训练函数"""
    
    # 定义特征类型
    from collections import namedtuple
    SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
    DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
    
    # ==================== 路径配置 ====================
    # 本地训练路径配置
    LOCAL_DATA_PATH = '/Users/masicheng/Desktop/搜广推/taobao-dssm-project/data'
    LOCAL_OUTPUT_DIR = '/Users/masicheng/Desktop/搜广推/taobao-dssm-project/outputs'
    
    # 云端训练路径配置
    CLOUD_DATA_PATH = './data'
    CLOUD_OUTPUT_DIR = './outputs'
    
    # ==================== 路径切换说明 ====================
    # 本地快速训练（使用现有采样数据）：
    # DATA_PATH = LOCAL_DATA_PATH
    # OUTPUT_DIR = LOCAL_OUTPUT_DIR
    # 注意：需要先运行 local_sampling.py 生成采样数据
    
    # 云端完整训练（处理全部数据）：
    # DATA_PATH = CLOUD_DATA_PATH  
    # OUTPUT_DIR = CLOUD_OUTPUT_DIR
    # 注意：需要先运行 process_data.py 处理完整数据
    
    # 当前使用的路径配置（手动切换）
    DATA_PATH = LOCAL_DATA_PATH  # 切换为 CLOUD_DATA_PATH 用于云端
    OUTPUT_DIR = LOCAL_OUTPUT_DIR  # 切换为 CLOUD_OUTPUT_DIR 用于云端
    
    logger.info(f"=== 当前路径配置 ===")
    logger.info(f"数据路径: {DATA_PATH}")
    logger.info(f"输出路径: {OUTPUT_DIR}")
    logger.info("=== 过拟合修复的Taobao DSSM训练开始 ===")
    
    logger.info("=== 过拟合修复的Taobao DSSM训练开始 ===")
    
    # 步骤1: 加载改进的数据
    logger.info("\n步骤1: 加载改进的数据")
    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_data_improved.csv'))
    test_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_data_improved.csv'))
    
    logger.info(f"训练数据形状: {train_df.shape}")
    logger.info(f"测试数据形状: {test_df.shape}")
    
    # 步骤2: 简化特征工程
    logger.info("\n步骤2: 简化特征工程")
    train_df, test_df, feature_encoders = simplified_feature_engineering(
        train_df, test_df, OUTPUT_DIR
    )
    
    logger.info(f"简化特征工程后训练数据形状: {train_df.shape}")
    logger.info(f"简化特征工程后测试数据形状: {test_df.shape}")
    
    # 步骤3: 创建简化模型配置
    simplified_config = create_simplified_model_config()
    logger.info(f"简化模型配置: {simplified_config}")
    
    # 步骤4: 准备模型输入
    logger.info("\n步骤4: 准备模型输入")
    
    # 定义简化的特征列表
    user_features = [
        'cms_segid', 'final_gender_code', 'age_level', 'pvalue_level', 'age_gender'
    ]
    
    item_features = [
        'cate_id', 'brand', 'price_robust'
    ]
    
    # 添加数值特征
    numeric_features = [col for col in train_df.columns if col.endswith('_selected')]
    item_features.extend(numeric_features)
    
    # 过滤出实际存在的特征
    available_features = train_df.columns.tolist()
    user_features = [feat for feat in user_features if feat in available_features]
    item_features = [feat for feat in item_features if feat in available_features]
    
    logger.info(f"简化用户特征: {user_features}")
    logger.info(f"简化物品特征: {item_features}")
    
    # 创建特征列
    user_feature_columns = []
    for feat in user_features:
        if feat in train_df.columns:
            vocab_size = len(feature_encoders[feat].classes_) + 1
            user_feature_columns.append(
                SparseFeat(
                    name=feat, 
                    vocabulary_size=vocab_size, 
                    embedding_dim=simplified_config['EMBEDDING_DIM']
                )
            )
    
    item_feature_columns = []
    for feat in item_features:
        if feat in train_df.columns:
            if feat in numeric_features or feat in ['price_robust']:
                item_feature_columns.append(
                    DenseFeat(name=feat, dimension=1)
                )
            else:
                vocab_size = len(feature_encoders[feat].classes_) + 1
                item_feature_columns.append(
                    SparseFeat(
                        name=feat, 
                        vocabulary_size=vocab_size, 
                        embedding_dim=simplified_config['EMBEDDING_DIM']
                    )
                )
    
    logger.info(f"用户特征列数: {len(user_feature_columns)}")
    logger.info(f"物品特征列数: {len(item_feature_columns)}")
    
    # 步骤5: 构建简化模型
    logger.info("\n步骤5: 构建简化模型")
    dssm_model = simplified_dssm_model(
        user_feature_columns, 
        item_feature_columns, 
        simplified_config
    )
    
    dssm_model.summary()
    
    # 编译模型
    dssm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=simplified_config['LEARNING_RATE']),
        loss="binary_crossentropy", 
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info("简化模型编译成功！")
    
    # 步骤6: 准备训练数据
    logger.info("\n步骤6: 准备训练数据")
    
    # 创建模型输入字典
    all_features = user_features + item_features
    train_model_input = {name: train_df[name] for name in all_features if name in train_df.columns}
    train_label = train_df['label'].values
    test_model_input = {name: test_df[name] for name in all_features if name in test_df.columns}
    test_label = test_df['label'].values
    
    logger.info(f"训练样本数: {len(train_label)}")
    logger.info(f"测试样本数: {len(test_label)}")
    
    # 计算类别权重
    pos_weight = len(train_df[train_df['label']==0]) / len(train_df[train_df['label']==1])
    class_weight = {0: 1, 1: pos_weight}
    logger.info(f"类别权重: {class_weight}")
    
    # 步骤7: 训练模型
    logger.info("\n步骤7: 训练简化模型")
    
    # 回调函数 - 更激进的早停
    early_stopping = EarlyStopping(
        monitor='val_auc', 
        mode='max', 
        patience=5,  # 减少patience
        verbose=1, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # 更温和的学习率衰减
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'best_simplified_model.keras'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    logger.info("开始简化模型训练...")
    
    history = dssm_model.fit(
        train_model_input, 
        train_label, 
        batch_size=simplified_config['BATCH_SIZE'],
        epochs=simplified_config['EPOCHS'],
        validation_data=(test_model_input, test_label),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        class_weight=class_weight
    )
    
    logger.info("简化模型训练完成！")
    
    # 步骤8: 模型评估
    logger.info("\n步骤8: 模型评估")
    test_loss, test_auc, test_precision, test_recall = dssm_model.evaluate(test_model_input, test_label)
    logger.info(f"测试损失: {test_loss:.4f}")
    logger.info(f"测试AUC: {test_auc:.4f}")
    logger.info(f"测试精确率: {test_precision:.4f}")
    logger.info(f"测试召回率: {test_recall:.4f}")
    
    # 计算F1分数
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    logger.info(f"测试F1分数: {f1_score:.4f}")
    
    # 保存模型
    model_save_path = os.path.join(OUTPUT_DIR, 'simplified_dssm_model.keras')
    dssm_model.save(model_save_path)
    logger.info(f"简化模型保存到: {model_save_path}")
    
    # 保存训练历史
    history_save_path = os.path.join(OUTPUT_DIR, 'simplified_training_history.pkl')
    with open(history_save_path, 'wb') as f:
        pickle.dump(history.history, f)
    logger.info(f"简化训练历史保存到: {history_save_path}")
    
    # 输出性能总结
    print(f"\n=== 过拟合修复训练性能总结 ===")
    print(f"训练样本数: {len(train_label):,}")
    print(f"测试样本数: {len(test_label):,}")
    print(f"最终测试AUC: {test_auc:.4f}")
    print(f"最终测试精确率: {test_precision:.4f}")
    print(f"最终测试召回率: {test_recall:.4f}")
    print(f"最终测试F1分数: {f1_score:.4f}")
    print(f"训练轮数: {len(history.history['loss'])}")
    
    # 分析过拟合程度
    if len(history.history['auc']) > 0:
        final_train_auc = history.history['auc'][-1]
        overfitting_degree = final_train_auc - test_auc
        print(f"最终训练AUC: {final_train_auc:.4f}")
        print(f"过拟合程度: {overfitting_degree:.4f}")
        
        # 判断改进效果
        if test_auc > 0.55:
            print(f"\n🎉 成功！验证AUC提升到{test_auc:.4f}，显著改进！")
        elif test_auc > 0.54:
            print(f"\n✅ 有改善！验证AUC提升到{test_auc:.4f}，小幅改进。")
        else:
            print(f"\n⚠️  验证AUC为{test_auc:.4f}，需要进一步调优。")
        
        if overfitting_degree < 0.1:
            print(f"✅ 过拟合控制良好！过拟合程度: {overfitting_degree:.4f}")
        elif overfitting_degree < 0.2:
            print(f"⚠️  过拟合程度适中: {overfitting_degree:.4f}")
        else:
            print(f"❌ 过拟合仍然严重: {overfitting_degree:.4f}")
    
    logger.info("=== 过拟合修复训练完成 ===")
    
    return {
        'test_auc': test_auc,
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': f1_score,
        'train_samples': len(train_label),
        'test_samples': len(test_label),
        'model_path': model_save_path
    }

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n过拟合修复训练成功完成！")
        print(f"最终验证AUC: {results['test_auc']:.4f}")
        print(f"最终精确率: {results['test_precision']:.4f}")
        print(f"最终召回率: {results['test_recall']:.4f}")
        print(f"最终F1分数: {results['test_f1']:.4f}")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
