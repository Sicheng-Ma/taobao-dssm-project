# Taobao DSSM 项目使用指南

## 快速开始

### 本地快速训练（推荐用于验证）

1. **生成采样数据**
   ```bash
   cd scripts
   python local_sampling.py
   ```

2. **训练模型**
   ```bash
   python train.py
   ```

### 云端完整训练（推荐用于生产）

1. **处理完整数据**
   ```bash
   cd scripts
   python process_data.py
   ```

2. **训练模型**
   ```bash
   python train.py
   ```

## 路径配置说明

### 本地环境配置
所有脚本都预设了本地路径：
- **数据路径**: `/Users/masicheng/Desktop/搜广推/taobao-dssm-project/data`
- **输出路径**: `/Users/masicheng/Desktop/搜广推/taobao-dssm-project/outputs`

### 云端环境配置
切换到云端路径：
- **数据路径**: `./data`
- **输出路径**: `./outputs`

## 路径切换方法

### 在 scripts/train.py 中
```python
# 本地训练
DATA_PATH = LOCAL_DATA_PATH
OUTPUT_DIR = LOCAL_OUTPUT_DIR

# 云端训练
DATA_PATH = CLOUD_DATA_PATH
OUTPUT_DIR = CLOUD_OUTPUT_DIR
```

### 在 scripts/process_data.py 中
```python
# 本地处理（小内存）
DATA_PATH = LOCAL_DATA_PATH
OUTPUT_DIR = LOCAL_OUTPUT_DIR
BEHAVIOR_SAMPLE_RATIO = 0.1
CHUNK_SIZE = 100000

# 云端处理（大内存）
DATA_PATH = CLOUD_DATA_PATH
OUTPUT_DIR = CLOUD_OUTPUT_DIR
BEHAVIOR_SAMPLE_RATIO = 1.0
CHUNK_SIZE = 2000000
```

### 在 scripts/local_sampling.py 中
```python
# 本地采样
DATA_PATH = LOCAL_DATA_PATH
OUTPUT_DIR = LOCAL_OUTPUT_DIR

# 云端采样
DATA_PATH = CLOUD_DATA_PATH
OUTPUT_DIR = CLOUD_OUTPUT_DIR
```

### 在 configs/sampling_config.py 中
```python
# 本地配置
OUTPUT_DIR = LOCAL_OUTPUT_DIR

# 云端配置
OUTPUT_DIR = CLOUD_OUTPUT_DIR
```

## 环境要求

### 本地环境
- **内存**: 8GB+ (推荐16GB)
- **存储**: 100GB+
- **Python**: 3.8+
- **依赖**: pandas, numpy, scikit-learn, tensorflow

### 云端环境
- **内存**: 16GB+ (推荐32GB)
- **存储**: 200GB+
- **GPU**: 推荐使用GPU加速
- **平台**: Google Colab, Kaggle, AWS, GCP

## 文件说明

### 核心脚本
- `scripts/train.py` - 主训练脚本（包含DSSM模型实现）
- `scripts/process_data.py` - 完整数据处理脚本
- `scripts/local_sampling.py` - 本地采样脚本

### 配置文件
- `configs/sampling_config.py` - 采样和训练配置

### 实验模块
- `experiments/upgrades/` - 高级功能模块

## 注意事项

1. **数据文件**: 大文件（behavior_log.csv等）已通过.gitignore忽略
2. **内存管理**: 本地环境建议使用采样数据
3. **路径切换**: 每次切换环境时记得修改路径配置
4. **依赖安装**: 确保所有Python依赖已正确安装

## 故障排除

### 内存不足
- 使用本地采样配置
- 减少CHUNK_SIZE
- 降低BEHAVIOR_SAMPLE_RATIO

### 路径错误
- 检查DATA_PATH和OUTPUT_DIR配置
- 确保数据文件存在于指定路径

### 导入错误
- 确保在scripts目录下运行脚本
- 检查Python路径配置
