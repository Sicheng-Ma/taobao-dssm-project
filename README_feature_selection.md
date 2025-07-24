# LightGBM 特征重要性分析与特征选择

本项目提供了基于LightGBM的特征重要性分析和特征选择功能，帮助您识别和选择最相关的特征。

## 功能特点

- 🔍 **特征重要性分析**: 使用LightGBM的增益重要性指标
- 📊 **可视化分析**: 自动生成特征重要性图表
- 🎯 **多种选择策略**: 支持阈值选择和Top-K选择
- 💾 **结果持久化**: 保存分析结果和选择的特征
- 📈 **性能对比**: 比较不同选择策略的效果

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```python
from src.feature_selection import perform_feature_selection
from src.config import SPARSE_FEATURES, DENSE_FEATURES

# 执行特征选择
selected_train, selected_test, selected_features = perform_feature_selection(
    train_df=train_df,
    test_df=test_df,
    target_col='label',
    sparse_features=SPARSE_FEATURES,
    dense_features=DENSE_FEATURES,
    output_dir='./output/feature_selection',
    importance_threshold=0.01,  # 保留重要性 >= 1% 的特征
    top_k_features=None
)
```

### 2. 使用LightGBMFeatureSelector类

```python
from src.feature_selection import LightGBMFeatureSelector

# 初始化选择器
selector = LightGBMFeatureSelector(
    importance_threshold=0.01,
    top_k_features=None,
    random_state=42
)

# 拟合模型
selector.fit(X_train, y_train, categorical_features=sparse_features)

# 获取特征重要性
importance_df = selector.get_feature_importance()

# 选择特征
selected_features = selector.selected_features

# 转换数据
X_selected = selector.transform(X_test)

# 可视化
selector.plot_feature_importance(top_n=20)
```

## 参数说明

### LightGBMFeatureSelector 参数

- `importance_threshold` (float): 特征重要性阈值，默认0.001
- `top_k_features` (int, optional): 选择前K个最重要的特征
- `random_state` (int): 随机种子，默认42

### perform_feature_selection 参数

- `train_df` (DataFrame): 训练数据
- `test_df` (DataFrame): 测试数据
- `target_col` (str): 目标列名
- `sparse_features` (list): 稀疏特征列表
- `dense_features` (list): 密集特征列表
- `output_dir` (str): 输出目录
- `importance_threshold` (float): 重要性阈值
- `top_k_features` (int, optional): 前K个特征
- `**lgb_params`: 额外的LightGBM参数

## 输出文件

特征选择完成后，会在输出目录生成以下文件：

- `feature_importance.csv`: 详细的特征重要性分数
- `selected_features.pkl`: 选择的特征列表
- `lightgbm_feature_selector.pkl`: 训练好的LightGBM模型
- `feature_importance.png`: 特征重要性可视化图表

## 使用示例

### 示例1: 阈值选择策略

```python
# 保留重要性 >= 0.5% 的特征
selected_train, selected_test, selected_features = perform_feature_selection(
    train_df=train_df,
    test_df=test_df,
    target_col='label',
    sparse_features=SPARSE_FEATURES,
    dense_features=DENSE_FEATURES,
    output_dir='./output/threshold_selection',
    importance_threshold=0.005,
    top_k_features=None
)
```

### 示例2: Top-K选择策略

```python
# 选择前10个最重要的特征
selected_train, selected_test, selected_features = perform_feature_selection(
    train_df=train_df,
    test_df=test_df,
    target_col='label',
    sparse_features=SPARSE_FEATURES,
    dense_features=DENSE_FEATURES,
    output_dir='./output/top_k_selection',
    importance_threshold=0.0,
    top_k_features=10
)
```

### 示例3: 加载已保存的结果

```python
from src.feature_selection import load_feature_selection_results

# 加载之前保存的结果
selected_features, feature_importance = load_feature_selection_results(
    './output/feature_selection'
)

print(f"选择的特征: {selected_features}")
print(f"特征重要性:\n{feature_importance.head()}")
```

## Jupyter Notebook 使用

运行 `notebooks/feature_selection_analysis.ipynb` 来获得完整的交互式分析体验：

```bash
jupyter notebook notebooks/feature_selection_analysis.ipynb
```

该notebook包含：
- 数据加载和预处理
- 特征重要性分析
- 可视化图表
- 不同策略对比
- 结果应用示例

## 命令行使用

运行示例脚本：

```bash
python feature_selection_example.py
```

## 特征选择策略建议

### 1. 阈值选择策略
- **严格选择 (0.01)**: 只保留重要性 >= 1% 的特征
- **中等选择 (0.005)**: 保留重要性 >= 0.5% 的特征  
- **宽松选择 (0.001)**: 保留重要性 >= 0.1% 的特征

### 2. Top-K选择策略
- 根据业务需求选择固定数量的特征
- 适合需要严格控制特征数量的场景

### 3. 混合策略
- 先使用阈值筛选，再选择Top-K
- 平衡特征数量和重要性

## 性能优化建议

1. **数据预处理**: 确保类别特征正确编码
2. **参数调优**: 根据数据规模调整LightGBM参数
3. **内存管理**: 对于大数据集，考虑分批处理
4. **并行计算**: 利用LightGBM的并行训练能力

## 注意事项

1. **数据质量**: 确保输入数据没有严重的缺失值或异常值
2. **特征编码**: 类别特征需要正确编码为数值
3. **过拟合**: 避免在特征选择过程中过拟合
4. **业务理解**: 结合业务知识解释特征重要性

## 故障排除

### 常见问题

1. **内存不足**: 减少批次大小或使用更小的数据集
2. **特征缺失**: 检查特征名称是否正确
3. **类别特征错误**: 确保类别特征正确编码
4. **模型收敛**: 调整学习率和迭代次数

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据
print(f"训练数据形状: {train_df.shape}")
print(f"特征列表: {sparse_features + dense_features}")
print(f"目标列分布: {train_df[target_col].value_counts()}")
```

## 贡献

欢迎提交Issue和Pull Request来改进这个特征选择模块！

## 许可证

本项目采用MIT许可证。 