# 话题极性条件下针对用户特征向量获取

基于深度学习的微博用户网络分析系统，集成情感分析、图特征提取和边预测功能。

## 📋 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置文件](#配置文件)
- [输出结果](#输出结果)
- [模型架构](#模型架构)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

## 🎯 项目概述

本项目是一个端到端的微博用户网络分析系统，主要用于：

- **情感分析**：基于多模型融合的中文微博文本情感分析
- **网络建模**：用户互动关系的图结构建模
- **链接预测**：使用图注意力网络（GAT）进行用户关系预测
- **特征工程**：集成文本嵌入、情感特征和网络拓扑特征

## ✨ 核心功能

### 1. 情感分析模块
- 支持多个预训练中文情感分析模型
- 加权融合多模型预测结果
- 计算情感分数、波动性和标签
- 批量处理和GPU加速

### 2. 数据预处理模块
- 图特征提取（PageRank、度中心性等）
- 文本嵌入降维（PCA）
- 多难度负样本采样策略
- 数据集划分和标准化

### 3. GAT训练模块
- 改进的图注意力网络架构
- 三阶段渐进式训练策略
- 早停机制和学习率调度
- 模型检查点保存

## 🏗️ 系统架构

```
微博网络数据
    ↓
情感分析模块 → 情感特征提取
    ↓
数据预处理模块 → 图特征工程 + 负样本采样
    ↓
GAT训练模块 → 边预测模型训练
    ↓
结果输出 → 模型检查点 + 性能报告 + 用户嵌入
```

## 🚀 安装说明

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA (可选，用于GPU加速)

### 依赖安装

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers
pip install scikit-learn pandas numpy tqdm networkx
```

### GPU环境配置

```bash
# 为CUDA 11.8安装PyTorch Geometric
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu118.html
```

## 🚀 快速开始

### 1. 基本使用

```python
from sentiment_analyzer import SentimentAnalyzer
from graph_preprocessor import main as preprocess_data
from gat_trainer import train_all_stages
from config import create_custom_config

# 创建配置
config = create_custom_config(
    raw_network_path="/path/to/your/weibo_network.pkl",
    output_dir="./results",
    epochs_baseline=100,
    learning_rate=0.003
)

# 1. 情感分析
analyzer = SentimentAnalyzer(config.paths.raw_network_path)
network, sentiment_df = analyzer.run_sentiment_analysis(
    results_csv_path="sentiment_results.csv"
)

# 2. 数据预处理
results, base_path, timestamp = preprocess_data()

# 3. GAT训练
stage_files = {
    'baseline': f"presampled_baseline_{timestamp}.pkl",
    'medium': f"presampled_medium_{timestamp}.pkl", 
    'advanced': f"presampled_advanced_{timestamp}.pkl"
}

training_results = train_all_stages(
    base_data_path=base_path,
    stage_files=stage_files,
    output_dir=config.paths.training_output_dir
)
```

### 2. 命令行使用

```bash
# 运行完整流程
python main.py --config config.yaml --input data/network.pkl --output results/

# 仅运行情感分析
python sentiment_analyzer.py --input data/network.pkl --output results/sentiment/

# 仅运行数据预处理
python graph_preprocessor.py --input data/network.pkl --output results/preprocessing/

# 仅运行GAT训练
python gat_trainer.py --base_data results/base_data.pkl --stage_files results/stages/ --output results/training/
```

## 📖 详细使用说明

### 情感分析模块

```python
from sentiment_analyzer import SentimentAnalyzer

# 初始化分析器
analyzer = SentimentAnalyzer("/path/to/network.pkl")

# 运行情感分析
network, results_df = analyzer.run_sentiment_analysis(
    output_path="updated_network.pkl",  # 可选：指定输出路径
    results_csv_path="sentiment_analysis.csv"  # 可选：保存详细结果
)

# 查看情感分布
print(results_df['sentiment_label'].value_counts())
print(f"平均情感分数: {results_df['sentiment_score'].mean():.3f}")
```

**输出特征**：
- `sentiment_score`: 0-1范围的情感分数（0=负面，1=正面）
- `sentiment_label`: 分类标签（negative/neutral/positive）
- `sentiment_volatility`: 情感波动性（标准差）

### 数据预处理模块

```python
from graph_preprocessor import load_and_prepare_data, create_presampled_datasets

# 加载和准备数据
data = load_and_prepare_data("/path/to/network.pkl")

# 创建预采样数据集
results, base_path, timestamp = create_presampled_datasets(
    data, output_dir="./preprocessing_output"
)
```

**负样本采样策略**：
- **Baseline**: 纯随机采样
- **Medium**: 50%随机 + 30%中等难度 + 20%困难
- **Advanced**: 20%随机 + 30%中等难度 + 50%困难

### GAT训练模块

```python
from gat_trainer import train_single_stage, train_all_stages

# 训练单个阶段
history, test_results, model_path, embeddings = train_single_stage(
    base_data_path="base_data.pkl",
    stage_data_path="stage_data.pkl", 
    output_dir="./training_output",
    stage_name="baseline"
)

# 训练所有阶段（推荐）
results = train_all_stages(
    base_data_path="base_data.pkl",
    stage_files={
        'baseline': "baseline_stage.pkl",
        'medium': "medium_stage.pkl",
        'advanced': "advanced_stage.pkl"
    },
    output_dir="./training_output"
)
```

## ⚙️ 配置文件

### 基本配置

```python
from config import Config, create_custom_config

# 使用默认配置
config = Config()

# 创建自定义配置
config = create_custom_config(
    raw_network_path="/path/to/data.pkl",
    output_dir="./results",
    epochs_baseline=120,
    epochs_medium=150,
    epochs_advanced=180,
    learning_rate=0.003,
    batch_size=4096
)
```

### 配置参数说明

| 参数类别 | 参数名 | 默认值 | 说明 |
|---------|--------|--------|------|
| **模型配置** | `hidden_channels` | 64 | GAT隐藏层维度 |
| | `out_channels` | 32 | 输出嵌入维度 |
| | `heads` | 4 | 注意力头数 |
| | `dropout` | 0.3 | Dropout比率 |
| **训练配置** | `epochs` | 120/150/180 | 各阶段训练轮数 |
| | `learning_rate` | 0.003/0.002/0.001 | 各阶段学习率 |
| | `batch_size` | 4096 | 边批处理大小 |
| | `patience` | 25/30/35 | 早停耐心值 |
| **数据配置** | `train_ratio` | 0.7 | 训练集比例 |
| | `val_ratio` | 0.15 | 验证集比例 |
| | `test_ratio` | 0.15 | 测试集比例 |

## 📊 输出结果

### 文件结构

```
output/
├── sentiment/
│   └── sentiment_analysis_results.csv     # 情感分析结果
├── preprocessing/
│   ├── base_data_YYYYMMDD_HHMMSS.pkl     # 基础图数据
│   ├── presampled_baseline_YYYYMMDD_HHMMSS.pkl    # 基线采样数据
│   ├── presampled_medium_YYYYMMDD_HHMMSS.pkl      # 中等采样数据
│   └── presampled_advanced_YYYYMMDD_HHMMSS.pkl    # 高级采样数据
└── training/
    ├── baseline_training_history_YYYYMMDD_HHMMSS.csv    # 训练历史
    ├── baseline_test_results_YYYYMMDD_HHMMSS.csv        # 测试结果
    ├── baseline_best_user_embeddings_YYYYMMDD_HHMMSS.csv # 用户嵌入
    ├── baseline_best_model_YYYYMMDD_HHMMSS.pth          # 模型检查点
    └── all_stages_comparison_YYYYMMDD_HHMMSS.csv        # 阶段对比
```

### 性能指标

- **Accuracy**: 分类准确率
- **F1-Score**: F1分数（主要评估指标）
- **Precision**: 精确率
- **Recall**: 召回率
- **AUC**: ROC曲线下面积

### 用户嵌入

训练完成后，每个用户会得到一个32维的向量表示，包含：
- 文本语义信息
- 情感特征
- 网络拓扑特征
- 用户行为模式

## 🧠 模型架构

### GAT网络结构

```
输入特征 (132维)
    ↓
GAT层1 (64×4头) + ELU + Dropout
    ↓  
GAT层2 (32×1头) + Dropout
    ↓
节点嵌入 (32维)
    ↓
边预测器 (MLP + BatchNorm + Dropout)
    ↓
边存在概率
```

### 特征组成

| 特征类型 | 维度 | 来源 |
|---------|------|------|
| PCA嵌入 | 128维 | 原始文本嵌入降维 |
| 情感分数 | 1维 | 多模型融合情感分析 |
| 情感波动性 | 1维 | 情感分数标准差 |
| PageRank | 1维 | 网络中心性指标 |
| 入度 | 1维 | 网络拓扑特征 |

## ⚡ 性能优化

### 内存优化
- 批处理训练减少显存占用
- 定期清理GPU缓存
- 梯度裁剪防止梯度爆炸

### 训练优化
- 学习率调度策略
- 早停机制避免过拟合
- 渐进式难度训练

### 推理优化
```python
# 模型推理示例
model.eval()
with torch.no_grad():
    embeddings = model.encode(features, edge_index)
    predictions = model.decode(embeddings, test_edges)
```


### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据维度
print(f"Features shape: {features.shape}")
print(f"Edge index shape: {edge_index.shape}")

# 验证配置
from config import validate_config
errors = validate_config(config)
if errors:
    print("Configuration errors:", errors)
```



## 🙏 致谢

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络框架
- [Transformers](https://huggingface.co/transformers/) - 预训练模型库
- [NetworkX](https://networkx.org/) - 图分析工具库