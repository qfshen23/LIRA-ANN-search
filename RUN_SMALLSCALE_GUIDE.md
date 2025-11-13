# LIRA Small-Scale 使用指南

## 概述

`LIRA_smallscale.py` 已经升级，现在支持通过命令行参数指定数据集、bucket数量和k值。

## 必需参数

执行脚本时，必须指定以下三个参数：

- `--dataset`: 数据集名称（例如：`sift`, `tiny5m`, `bigann10m`）
- `--n_bkt`: bucket/cluster 数量（例如：`64`, `128`, `256`）
- `--k`: 最近邻数量（例如：`10`, `100`）

## 可选参数

- `--redundancy_ratio`: 冗余比例（默认：`0.03`，即3%）
- `--duplicate_type`: 冗余类型（默认：`model`，可选：`None`, `model`）
- `--n_epoch`: 训练轮数（默认：`10`）
- `--batch_size`: 批次大小（默认：`512`）
- `--data_path`: 数据集路径（默认：`/data/vector_datasets`）

## 使用方法

### 1. 单次执行

运行单个配置的基本命令：

```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10
```

带可选参数的示例：

```bash
python LIRA_smallscale.py \
    --dataset sift \
    --n_bkt 128 \
    --k 10 \
    --redundancy_ratio 0.05 \
    --duplicate_type model \
    --n_epoch 15
```

### 2. 批量执行 - 简化版（推荐）

使用预定义的配置批量执行：

```bash
# 首先添加执行权限
chmod +x run_smallscale_simple.sh

# 运行脚本
./run_smallscale_simple.sh
```

这个脚本包含以下预定义配置：
- sift, n_bkt=64, k=10
- sift, n_bkt=128, k=10
- sift, n_bkt=256, k=10
- tiny5m, n_bkt=64, k=10
- tiny5m, n_bkt=128, k=10
- bigann10m, n_bkt=256, k=10

### 3. 批量执行 - 完整版

对多个参数组合进行全量测试：

```bash
# 首先添加执行权限
chmod +x run_smallscale.sh

# 运行脚本
./run_smallscale.sh
```

这个脚本会遍历：
- 3个数据集: sift, tiny5m, bigann10m
- 3个n_bkt值: 64, 128, 256
- 2个k值: 10, 100
- 3个redundancy_ratio: 0.01, 0.03, 0.05
- 2个duplicate_type: model, None

总共 3×3×2×3×2 = 108 个配置组合

**警告**: 完整版执行时间很长！建议先修改脚本中的配置列表再运行。

### 4. 自定义批量脚本

你可以根据需要修改 `run_smallscale_simple.sh`，编辑配置数组：

```bash
configs=(
    "sift 64 10"
    "sift 128 10"
    "your_dataset your_n_bkt your_k"
)
```

## 输出

执行结果会保存在以下位置：

- **日志文件**: `./logs/{dataset}/ML_kmeans_RE_FLAT/{config}.txt`
- **CSV结果**: `./logs/{dataset}/ML_kmeans_RE_FLAT/{config}.csv`
- **调优结果**: `./logs/{dataset}/ML_kmeans_RE_FLAT/{config}_tuning_threshold/`

## 示例

### 示例 1: 在 SIFT 数据集上测试不同的 bucket 数量

```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10
python LIRA_smallscale.py --dataset sift --n_bkt 128 --k 10
python LIRA_smallscale.py --dataset sift --n_bkt 256 --k 10
```

### 示例 2: 比较不同的冗余比例

```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10 --redundancy_ratio 0.01
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10 --redundancy_ratio 0.03
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10 --redundancy_ratio 0.05
```

### 示例 3: 测试不同的k值

```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 50
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 100
```

## 错误处理

如果忘记指定必需参数，脚本会提示错误：

```
ValueError: 参数 --dataset 是必需的！例如: --dataset sift
ValueError: 参数 --k 是必需的！例如: --k 10
ValueError: 参数 --n_bkt 是必需的！例如: --n_bkt 64
```

## 注意事项

1. 确保数据集路径正确（默认为 `/data/vector_datasets`）
2. 确保有足够的磁盘空间存储日志和结果
3. 对于大数据集（如 bigann10m），执行时间可能较长
4. 建议先在小数据集（如 sift）上测试配置

## 支持的数据集

- `sift`: SIFT 1M
- `tiny5m`: Tiny 5M
- `bigann10m`: BigANN 10M
- 其他数据集需要确保在 `/data/vector_datasets` 目录下有相应文件

## 常见问题

**Q: 如何只运行特定的数据集？**

A: 直接使用命令行参数：
```bash
python LIRA_smallscale.py --dataset your_dataset --n_bkt 64 --k 10
```

**Q: 如何修改批量脚本的配置？**

A: 编辑 `run_smallscale_simple.sh`，修改 `configs` 数组。

**Q: 执行时间太长怎么办？**

A: 可以减少批量脚本中的配置数量，或者使用 `nohup` 在后台运行：
```bash
nohup ./run_smallscale_simple.sh > batch_run.log 2>&1 &
```

