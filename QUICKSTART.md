# LIRA_smallscale.py 快速开始

## 🚀 立即开始

### 方式一：单次执行（推荐用于测试）

```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10
```

### 方式二：批量执行（推荐用于实验）

```bash
# 添加执行权限（首次需要）
chmod +x run_smallscale_simple.sh

# 运行批量脚本
./run_smallscale_simple.sh
```

## 📋 必需参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--dataset` | 数据集名称 | `sift`, `tiny5m`, `bigann10m` |
| `--n_bkt` | bucket数量 | `64`, `128`, `256` |
| `--k` | 最近邻数量 | `10`, `100` |

## 🎛️ 常用可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--redundancy_ratio` | `0.03` | 冗余比例 (3%) |
| `--duplicate_type` | `model` | 冗余类型 (`model` 或 `None`) |
| `--n_epoch` | `10` | 训练轮数 |

## 📝 常用命令示例

### 基础测试
```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10
```

### 更改冗余比例
```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10 --redundancy_ratio 0.05
```

### 不使用冗余
```bash
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10 --duplicate_type None
```

### 大数据集测试
```bash
python LIRA_smallscale.py --dataset bigann10m --n_bkt 256 --k 10
```

### 更多最近邻
```bash
python LIRA_smallscale.py --dataset sift --n_bkt 128 --k 100
```

## 📊 结果位置

- 日志文件: `./logs/{dataset}/ML_kmeans_RE_FLAT/{config}.txt`
- CSV结果: `./logs/{dataset}/ML_kmeans_RE_FLAT/{config}.csv`
- 调优结果: `./logs/{dataset}/ML_kmeans_RE_FLAT/{config}_tuning_threshold/`

## 🔧 批量脚本配置

编辑 `run_smallscale_simple.sh` 中的配置：

```bash
configs=(
    "sift 64 10"
    "sift 128 10"
    "your_dataset your_n_bkt your_k"  # 添加你的配置
)
```

## ❓ 常见问题

**Q: 如何查看所有参数？**
```bash
python LIRA_smallscale.py --help
```

**Q: 忘记指定参数会怎样？**

会收到清晰的错误提示：
```
ValueError: 参数 --dataset 是必需的！例如: --dataset sift
```

**Q: 如何后台运行批量任务？**
```bash
nohup ./run_smallscale_simple.sh > batch.log 2>&1 &
```

## 📚 详细文档

- 完整指南: [`RUN_SMALLSCALE_GUIDE.md`](RUN_SMALLSCALE_GUIDE.md)
- 修改说明: [`CHANGES_SUMMARY.md`](CHANGES_SUMMARY.md)

## ⚡ 快速测试流程

```bash
# 1. 测试参数功能
chmod +x test_smallscale_args.sh
./test_smallscale_args.sh

# 2. 单次测试
python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10

# 3. 批量测试
chmod +x run_smallscale_simple.sh
./run_smallscale_simple.sh
```

---

💡 **提示**: 建议先在小数据集（如 `sift`）上测试配置，再在大数据集上运行！

