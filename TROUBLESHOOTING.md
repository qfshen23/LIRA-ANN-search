# 故障排除指南

## 问题：`argument --n_bkt/--n-bkt: expected one argument`

### 错误信息
```
LIRA_smallscale.py: error: argument --n_bkt/--n-bkt: expected one argument
```

### 原因分析

这个错误通常由以下原因引起：

1. **变量未正确传递** - bash 变量没有被引号包围
2. **变量为空** - 配置解析失败导致变量为空
3. **命令行格式问题** - 参数之间有额外的空格或换行

### 解决方案

#### 方案 1：直接命令行测试（推荐用于排查）

首先，直接运行命令测试：

```bash
python3 LIRA_smallscale.py --dataset tiny5m --n_bkt 256 --k 10
```

如果这个命令能正常工作（可能会因为数据文件不存在而失败，但不应该有参数错误），说明问题在脚本中。

#### 方案 2：使用测试脚本

```bash
# 添加执行权限
chmod +x test_single_run.sh

# 运行测试
./test_single_run.sh
```

#### 方案 3：使用调试脚本

```bash
# 添加执行权限
chmod +x debug_args.sh

# 运行调试
./debug_args.sh
```

这会显示参数解析的详细信息，帮助你找到问题所在。

#### 方案 4：修改后的批量脚本

我已经更新了 `run_smallscale_simple.sh`，在所有变量周围添加了引号：

```bash
python3 LIRA_smallscale.py \
    --dataset "${dataset}" \
    --n_bkt "${n_bkt}" \
    --k "${k}"
```

现在应该可以正常工作了。

### 手动检查步骤

#### 步骤 1：检查脚本中的配置

编辑 `run_smallscale_simple.sh`，确认配置格式正确：

```bash
configs=(
    "tiny5m 256 10"
    "deep10m 256 10"
    "spacev10m 256 10"
    "bigann10m 256 10"
)
```

每行应该是 `"dataset n_bkt k"` 格式，用空格分隔。

#### 步骤 2：检查 Python 命令

在脚本中，确认命令格式：

```bash
python3 LIRA_smallscale.py \
    --dataset "${dataset}" \
    --n_bkt "${n_bkt}" \
    --k "${k}"
```

注意：
- 变量要用 `"${variable}"` 包围
- 反斜杠 `\` 后面不能有空格
- 参数名和参数值之间用空格分隔

#### 步骤 3：手动运行单个配置

从脚本中复制一个配置，手动解析并运行：

```bash
# 设置变量
config="tiny5m 256 10"
read -r dataset n_bkt k <<< "$config"

# 显示变量值（调试用）
echo "dataset: [$dataset]"
echo "n_bkt: [$n_bkt]"
echo "k: [$k]"

# 运行命令
python3 LIRA_smallscale.py --dataset "${dataset}" --n_bkt "${n_bkt}" --k "${k}"
```

### 常见错误模式

#### 错误 1：变量未加引号

❌ **错误写法：**
```bash
python3 LIRA_smallscale.py --dataset ${dataset} --n_bkt ${n_bkt} --k ${k}
```

✅ **正确写法：**
```bash
python3 LIRA_smallscale.py --dataset "${dataset}" --n_bkt "${n_bkt}" --k "${k}"
```

#### 错误 2：反斜杠后有空格

❌ **错误写法：**
```bash
python3 LIRA_smallscale.py \ 
    --dataset "${dataset}" \
    --n_bkt "${n_bkt}" \
    --k "${k}"
```

✅ **正确写法：**
```bash
python3 LIRA_smallscale.py \
    --dataset "${dataset}" \
    --n_bkt "${n_bkt}" \
    --k "${k}"
```

#### 错误 3：配置字符串格式错误

❌ **错误写法：**
```bash
configs=(
    "tiny5m  256  10"  # 多个空格
    "deep10m,256,10"   # 使用逗号
)
```

✅ **正确写法：**
```bash
configs=(
    "tiny5m 256 10"    # 单个空格
    "deep10m 256 10"
)
```

### 快速测试命令集

```bash
# 1. 测试 Python 脚本是否存在
ls -l LIRA_smallscale.py

# 2. 测试参数帮助
python3 LIRA_smallscale.py --help

# 3. 测试单个配置（直接命令）
python3 LIRA_smallscale.py --dataset tiny5m --n_bkt 256 --k 10

# 4. 测试变量解析
config="tiny5m 256 10"
read -r dataset n_bkt k <<< "$config"
echo "dataset=[$dataset] n_bkt=[$n_bkt] k=[$k]"

# 5. 测试完整命令（带变量）
python3 LIRA_smallscale.py --dataset "${dataset}" --n_bkt "${n_bkt}" --k "${k}"
```

### 如果问题仍然存在

如果以上方法都无法解决问题，请尝试：

1. **查看完整错误信息**：
```bash
python3 LIRA_smallscale.py --dataset tiny5m --n_bkt 256 --k 10 2>&1 | tee error.log
```

2. **检查 Python 版本**：
```bash
python3 --version
```

3. **检查是否有隐藏字符**：
```bash
cat -A run_smallscale_simple.sh | grep "python3"
```

4. **重新创建脚本**：
如果怀疑脚本文件损坏，可以删除并重新创建：
```bash
rm run_smallscale_simple.sh
# 然后从文档重新创建
```

### 立即可用的解决方案

如果你急需运行实验，可以先用最简单的方式：

**创建一个新的简单脚本 `run_one.sh`：**

```bash
#!/bin/bash
python3 LIRA_smallscale.py --dataset tiny5m --n_bkt 256 --k 10
python3 LIRA_smallscale.py --dataset deep10m --n_bkt 256 --k 10
python3 LIRA_smallscale.py --dataset spacev10m --n_bkt 256 --k 10
python3 LIRA_smallscale.py --dataset bigann10m --n_bkt 256 --k 10
```

然后运行：
```bash
chmod +x run_one.sh
./run_one.sh
```

这样可以绕过变量解析的问题，直接使用硬编码的参数。

## 其他常见问题

### 数据文件找不到

```
ValueError: Ground truth file not found for dataset tiny5m
```

**解决方案**：确保数据文件在正确的位置：
```bash
ls -l /data/vector_datasets/tiny5m*
```

### GPU 内存不足

**解决方案**：在脚本中添加 GPU 选择或使用 CPU：
```bash
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
# 或
export CUDA_VISIBLE_DEVICES=""  # 使用CPU
```

### 权限问题

```
bash: ./run_smallscale_simple.sh: Permission denied
```

**解决方案**：
```bash
chmod +x run_smallscale_simple.sh
```

## 需要帮助？

如果以上方法都无法解决问题，请提供：

1. 完整的错误信息
2. `run_smallscale_simple.sh` 的内容
3. 直接运行命令的结果：
   ```bash
   python3 LIRA_smallscale.py --dataset tiny5m --n_bkt 256 --k 10
   ```

