#!/bin/bash

# ============================================================================
# 批量运行 LIRA_smallscale.py 的脚本
# 用法: bash run_batch_smallscale.sh
# ============================================================================

# 设置环境
source venv/bin/activate

# 定义数据集与对应分区数的配对（格式：dataset:n_bkt）
DATASET_CONFIGS=(
    "gist:64"
    "tiny5m:256"
    "deep10m:256"
    "spacev10m:256"
    "bigann10m:256"
)

# 定义冗余比例列表
REDUNDANCY_RATIOS=(0.03)

# 定义其他固定参数
K=10
N_EPOCH=10
DUPLICATE_TYPE="model"

# 统计总任务数
TOTAL_TASKS=$((${#DATASET_CONFIGS[@]} * ${#REDUNDANCY_RATIOS[@]}))
CURRENT_TASK=0

echo "=========================================="
echo "LIRA Small-Scale 批量实验"
echo "=========================================="
echo "数据集配置:"
for config in "${DATASET_CONFIGS[@]}"; do
    IFS=':' read -r dataset n_bkt <<< "$config"
    echo "  - $dataset: n_bkt=$n_bkt"
done
echo "冗余比例: ${REDUNDANCY_RATIOS[@]}"
echo "总任务数: $TOTAL_TASKS"
echo "=========================================="
echo ""

# 双层循环：遍历每个数据集配置 x 每个冗余比例
for config in "${DATASET_CONFIGS[@]}"; do
    # 分离 dataset 和 n_bkt
    IFS=':' read -r dataset n_bkt <<< "$config"
    
    for redundancy_ratio in "${REDUNDANCY_RATIOS[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        
        echo "----------------------------------------------------------------------"
        echo "[任务 $CURRENT_TASK/$TOTAL_TASKS] 开始"
        echo "  数据集: $dataset"
        echo "  分区数: $n_bkt"
        echo "  冗余比例: $redundancy_ratio"
        echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "----------------------------------------------------------------------"
        
        # 执行 Python 脚本
        python LIRA_smallscale.py \
            --dataset "$dataset" \
            --n_bkt "$n_bkt" \
            --k "$K" \
            --n_epoch "$N_EPOCH" \
            --redundancy_ratio "$redundancy_ratio" \
            --duplicate_type "$DUPLICATE_TYPE"
        
        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "✓ [任务 $CURRENT_TASK/$TOTAL_TASKS] 完成"
        else
            echo "✗ [任务 $CURRENT_TASK/$TOTAL_TASKS] 失败"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "所有任务完成！"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

