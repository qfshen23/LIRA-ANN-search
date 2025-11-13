#!/bin/bash

# 单次测试脚本 - 测试单个配置
# 使用方法: ./test_single_run.sh

echo "========================================"
echo "   测试单个配置运行"
echo "========================================"
echo ""

# 配置
DATASET="tiny5m"
N_BKT="256"
K="10"

echo "配置信息:"
echo "  Dataset: ${DATASET}"
echo "  n_bkt: ${N_BKT}"
echo "  k: ${K}"
echo ""

echo "执行命令:"
echo "python3 LIRA_smallscale.py --dataset \"${DATASET}\" --n_bkt \"${N_BKT}\" --k \"${K}\""
echo ""
echo "----------------------------------------"
echo ""

# 执行
python3 LIRA_smallscale.py --dataset "${DATASET}" --n_bkt "${N_BKT}" --k "${K}"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 执行成功!"
else
    echo ""
    echo "✗ 执行失败!"
    echo ""
    echo "提示: 如果是数据文件不存在的错误，这是正常的"
    echo "      如果是参数错误，请检查参数格式"
fi

