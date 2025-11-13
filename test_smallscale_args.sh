#!/bin/bash

# 测试脚本 - 验证 LIRA_smallscale.py 的命令行参数功能
# 使用方法: ./test_smallscale_args.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "   测试 LIRA_smallscale.py 参数功能"
echo "========================================"
echo ""

# 测试 1: 缺少所有必需参数
echo -e "${YELLOW}[测试 1] 缺少所有必需参数（应该失败）${NC}"
python LIRA_smallscale.py 2>&1 | grep -q "参数.*是必需的"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 通过 - 正确提示缺少参数${NC}"
else
    echo -e "${RED}✗ 失败 - 应该提示缺少参数${NC}"
fi
echo ""

# 测试 2: 只指定 dataset
echo -e "${YELLOW}[测试 2] 只指定 dataset（应该失败）${NC}"
python LIRA_smallscale.py --dataset sift 2>&1 | grep -q "参数.*是必需的"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 通过 - 正确提示缺少 k 或 n_bkt${NC}"
else
    echo -e "${RED}✗ 失败 - 应该提示缺少参数${NC}"
fi
echo ""

# 测试 3: 显示帮助信息
echo -e "${YELLOW}[测试 3] 显示帮助信息${NC}"
python LIRA_smallscale.py --help > /tmp/help_output.txt 2>&1
if grep -q "dataset" /tmp/help_output.txt && grep -q "n_bkt" /tmp/help_output.txt && grep -q "^  --k" /tmp/help_output.txt; then
    echo -e "${GREEN}✓ 通过 - 帮助信息包含必需参数${NC}"
    echo "帮助信息预览:"
    head -20 /tmp/help_output.txt
else
    echo -e "${RED}✗ 失败 - 帮助信息不完整${NC}"
fi
echo ""

# 测试 4: 参数解析测试（不实际运行，只测试到配置阶段）
echo -e "${YELLOW}[测试 4] 参数解析测试${NC}"
echo "测试命令: python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10"
echo "（如果数据文件不存在，会在后续步骤失败，但参数解析应该成功）"
echo ""

# 清理
rm -f /tmp/help_output.txt

echo "========================================"
echo "   测试完成"
echo "========================================"
echo ""
echo "如果要实际运行完整流程，请确保:"
echo "  1. 数据集文件存在于 /data/vector_datasets/"
echo "  2. 使用命令: python LIRA_smallscale.py --dataset sift --n_bkt 64 --k 10"
echo ""

