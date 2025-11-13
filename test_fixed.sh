#!/bin/bash

echo "=========================================="
echo "Testing Fixed IVF Implementation"
echo "=========================================="
echo ""

# Clean up old binary
echo "Step 1: Cleaning up old binary..."
rm -f compute_knn
echo ""

# Compile
echo "Step 2: Compiling fixed code..."
bash build_knn.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Compilation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: Running test with SIFT dataset"
echo "=========================================="
echo ""

# Test with smaller parameters first
DATASET="sift"
DATA_PATH="/data/vector_datasets"
K=10
NPROBE=64
N_THREADS=24

echo "Test configuration:"
echo "  Dataset: $DATASET"
echo "  K: $K"
echo "  nprobe: $NPROBE"
echo "  Threads: $N_THREADS"
echo ""

# Check if data exists
DATA_FILE="$DATA_PATH/$DATASET/${DATASET}_base.fvecs"
if [ ! -f "$DATA_FILE" ]; then
    echo "⚠ Warning: Data file not found: $DATA_FILE"
    echo "Please ensure the dataset is available."
    echo ""
fi

# Run with time measurement
echo "Running: ./compute_knn $DATASET $DATA_PATH $K $NPROBE $N_THREADS"
echo ""
echo "=========================================="
echo ""

time ./compute_knn "$DATASET" "$DATA_PATH" "$K" "$NPROBE" "$N_THREADS"

EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS! No core dump!"
    echo "=========================================="
    echo ""
    echo "Cache files:"
    ls -lh "$DATA_PATH/$DATASET/knn_cache/"*ivf* 2>/dev/null | tail -5
    echo ""
    echo "Latest metadata:"
    LATEST_META=$(ls -t "$DATA_PATH/$DATASET/knn_cache/"*ivf*.meta 2>/dev/null | head -1)
    if [ -f "$LATEST_META" ]; then
        cat "$LATEST_META"
    fi
    echo ""
    echo "=========================================="
    echo "✓ Ready to run LIRA!"
    echo "  python LIRA_smallscale.py --dataset $DATASET"
    echo "=========================================="
else
    echo "✗ FAILED with exit code: $EXIT_CODE"
    echo "=========================================="
    
    # Check for core dump
    if [ -f "core" ] || ls core.* 2>/dev/null; then
        echo ""
        echo "Core dump detected. Debugging info:"
        echo "  gdb ./compute_knn core  # to analyze"
        echo ""
        echo "Common issues:"
        echo "  1. Data file format mismatch"
        echo "  2. Insufficient memory"
        echo "  3. FAISS library version mismatch"
    fi
    
    exit 1
fi

