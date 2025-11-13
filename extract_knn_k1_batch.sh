#!/bin/bash

# Batch script to extract K=1 from K=10 KNN files for multiple datasets
# This will process all datasets that have K=10 KNN files

# Configuration
DATA_PATH="/data/vector_datasets"
PYTHON_SCRIPT="./extract_knn_k1.py"

# List of datasets to process (empty means auto-detect all)
DATASETS=(
    "sift10m"
)

echo "======================================"
echo "K=1 KNN Extraction Batch Script"
echo "======================================"
echo "Data path: $DATA_PATH"
echo ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "./venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ./venv/bin/activate
    echo ""
fi

# Process each dataset
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for dataset in "${DATASETS[@]}"
do
    echo "======================================"
    echo "Processing: $dataset"
    echo "======================================"
    
    # Check if dataset directory exists
    if [ ! -d "$DATA_PATH/$dataset" ]; then
        echo "⊘ Dataset directory not found: $DATA_PATH/$dataset"
        echo "  Skipping..."
        echo ""
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    
    # Check if K=10 KNN file exists
    KNN_CACHE="$DATA_PATH/$dataset/knn_cache"
    if [ ! -d "$KNN_CACHE" ]; then
        echo "⊘ KNN cache directory not found: $KNN_CACHE"
        echo "  Skipping..."
        echo ""
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    
    KNN10_FILES=$(find "$KNN_CACHE" -name "*knn10*.bin" 2>/dev/null)
    if [ -z "$KNN10_FILES" ]; then
        echo "⊘ No K=10 KNN file found"
        echo "  Skipping..."
        echo ""
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    
    # Run extraction
    python3 "$PYTHON_SCRIPT" "$dataset" "$DATA_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "✗ $dataset failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

# Summary
echo "======================================"
echo "Batch Processing Summary"
echo "======================================"
echo "Successful: $SUCCESS_COUNT"
echo "Failed:     $FAIL_COUNT"
echo "Skipped:    $SKIP_COUNT"
echo "Total:      $((SUCCESS_COUNT + FAIL_COUNT + SKIP_COUNT))"
echo "======================================"

if [ $FAIL_COUNT -eq 0 ] && [ $SUCCESS_COUNT -gt 0 ]; then
    echo "All datasets processed successfully!"
    exit 0
else
    exit 1
fi

