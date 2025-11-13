#!/bin/bash

# Script to precompute KNN for multiple datasets
# This will generate cache files that Python scripts can directly load

# Configuration
DATA_PATH="/data/vector_datasets"
K=10
N_THREADS=32
NPROBE=64  # Number of clusters to probe (0 for exact search, 64-128 recommended for speed)

# List of datasets to process
DATASETS=(
    "sift10m"
)

echo "======================================"
echo "KNN Precomputation Script"
echo "======================================"
echo "Data path: $DATA_PATH"
echo "K: $K"
echo "Threads: $N_THREADS"
echo "Nprobe: $NPROBE $([ $NPROBE -eq 0 ] && echo '(Exact search)' || echo '(Approximate IVF)')"
echo ""

# Check if compute_knn exists
if [ ! -f "./compute_knn" ]; then
    echo "Error: compute_knn executable not found!"
    echo "Please run: bash build_knn.sh"
    exit 1
fi

# Process each dataset
for dataset in "${DATASETS[@]}"
do
    echo "======================================"
    echo "Processing dataset: $dataset"
    echo "======================================"
    
    ./compute_knn "$dataset" "$DATA_PATH" "$K" "$NPROBE" "$N_THREADS"
    
    if [ $? -eq 0 ]; then
        echo "✓ $dataset completed successfully"
    else
        echo "✗ $dataset failed"
    fi
    echo ""
done

echo "======================================"
echo "All datasets processed!"
echo "======================================"

