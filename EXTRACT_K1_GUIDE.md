# K=1 KNN Extraction Guide

This guide explains how to extract K=1 KNN data from precomputed K=10 KNN files.

## Overview

Instead of recomputing K=1 KNN from scratch, you can directly extract the first nearest neighbor from existing K=10 KNN files. This is much faster and produces identical results.

## Quick Start

### Extract K=1 for a Single Dataset

```bash
# Basic usage
python3 extract_knn_k1.py <dataset_name>

# With custom data path
python3 extract_knn_k1.py <dataset_name> /path/to/data

# Examples
python3 extract_knn_k1.py sift
python3 extract_knn_k1.py bigann10m /data/vector_datasets
```

### Batch Processing for All Datasets

```bash
# Process all datasets with K=10 KNN files
bash extract_knn_k1_batch.sh
```

## How It Works

### Input
- Reads K=10 KNN file: `{dataset}/knn_cache/{dataset}-data_self_knn10-n{n}_ivf_nprobe{nprobe}.bin`
- Format: Binary file with int32 values, shape (n, 10)

### Processing
- Extracts only the first column (first nearest neighbor)
- Creates K=1 array of shape (n, 1)

### Output
- Saves K=1 KNN file: `{dataset}/knn_cache/{dataset}-data_self_knn1-n{n}_ivf_nprobe{nprobe}.bin`
- Creates metadata file: `{same_name}.meta`

## File Structure

### Before
```
/data/vector_datasets/sift/knn_cache/
├── sift-data_self_knn10-n1000000_ivf_nprobe64.bin
└── sift-data_self_knn10-n1000000_ivf_nprobe64.bin.meta
```

### After
```
/data/vector_datasets/sift/knn_cache/
├── sift-data_self_knn10-n1000000_ivf_nprobe64.bin
├── sift-data_self_knn10-n1000000_ivf_nprobe64.bin.meta
├── sift-data_self_knn1-n1000000_ivf_nprobe64.bin          # NEW
└── sift-data_self_knn1-n1000000_ivf_nprobe64.bin.meta     # NEW
```

## Usage in Python Code

### Loading K=1 KNN Data

```python
import numpy as np

# Example for SIFT dataset
dataset_name = "sift"
data_path = "/data/vector_datasets"
knn_file = f"{data_path}/{dataset_name}/knn_cache/{dataset_name}-data_self_knn1-n1000000_ivf_nprobe64.bin"

# Load K=1 KNN data
knn_k1 = np.fromfile(knn_file, dtype=np.int32).reshape(-1, 1)

# Or load and flatten to 1D array
knn_k1 = np.fromfile(knn_file, dtype=np.int32)

print(f"K=1 KNN shape: {knn_k1.shape}")
print(f"First 10 nearest neighbors: {knn_k1[:10]}")
```

### Comparing K=1 and K=10

```python
import numpy as np

# Load K=10
knn_k10 = np.fromfile("...-knn10-...bin", dtype=np.int32).reshape(-1, 10)

# Load K=1
knn_k1 = np.fromfile("...-knn1-...bin", dtype=np.int32).reshape(-1, 1)

# They should be identical
assert np.all(knn_k10[:, 0:1] == knn_k1), "K=1 should match first column of K=10"
print("✓ K=1 matches K=10[:, 0] perfectly!")
```

## Benefits

### Time Savings
- **Recomputation**: 10-60 minutes per dataset (depends on size)
- **Extraction**: < 1 second per dataset
- **Speedup**: 600-3600x faster!

### Space Efficiency
- K=10 file size: ~40 MB (for 1M vectors)
- K=1 file size: ~4 MB (for 1M vectors)
- Save 90% storage for K=1 tasks

### Accuracy
- **Identical results**: K=1 is exactly the first neighbor from K=10
- **No approximation error**: Direct extraction, not recomputation
- **Same nprobe**: Inherits the nprobe setting from K=10

## Example Output

```bash
$ python3 extract_knn_k1.py sift

Dataset: sift
Data path: /data/vector_datasets

Found K=10 KNN file: sift-data_self_knn10-n1000000_ivf_nprobe64.bin
============================================================
Extracting K=1 from K=10 KNN file
============================================================
Input file:  .../sift-data_self_knn10-n1000000_ivf_nprobe64.bin
Output file: .../sift-data_self_knn1-n1000000_ivf_nprobe64.bin

Metadata found:
  Number of vectors: 1000000
  Original K: 10
  Method: ivf_approximate
  Nprobe: 64

Reading K=10 KNN data...
  Shape: (1000000, 10)
  Data type: int32
  Memory usage: 38.15 MB

Extracting K=1 (first nearest neighbor)...
  K=1 shape: (1000000, 1)
  K=1 memory usage: 3.81 MB

Statistics:
  Min neighbor ID: 0
  Max neighbor ID: 999999
  Sample neighbors (first 10 vectors):
    Vector 0 -> Nearest neighbor: 12543
    Vector 1 -> Nearest neighbor: 87234
    ...

Saving K=1 KNN data...
  Saved to: .../sift-data_self_knn1-n1000000_ivf_nprobe64.bin
  Saved metadata to: .../sift-data_self_knn1-n1000000_ivf_nprobe64.bin.meta

============================================================
SUCCESS! K=1 extraction completed
============================================================

To load in Python:
  knn_k1 = np.fromfile('...', dtype=np.int32).reshape(1000000, 1)
```

## Batch Processing Example

```bash
$ bash extract_knn_k1_batch.sh

======================================
K=1 KNN Extraction Batch Script
======================================
Data path: /data/vector_datasets

Activating virtual environment...

======================================
Processing: sift
======================================
✓ sift completed successfully

======================================
Processing: gist
======================================
✓ gist completed successfully

======================================
Processing: bigann10m
======================================
✓ bigann10m completed successfully

======================================
Batch Processing Summary
======================================
Successful: 3
Failed:     0
Skipped:    4
Total:      7
======================================
All datasets processed successfully!
```

## Troubleshooting

### "No K=10 KNN file found"
- Make sure you've run `bash precompute_knn.sh` first
- Check that K=10 KNN files exist in `{dataset}/knn_cache/`

### "Cannot open file"
- Verify file permissions
- Check that the data path is correct
- Ensure sufficient disk space for output files

### "Shape mismatch"
- The script auto-detects the correct shape
- If issues persist, check the .meta file for correct parameters

## Integration with LIRA

The extracted K=1 files work seamlessly with LIRA:

```python
# In your LIRA script
import numpy as np

# Load K=1 KNN
knn_cache_file = f"{data_path}/{dataset_name}/knn_cache/{dataset_name}-data_self_knn1-n{n}_ivf_nprobe64.bin"
if os.path.exists(knn_cache_file):
    data_self_knn = np.fromfile(knn_cache_file, dtype=np.int32).reshape(n, 1)
    print(f"Loaded K=1 KNN from cache: {knn_cache_file}")
else:
    # Fallback: compute K=1 from scratch
    data_self_knn = compute_knn(data, k=1)
```

## Summary

✅ **Fast**: Extract in seconds, not minutes
✅ **Accurate**: Identical to recomputation
✅ **Easy**: One command per dataset
✅ **Efficient**: 10x smaller files for K=1
✅ **Compatible**: Works with existing LIRA code

For any issues, check the metadata files (`.meta`) for detailed information about the KNN computation parameters.

