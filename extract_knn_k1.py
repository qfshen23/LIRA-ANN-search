#!/usr/bin/env python3
"""
Extract K=1 KNN data from K=10 KNN files

This script reads precomputed K=10 KNN files and extracts only the first
nearest neighbor (K=1) for each vector, saving it to a new file.

Usage:
    python extract_knn_k1.py <dataset_name> [data_path]
    
Example:
    python extract_knn_k1.py sift /data/vector_datasets
    python extract_knn_k1.py bigann10m
"""

import sys
import os
import numpy as np
import glob
from pathlib import Path


def find_knn10_file(dataset_name, data_path="/data/vector_datasets"):
    """
    Find the K=10 KNN file for the given dataset
    """
    dataset_dir = os.path.join(data_path, dataset_name)
    knn_cache_dir = os.path.join(dataset_dir, "knn_cache")
    
    if not os.path.exists(knn_cache_dir):
        print(f"Error: KNN cache directory not found: {knn_cache_dir}")
        return None
    
    # Look for files with knn10 in the name
    pattern = os.path.join(knn_cache_dir, f"{dataset_name}-data_self_knn10-*.bin")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Error: No K=10 KNN file found matching pattern: {pattern}")
        return None
    
    # If multiple files found, prefer the one with highest nprobe or exact search
    if len(files) > 1:
        print(f"Found {len(files)} K=10 KNN files:")
        for f in files:
            print(f"  - {f}")
        # Use the first one
        print(f"Using: {files[0]}")
    
    return files[0]


def read_knn_metadata(knn_file):
    """
    Read metadata from the .meta file if it exists
    """
    meta_file = knn_file + ".meta"
    metadata = {}
    
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    return metadata


def extract_k1_from_k10(input_file, output_file):
    """
    Extract K=1 from K=10 KNN file
    
    Args:
        input_file: Path to K=10 KNN binary file
        output_file: Path to save K=1 KNN binary file
    """
    print("=" * 60)
    print("Extracting K=1 from K=10 KNN file")
    print("=" * 60)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Read metadata to get number of vectors
    metadata = read_knn_metadata(input_file)
    
    if 'n' in metadata:
        n = int(metadata['n'])
        k_original = int(metadata.get('k', 10))
        print(f"Metadata found:")
        print(f"  Number of vectors: {n}")
        print(f"  Original K: {k_original}")
        if 'method' in metadata:
            print(f"  Method: {metadata['method']}")
        if 'nprobe' in metadata:
            print(f"  Nprobe: {metadata['nprobe']}")
        print()
    else:
        # Try to infer from file size
        file_size = os.path.getsize(input_file)
        # Each element is int32 (4 bytes), and we have n * k elements
        total_elements = file_size // 4
        n = total_elements // 10
        k_original = 10
        print(f"Warning: No metadata file found, inferring from file size")
        print(f"  Inferred number of vectors: {n}")
        print(f"  Inferred K: {k_original}")
        print()
    
    # Read the K=10 data
    print("Reading K=10 KNN data...")
    try:
        knn_k10 = np.fromfile(input_file, dtype=np.int32)
        
        # Reshape to (n, k)
        if knn_k10.size != n * k_original:
            # Try to auto-detect k
            k_original = knn_k10.size // n
            print(f"  Auto-detected K: {k_original}")
        
        knn_k10 = knn_k10.reshape(n, k_original)
        print(f"  Shape: {knn_k10.shape}")
        print(f"  Data type: {knn_k10.dtype}")
        print(f"  Memory usage: {knn_k10.nbytes / 1024 / 1024:.2f} MB")
        print()
    except Exception as e:
        print(f"Error reading K=10 file: {e}")
        return False
    
    # Extract first column (K=1)
    print("Extracting K=1 (first nearest neighbor)...")
    knn_k1 = knn_k10[:, 0:1]  # Keep 2D shape (n, 1)
    print(f"  K=1 shape: {knn_k1.shape}")
    print(f"  K=1 memory usage: {knn_k1.nbytes / 1024 / 1024:.2f} MB")
    print()
    
    # Show some statistics
    print("Statistics:")
    print(f"  Min neighbor ID: {knn_k1.min()}")
    print(f"  Max neighbor ID: {knn_k1.max()}")
    print(f"  Sample neighbors (first 10 vectors):")
    for i in range(min(10, n)):
        print(f"    Vector {i} -> Nearest neighbor: {knn_k1[i, 0]}")
    print()
    
    # Save K=1 data
    print("Saving K=1 KNN data...")
    knn_k1.tofile(output_file)
    print(f"  Saved to: {output_file}")
    
    # Save metadata
    meta_output = output_file + ".meta"
    with open(meta_output, 'w') as f:
        if metadata:
            # Copy original metadata
            for key, value in metadata.items():
                if key != 'k':  # Update k value
                    f.write(f"{key}: {value}\n")
        f.write(f"k: 1\n")
        f.write(f"extracted_from: {input_file}\n")
    print(f"  Saved metadata to: {meta_output}")
    print()
    
    print("=" * 60)
    print("SUCCESS! K=1 extraction completed")
    print("=" * 60)
    print()
    print("To load in Python:")
    print(f"  knn_k1 = np.fromfile('{output_file}', dtype=np.int32).reshape({n}, 1)")
    print()
    
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable datasets in /data/vector_datasets:")
        data_path = "/data/vector_datasets"
        if os.path.exists(data_path):
            datasets = [d for d in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, d))]
            for ds in sorted(datasets):
                knn_cache = os.path.join(data_path, ds, "knn_cache")
                if os.path.exists(knn_cache):
                    knn_files = glob.glob(os.path.join(knn_cache, "*knn10*.bin"))
                    if knn_files:
                        print(f"  ✓ {ds} (has K=10 KNN)")
                    else:
                        print(f"  - {ds}")
                else:
                    print(f"  - {ds}")
        print()
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "/data/vector_datasets"
    
    print(f"Dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    print()
    
    # Find K=10 KNN file
    knn10_file = find_knn10_file(dataset_name, data_path)
    if not knn10_file:
        sys.exit(1)
    
    # Generate output filename
    # Replace knn10 with knn1 in the filename
    output_file = knn10_file.replace("-data_self_knn10-", "-data_self_knn1-")
    
    # If the replacement didn't work, try another pattern
    if output_file == knn10_file:
        base, ext = os.path.splitext(knn10_file)
        output_file = base.replace("knn10", "knn1") + ext
    
    # Extract K=1
    success = extract_k1_from_k10(knn10_file, output_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

