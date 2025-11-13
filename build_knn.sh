#!/bin/bash

# Build script for compute_knn.cpp

echo "Building KNN computation program..."

# Check if FAISS is installed
if ! pkg-config --exists faiss 2>/dev/null; then
    echo "Warning: FAISS not found via pkg-config"
    echo "Attempting to compile with default paths..."
fi

# Compilation options
CXX=g++
# Enable aggressive SIMD optimizations
CXXFLAGS="-O3 -std=c++11 -fopenmp -march=native -mtune=native"
CXXFLAGS="$CXXFLAGS -mavx2 -mfma -msse4.2"  # AVX2, FMA, SSE4.2
# Check for AVX-512 support and add if available
if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
    echo "✓ AVX-512 detected, enabling AVX-512 optimizations"
    CXXFLAGS="$CXXFLAGS -mavx512f -mavx512dq -mavx512bw -mavx512vl"
else
    echo "AVX-512 not detected, using AVX2"
fi
CXXFLAGS="$CXXFLAGS -funroll-loops -ffast-math"  # Additional optimizations
INCLUDES="-I/usr/local/include -I/usr/include"
LDFLAGS="-L/usr/local/lib -L/usr/lib"
LIBS="-lfaiss -lopenblas -lgomp -fopenmp"

# Print compilation flags
echo "Compilation flags: $CXXFLAGS"
echo ""

# Try to compile
echo "Compiling compute_knn.cpp with SIMD optimizations..."
$CXX $CXXFLAGS $INCLUDES compute_knn.cpp -o compute_knn $LDFLAGS $LIBS

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful!"
    echo "Executable: ./compute_knn"
    echo ""
    
    # Verify SIMD instructions
    echo "Checking compiled binary for SIMD instructions:"
    objdump -d compute_knn | grep -q "vfmadd" && echo "  ✓ FMA instructions found"
    objdump -d compute_knn | grep -q "vmov" && echo "  ✓ AVX instructions found"
    objdump -d compute_knn | grep -q "vpadd" && echo "  ✓ AVX2 instructions found"
    echo ""
    
    echo "Usage: ./compute_knn <dataset_name> <data_path> <k> [use_approx] [n_threads]"
    echo "Example: ./compute_knn sift /data/vector_datasets 10 0 24"
else
    echo "✗ Compilation failed!"
    echo ""
    echo "If FAISS is installed in a custom location, modify the INCLUDES and LDFLAGS variables."
    echo "For example:"
    echo "  INCLUDES=\"-I/path/to/faiss/include\""
    echo "  LDFLAGS=\"-L/path/to/faiss/lib\""
    exit 1
fi

