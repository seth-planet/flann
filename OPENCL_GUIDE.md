# FLANN OpenCL Guide

This guide covers building, using, and troubleshooting FLANN with OpenCL GPU acceleration.

## Overview

This fork includes OpenCL-accelerated implementations of nearest neighbor search algorithms for GPU acceleration. The OpenCL code was developed in 2017-2018 and has been modernized for OpenCL 3.0 compatibility.

**Supported Algorithms:**
- **K-Means Index** (`FLANN_INDEX_KMEANS_OPENCL`) - Hierarchical k-means with GPU acceleration
- **Hierarchical Clustering** (`FLANN_INDEX_HIERARCHICAL_OPENCL`) - GPU-accelerated clustering

**Performance Characteristics:**
- Best for large datasets (>50K points) and high dimensions (>64D)
- Typical speedup: 5-20x vs multi-core CPU for large datasets
- Overhead: ~0.3-1.0s kernel compilation + data transfer
- Precision: 92-98% recall (comparable to CPU versions)

## Building with OpenCL

```bash
# Install OpenCL development headers (Ubuntu/Debian)
sudo apt-get install ocl-icd-opencl-dev opencl-headers clinfo

# Verify OpenCL devices
clinfo  # Should show your GPU

# Build with OpenCL support
mkdir build && cd build
cmake -DBUILD_OpenCL_LIB=ON -DBUILD_TESTS=ON ..
make -j$(nproc)
```

## System Requirements

**Hardware:**
- OpenCL 1.2+ compatible device (GPU, CPU, or accelerator)
- Minimum 2GB device memory (4GB+ recommended)
- Tested platforms:
  - NVIDIA GPUs (Tesla T4, RTX series) with CUDA OpenCL backend
  - AMD GPUs with ROCm OpenCL
  - Intel integrated/discrete GPUs

**Software:**
```bash
# Ubuntu/Debian
sudo apt-get install ocl-icd-opencl-dev opencl-headers

# Fedora/RHEL
sudo dnf install ocl-icd-devel opencl-headers

# Verify installation
clinfo | grep "Platform Name"
```

## Usage Example

```cpp
#include <flann/flann.hpp>

// Load your dataset
flann::Matrix<float> dataset = ...;  // Your data (N x D)

// Create OpenCL-accelerated K-Means index
flann::Index<flann::L2<float>> index(
    dataset,
    flann::KMeansOpenCLIndexParams(
        32,    // branching factor
        11,    // iterations
        flann::FLANN_CENTERS_RANDOM,
        0.2    // cb_index
    )
);

index.buildIndex();  // Builds index structure

// Perform search (automatically uses GPU)
flann::Matrix<float> queries = ...;      // Query vectors
flann::Matrix<int> indices(queries.rows, 10);    // k=10
flann::Matrix<float> distances(queries.rows, 10);

index.knnSearch(queries, indices, distances, 10,
                flann::SearchParams(128));  // checks parameter
```

## When to Use OpenCL

**Good use cases:**
- Large datasets (>50,000 points)
- High dimensions (>64D)
- Batch queries (>100 queries)
- Multi-vendor GPU support needed (AMD, Intel, NVIDIA)

**When NOT to use:**
- Small datasets (<10,000 points) - overhead dominates
- Few queries (<10) - transfer cost not amortized
- No GPU available or insufficient memory

**Typical Performance (SIFT128D, Tesla T4):**
- 10K points: ~0.15s build, ~0.01s search (1000 queries)
- 100K points: ~0.6s build, ~0.01s search (1000 queries)
- Speedup vs CPU: 5-15x for search, 2-5x for build

## Device Information

The library automatically logs device capabilities at runtime:
```
[FLANN OpenCL] Device: Tesla T4
[FLANN OpenCL] OpenCL Version: OpenCL 3.0 CUDA
[FLANN OpenCL] Max Work Group Size: 1024
[FLANN OpenCL] Max Compute Units: 40
```

## Troubleshooting

### Build Issues

**Error: "CL/opencl.h not found"**
```bash
sudo apt-get install ocl-icd-opencl-dev opencl-headers
```

**Error: "No OpenCL platforms found"**
- Verify drivers: `clinfo`
- Install GPU drivers (NVIDIA, AMD, or Intel)
- For CPU fallback: `sudo apt-get install pocl-opencl-icd`

### Runtime Issues

**OpenCL error at initialization:**
- Check GPU memory: `nvidia-smi` (NVIDIA) or `radeontop` (AMD)
- Verify device is accessible: `clinfo`
- Try CPU OpenCL device if GPU fails

**Poor performance:**
- Ensure using GPU device (check logged device name)
- Small datasets have overhead that dominates - use CPU version
- Profile GPU utilization: `nvidia-smi -l 1` during search

**Kernel compilation errors:**
- Check OpenCL version compatibility (code targets 1.2+)
- Review kernel build logs in error messages
- Some older GPUs may have work group size limitations

## OpenCL vs CUDA Comparison (December 2024)

Both OpenCL and CUDA backends provide GPU acceleration. This section compares their performance using the same CPU index files with proper warmup queries.

**Methodology:** All benchmarks use `scripts/benchmark_comprehensive.cpp` with:
- 2 warmup runs (discarded from timing)
- 5 timed runs (statistics reported)
- Same CPU index construction for both backends
- Ground truth computed via brute-force linear search

```bash
# Run OpenCL vs CUDA comparison
./test/benchmark_comprehensive datasets/brief100K.h5 --gpu-only --warmup=2 --runs=5
./test/benchmark_comprehensive datasets/sift100K.h5 --kmeans --gpu-only --warmup=2 --runs=5
```

### Binary Hierarchical (brief100K - 100K × 32 bytes, Hamming)

| Metric | OpenCL | CUDA | Notes |
|--------|--------|------|-------|
| Build time (CPU) | 0.303s ± 0.008 | 0.305s ± 0.002 | Equal |
| GPU Upload | 0.357s | 0.014s | CUDA 25x faster |
| Search (warm) | 0.005s | 0.005s | Equal |
| Precision | 81.7% | 89.7% | CUDA +8% |
| Throughput | 208,928 q/s | 182,498 q/s | OpenCL slightly higher |

### Float K-Means (sift100K - 99K × 128 floats, L2)

| Metric | OpenCL | CUDA | Notes |
|--------|--------|------|-------|
| Build time (CPU) | 10.74s ± 0.16 | 10.83s ± 0.11 | Equal |
| GPU Upload | 0.493s | 0.152s | CUDA 3.2x faster |
| Search (warm) | 0.012s | 0.010s | CUDA 17% faster |
| Precision | 77.2% | 88.4% | CUDA +11% |
| Throughput | 86,238 q/s | 97,194 q/s | CUDA 13% higher |

### When to Choose OpenCL

| Use Case | Recommendation |
|----------|----------------|
| AMD/Intel GPUs | **OpenCL** - only option |
| Multi-vendor deployment | **OpenCL** - portable |
| NVIDIA with max precision | **CUDA** - 8-11% better recall |
| NVIDIA with fast upload | **CUDA** - 3-25x faster upload |
| Cross-platform compatibility | **OpenCL** - runs everywhere |

**Key Insight:** OpenCL provides multi-vendor support with competitive search throughput. CUDA offers faster GPU upload and higher precision on NVIDIA hardware.

---

## Known Limitations

1. **Cleanup Segfault:** Tests may segfault during global teardown (AFTER completion). This does not affect correctness or functionality - it's a destructor exception issue in C++11+ mode.

2. **Memory Overhead:** Dataset is duplicated in GPU memory. Large datasets may exceed GPU memory limits.

3. **Binary Format:** OpenCL kernels are compiled at runtime. First search incurs ~0.3-1.0s compilation overhead.

4. **Precision Gap:** OpenCL achieves 77-82% precision vs CUDA's 88-90% due to differences in tree traversal optimizations.

## Implementation Details

**Files:**
- `src/cpp/flann/algorithms/kmeans_opencl_index.h` - K-Means OpenCL implementation
- `src/cpp/flann/algorithms/hierarchical_opencl_index.h` - Hierarchical clustering
- `src/cpp/flann/algorithms/nn_opencl_index.h` - Base OpenCL infrastructure

**Design:**
- Template-based like CPU versions
- Automatic kernel generation based on distance metric and parameters
- Uses OpenCL 1.2 API with 3.0 compatibility fixes
- Work group sizes automatically tuned to device capabilities

## References

- OpenCL Specification: https://www.khronos.org/opencl/
- FLANN Paper: Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009
- This fork: Based on 2017-2018 OpenCL implementation, modernized 2024
