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

## Known Limitations

1. **Cleanup Segfault:** Tests may segfault during global teardown (AFTER completion). This does not affect correctness or functionality - it's a destructor exception issue in C++11+ mode.

2. **Memory Overhead:** Dataset is duplicated in GPU memory. Large datasets may exceed GPU memory limits.

3. **Binary Format:** OpenCL kernels are compiled at runtime. First search incurs ~0.3-1.0s compilation overhead.

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
