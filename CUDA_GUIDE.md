# FLANN CUDA Guide

This guide covers building, using, and optimizing FLANN with CUDA GPU acceleration.

## Table of Contents

1. [Building with CUDA Support](#building-with-cuda-support)
2. [Using CUDA-Accelerated Algorithms](#using-cuda-accelerated-algorithms)
3. [Advantages](#advantages)
4. [Disadvantages and Limitations](#disadvantages-and-limitations)
5. [Gotchas and Common Issues](#gotchas-and-common-issues)
6. [Cross-Compilation for Jetson Orin AGX](#cross-compilation-for-jetson-orin-agx)
7. [Performance Tuning](#performance-tuning)

---

## Building with CUDA Support

### Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| CUDA Toolkit | 11.0+ | Install from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads) |
| CMake | 3.18+ | Required for modern CUDA language support |
| NVIDIA Driver | 450.80.02+ | Check with `nvidia-smi` |
| GPU Compute Capability | 6.0+ | Pascal architecture or newer |

**Supported GPU Architectures:**

| Architecture | GPU Family | Example Devices |
|--------------|------------|-----------------|
| 75 | Turing | Tesla T4, RTX 20-series |
| 80 | Ampere | A100, RTX 30-series (laptop) |
| 86 | Ampere | RTX 30-series (desktop), A10 |
| 87 | Ampere | **Jetson Orin AGX/NX** |
| 89 | Ada Lovelace | RTX 40-series, L4 |
| 90 | Hopper | H100 |

### Basic Build

```bash
# Install CUDA Toolkit (Ubuntu/Debian)
sudo apt-get install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi

# Build FLANN with CUDA
mkdir build && cd build
cmake -DBUILD_CUDA_LIB=ON ..
make -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_CUDA_LIB` | OFF | Enable CUDA support |
| `CMAKE_CUDA_ARCHITECTURES` | 75;80;86;87;89;90 | Target GPU architectures |
| `NVCC_COMPILER_BINDIR` | (system) | Path to C++ compiler for cross-compilation |

### Architecture Selection Methods

**Method 1: Auto-detect current GPU (fastest build)**
```bash
cmake -DBUILD_CUDA_LIB=ON -DCMAKE_CUDA_ARCHITECTURES=native ..
```

**Method 2: Explicit architecture(s)**
```bash
# Single architecture
cmake -DBUILD_CUDA_LIB=ON -DCMAKE_CUDA_ARCHITECTURES=86 ..

# Multiple architectures (larger binary)
cmake -DBUILD_CUDA_LIB=ON -DCMAKE_CUDA_ARCHITECTURES="75;86;89" ..
```

**Method 3: Environment variable**
```bash
export CUDA_ARCHITECTURES=87
cmake -DBUILD_CUDA_LIB=ON ..
```

**Method 4: Default (all supported)**
```bash
cmake -DBUILD_CUDA_LIB=ON ..
# Builds for: 75, 80, 86, 87, 89, 90
```

### Build Output

Two library variants are produced:
- `libflann_cuda.so` - Shared library
- `libflann_cuda_s.a` - Static library

---

## Using CUDA-Accelerated Algorithms

FLANN provides two CUDA-accelerated index types:

| Index Type | Best For | Distance Metric |
|------------|----------|-----------------|
| **K-Means CUDA** | Floating-point data (SIFT, etc.) | L2 (Euclidean) |
| **Hierarchical CUDA** | Binary descriptors (BRIEF, ORB) | Hamming |

### K-Means CUDA Index

```cpp
#define FLANN_USE_CUDA
#include <flann/flann.hpp>

// Load dataset (floating-point, e.g., SIFT descriptors)
flann::Matrix<float> dataset(data_ptr, num_points, 128);  // N x 128

// Create K-Means CUDA index
flann::KMeansCUDAIndexParams params(
    32,    // branching factor (children per node)
    11,    // k-means iterations per level
    flann::FLANN_CENTERS_RANDOM,  // center initialization
    0.2f   // cb_index (cluster boundary)
);

flann::Index<flann::L2<float>> index(dataset, params);

// Step 1: Build CPU tree structure
index.buildIndex();

// Step 2: Upload to GPU (CRITICAL - do this once!)
int k = 5;  // number of neighbors
index.buildCUDAKnnSearch(k, flann::SearchParams(128));

// Step 3: Search (automatically uses GPU)
flann::Matrix<float> queries(query_ptr, num_queries, 128);
flann::Matrix<size_t> indices(new size_t[num_queries * k], num_queries, k);
flann::Matrix<float> distances(new float[num_queries * k], num_queries, k);

index.knnSearch(queries, indices, distances, k, flann::SearchParams(128));
```

### Hierarchical CUDA Index

```cpp
#define FLANN_USE_CUDA
#include <flann/flann.hpp>

// Load dataset (binary descriptors, e.g., BRIEF)
flann::Matrix<unsigned char> dataset(data_ptr, num_points, 32);  // N x 32 bytes

// Create Hierarchical CUDA index
flann::HierarchicalCUDAIndexParams params(
    32,    // branching factor
    flann::FLANN_CENTERS_RANDOM,  // center initialization
    4,     // number of trees
    100    // max leaf size
);

flann::Index<flann::Hamming<unsigned char>> index(dataset, params);

// Step 1: Build CPU tree structure
index.buildIndex();

// Step 2: Upload to GPU
int k = 3;
index.buildCUDAKnnSearch(k, flann::SearchParams(2000));

// Step 3: Search
flann::Matrix<unsigned char> queries(query_ptr, num_queries, 32);
flann::Matrix<size_t> indices(new size_t[num_queries * k], num_queries, k);
flann::Matrix<int> distances(new int[num_queries * k], num_queries, k);

index.knnSearch(queries, indices, distances, k, flann::SearchParams(2000));
```

### Index Parameters Reference

**KMeansCUDAIndexParams:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branching` | int | 32 | Children per tree node (16-64 typical) |
| `iterations` | int | 11 | K-means iterations per level |
| `centers_init` | enum | RANDOM | `FLANN_CENTERS_RANDOM`, `FLANN_CENTERS_GONZALEZ`, `FLANN_CENTERS_KMEANSPP` |
| `cb_index` | float | 0.2 | Cluster boundary for search radius (0.0-1.0) |

**HierarchicalCUDAIndexParams:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branching` | int | 32 | Children per tree node |
| `centers_init` | enum | RANDOM | Center initialization method |
| `trees` | int | 4 | Number of independent trees |
| `leaf_max_size` | int | 100 | Maximum points per leaf |

**SearchParams:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checks` | int | 32 | Max leaf nodes to search (higher = more accurate) |
| `eps` | float | 0.0 | Approximation factor |
| `sorted` | bool | true | Sort results by distance |

---

## Advantages

### Performance
- **5-20x speedup** over multi-core CPU for large datasets
- **Batch query optimization** - amortize transfer costs over many queries
- **High throughput** - 30,000-40,000 queries/second on modern GPUs

### Precision
- **K-Means CUDA**: 79-100% recall depending on parameters
- **Hierarchical CUDA**: 93-99% recall

### Ease of Use
- **Same API** - `knnSearch()` works identically after GPU setup
- **Explicit GPU setup** - call `buildCUDAKnnSearch(k)` once to enable GPU search

### Tested Results (November 2024)

| Algorithm | Dataset | Precision | Search Speed |
|-----------|---------|-----------|--------------|
| K-Means CUDA | SIFT10K | 99.4% | ~7 µs/query |
| K-Means CUDA | SIFT100K | 95.8% | ~10 µs/query |
| Hierarchical CUDA | Brief100K | 97.2% | ~10 µs/query |

**Test Coverage:** 16/19 tests passing
- K-Means: 8/11 tests pass (3 tests require branching=7, not supported by cooperative kernel)
- Hierarchical: 8/8 tests pass

---

## Disadvantages and Limitations

### Hardware Requirements
- **NVIDIA GPUs only** - no AMD/Intel support (use OpenCL for those)
- **GPU memory** - requires ~2-2.5x dataset size in GPU memory
- **Compute Capability 6.0+** - Pascal architecture or newer

### Overhead
- **One-time upload cost** - 300-2000ms for `buildCUDAKnnSearch()`
- **Not beneficial for small datasets** - overhead dominates below ~10K points
- **Not beneficial for few queries** - transfer cost not amortized below ~100 queries

### Functional Limitations
- **Limited k values** for cooperative kernels (see Performance Tuning)
- **Dynamic updates invalidate GPU** - `addPoints()`/`removePoint()` require re-upload

### When NOT to Use CUDA

| Scenario | Recommendation |
|----------|----------------|
| Dataset < 10,000 points | Use CPU version |
| Single query at a time | Use CPU version |
| No NVIDIA GPU | Use OpenCL or CPU version |
| Frequent index updates | Use CPU version |

---

## Gotchas and Common Issues

### 1. Must Call buildCUDAKnnSearch() Before Searching

```cpp
// WRONG - searches will use CPU fallback
index.buildIndex();
index.knnSearch(...);  // Uses CPU!

// CORRECT
index.buildIndex();
index.buildCUDAKnnSearch(k, params);  // Upload to GPU
index.knnSearch(...);  // Now uses GPU
```

### 2. Test Data Files Must Be in Working Directory

```bash
# Tests expect data files in current directory
cd /path/to/flann/test
./flann_kmeans_cuda_test  # Works

# From different directory - fails
cd /tmp
/path/to/flann/test/flann_kmeans_cuda_test  # Can't find sift100K.h5!
```

### 3. Dynamic Updates Invalidate GPU Data

```cpp
index.buildCUDAKnnSearch(k, params);  // Upload to GPU
index.addPoints(new_points);          // GPU data now STALE!
index.buildCUDAKnnSearch(k, params);  // Must re-upload
```

### 4. LOC_SIZE=1024 Fails on Most GPUs

The cooperative kernel with LOC_SIZE=1024 exceeds register/shared memory limits:
```
CUDA error: too many resources requested for launch
```
Maximum viable LOC_SIZE is 512 for K-Means kernel.

### 5. FLANN_USE_CUDA Must Be Defined Before Including Headers

```cpp
// CORRECT
#define FLANN_USE_CUDA
#include <flann/flann.hpp>

// WRONG - CUDA indices won't be available
#include <flann/flann.hpp>
#define FLANN_USE_CUDA  // Too late!
```

### 6. Linking Requirements

```cmake
# CMakeLists.txt
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_app flann_cuda CUDA::cudart)
```

### 7. Unsupported k-Values Throw Exceptions

Unlike the CPU implementation, CUDA indices **throw exceptions** for unsupported k-values:

```cpp
// K-Means CUDA supports: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100
index.buildCUDAKnnSearch(3, params);  // THROWS FLANNException!

// Hierarchical CUDA supports: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128
index.buildCUDAKnnSearch(17, params);  // THROWS FLANNException!
```

**There is NO automatic CPU fallback.** If you need an unsupported k-value, use the CPU version.

### 8. Thread Safety

**CUDA indices are NOT thread-safe for concurrent searches.** A single index instance should not be used from multiple threads simultaneously.

```cpp
// WRONG - concurrent access causes undefined behavior
std::thread t1([&index]{ index.knnSearch(...); });
std::thread t2([&index]{ index.knnSearch(...); });

// CORRECT - each thread has its own index
auto createIndex = [&dataset]() {
    auto index = Index<L2<float>>(dataset, KMeansCUDAIndexParams(...));
    index.buildIndex();
    index.buildCUDAKnnSearch(k, params);
    return index;
};

std::thread t1([&]{ auto idx = createIndex(); idx.knnSearch(...); });
std::thread t2([&]{ auto idx = createIndex(); idx.knnSearch(...); });
```

---

## Cross-Compilation for Jetson Orin AGX

The Jetson Orin AGX uses compute capability **87** (sm_87).

### Option 1: Native Compilation on Jetson (Recommended)

```bash
# On Jetson Orin AGX (JetPack 5.x or 6.x installed)
git clone https://github.com/your/flann.git
cd flann
mkdir build && cd build

# Build for native architecture
cmake -DBUILD_CUDA_LIB=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
      ..
make -j$(nproc)
```

### Option 2: Cross-Compilation from x86_64 Host

**Prerequisites:**
- NVIDIA SDK Manager with JetPack installed
- Cross-compilation toolchain from JetPack
- CUDA cross-compilation toolkit

```bash
# Set up cross-compilation environment
export JETPACK_ROOT=/path/to/jetpack
export CROSS_COMPILE=${JETPACK_ROOT}/aarch64-linux-gnu/bin/aarch64-linux-gnu-

# Configure cross-compilation
cmake -DBUILD_CUDA_LIB=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
      -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ \
      -DNVCC_COMPILER_BINDIR=${JETPACK_ROOT}/aarch64-linux-gnu/bin \
      ..

make -j$(nproc)
```

### Option 3: Docker Cross-Compilation

```dockerfile
# Dockerfile.jetson-cross
FROM nvcr.io/nvidia/l4t-cuda:11.4.19-devel

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libhdf5-dev

# Copy source
COPY . /flann
WORKDIR /flann/build

# Configure and build
RUN cmake -DBUILD_CUDA_LIB=ON \
          -DCMAKE_CUDA_ARCHITECTURES=87 \
          .. && \
    make -j$(nproc)
```

### JetPack Requirements

| JetPack Version | CUDA Version | L4T Version | Notes |
|-----------------|--------------|-------------|-------|
| JetPack 5.x | CUDA 11.4 | L4T R35.x | Orin AGX, Orin NX |
| JetPack 6.x | CUDA 12.x | L4T R36.x | Latest, recommended |

### Jetson-Specific Optimizations

```bash
# Maximize GPU performance on Jetson
sudo nvpmodel -m 0  # MAX performance mode
sudo jetson_clocks  # Lock clocks to maximum
```

### Verifying Jetson Build

```bash
# On Jetson
./flann_hierarchical_cuda_test --gtest_filter="*TestSearch*"

# Expected output:
# [       OK ] HierarchicalCUDA_Brief100K.TestSearch (XXX ms)
# Precision: 0.96+
```

---

## Performance Tuning

### LOC_SIZE (Thread Block Size)

LOC_SIZE controls the number of threads per query in cooperative kernels. Higher values improve precision but increase latency.

**K-Means CUDA Benchmarks (SIFT100K, k=5):**

| LOC_SIZE | Precision | Latency | Notes |
|----------|-----------|---------|-------|
| 32 | 87.7% | 5.91 µs/query | Fastest, lowest precision |
| 64 | 93.4% | 7.74 µs/query | |
| **128** | **97.4%** | **14.73 µs/query** | **Default - best balance** |
| 256 | 99.3% | 16.57 µs/query | High precision |
| 512 | 100% | 30.78 µs/query | Maximum precision |
| 1024 | FAILED | - | Exceeds GPU resource limits |

**Hierarchical CUDA Benchmarks (Brief100K, k=3):**

| LOC_SIZE | Precision | Latency | Notes |
|----------|-----------|---------|-------|
| 128 | 90.5% | 37.1 µs/query | Fastest |
| **256** | **96.8%** | **41.8 µs/query** | **Default - best balance** |
| 512 | 99.3% | 49.1 µs/query | Highest precision |

### Supported k Values

**IMPORTANT:** K-Means CUDA and Hierarchical CUDA support different k-values.

#### K-Means CUDA

**Supported k values:** 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100 (13 values)

**Requirements:**
- Branching factor must be 32 or 64
- Unsupported k values will throw an exception

#### Hierarchical CUDA

**Supported k values:** 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128 (16 values)

**Performance Tiers:**

| Tier | k Values | Optimization | Speedup |
|------|----------|--------------|---------|
| **Tier 1** (optimal) | 16, 32, 64, 128 | int4 vectorization + cache-aligned | 20-25% |
| **Tier 2** (good) | 8, 24 | int4 vectorization | 15-20% |
| **Tier 3** (vectorized) | 4, 12, 20 | int4 vectorization | 10-15% |
| **Tier 4** (compatible) | 1, 2, 3, 5, 10, 50, 100 | Scalar fallback | Baseline |

**Recommendation:** Choose k that is a **multiple of 16** for best performance.

### Dataset Size Guidelines

| Dataset Size | Recommendation |
|--------------|----------------|
| < 1,000 points | Use CPU - overhead dominates |
| 1,000 - 10,000 | CPU usually faster unless batch queries |
| 10,000 - 100,000 | GPU beneficial for batch queries (100+) |
| > 100,000 | GPU strongly recommended |

### Checks Parameter

The `checks` parameter in SearchParams controls search depth:

```cpp
// Lower checks = faster but less accurate
flann::SearchParams(32)   // Fast, ~85% recall

// Higher checks = slower but more accurate
flann::SearchParams(128)  // Balanced, ~95% recall
flann::SearchParams(512)  // Thorough, ~99% recall
```

### Memory Optimization

```cpp
// Persistent GPU buffers (handled internally)
// The library pre-allocates GPU buffers during buildCUDAKnnSearch()
// and reuses them across searches for ~14% speedup.

// Note: Pinned memory (cudaMallocHost) provides NO benefit for small
// transfers (<1MB). A/B testing showed identical performance for the
// ~60KB query/result buffers used in typical searches.
```

### Profiling

```bash
# NVIDIA profiler
nsys profile ./my_flann_app

# Detailed kernel analysis
ncu --set full ./my_flann_app
```

---

## Troubleshooting

### Build Errors

**"nvcc not found"**
```bash
export PATH=/usr/local/cuda/bin:$PATH
# Or install CUDA Toolkit
```

**"Unsupported GPU architecture"**
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
# Adjust CMAKE_CUDA_ARCHITECTURES accordingly
```

### Runtime Errors

**"CUDA error: out of memory"**
- Reduce dataset size
- Use `nvidia-smi` to check available memory
- Close other GPU applications

**"CUDA error: too many resources requested for launch"**
- LOC_SIZE is too high for your GPU
- Try LOC_SIZE=128 or LOC_SIZE=256

**Poor performance / slower than CPU**
- Dataset too small (< 10K points)
- Too few queries (< 100)
- Check `nvidia-smi` - GPU might be throttling

### Verification

```bash
# Run CUDA tests
cd /path/to/flann/test
./flann_kmeans_cuda_test
./flann_hierarchical_cuda_test

# Expected: All tests PASSED with precision > 90%
```

---

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [FLANN Paper](https://www.cs.ubc.ca/research/flann/) - Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009
- [JetPack Documentation](https://developer.nvidia.com/embedded/jetpack)
