# FLANN CUDA Guide

This guide covers building, using, and optimizing FLANN with CUDA GPU acceleration.

## Table of Contents

1. [Building with CUDA Support](#building-with-cuda-support)
2. [Using CUDA-Accelerated Algorithms](#using-cuda-accelerated-algorithms)
   - [GPU Index Format (v2.0)](#gpu-index-format-v20)
3. [Advantages](#advantages)
   - [Benchmark Results](#benchmark-results-december-2024)
4. [Disadvantages and Limitations](#disadvantages-and-limitations)
5. [Gotchas and Common Issues](#gotchas-and-common-issues)
6. [Cross-Compilation for Jetson Orin AGX](#cross-compilation-for-jetson-orin-agx)
7. [Performance Tuning](#performance-tuning)
8. [Benchmarking Tools](#benchmarking-tools)

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

### GPU Index Format (v2.0)

FLANN provides a GPU-optimized index format (v2.0) that dramatically reduces cold-start times by storing pre-computed GPU arrays directly in the file.

#### Benefits of GPU v2.0 Format

| Metric | CPU-to-GPU Conversion | GPU v2.0 Format | Improvement |
|--------|----------------------|-----------------|-------------|
| **Binary Hierarchical** (77K points) | 37 ms total ready | 9 ms total ready | **4.2x faster** |
| **Float KMeans** (100K points) | 251 ms total ready | 121 ms total ready | **2.1x faster** |
| File size overhead | Baseline | +0-2% | Negligible |

#### Converting to GPU v2.0 Format

```bash
# Use the conversion utility
./bin/flann_convert_to_gpu input.db output.gpu.idx --verify --verbose

# Example with real data
./bin/flann_convert_to_gpu \
  my_index.db \
  my_index.gpu.idx \
  --k=10 --checks=128 --verify
```

#### Loading GPU v2.0 Format

```cpp
// GPU v2.0 format is automatically detected on load
flann::Matrix<float> empty;
flann::Index<flann::L2<float>> index(empty, flann::SavedIndexParams("index.gpu.idx"));

// GPU setup is much faster for v2.0 format (~1ms vs ~150ms)
index.buildCUDAKnnSearch(10, flann::SearchParams(128));

// Search normally
index.knnSearch(queries, indices, distances, k, params);
```

#### When to Use GPU v2.0 Format

| Use Case | Recommendation |
|----------|----------------|
| Containerized deployments | **Use GPU v2.0** - cold starts benefit most |
| Serverless functions | **Use GPU v2.0** - minimize startup latency |
| Long-running services | Either format - setup cost amortizes |
| Frequent index updates | Use CPU format - easier to rebuild |

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
- **2-10x search speedup** over CPU depending on dataset size and type
- **4x faster cold starts** with GPU v2.0 format vs CPU-to-GPU conversion
- **High throughput** - 100,000+ queries/second on modern GPUs

### Precision
- **K-Means CUDA**: 89-100% recall depending on parameters
- **Hierarchical CUDA**: 90-99% recall

### Ease of Use
- **Same API** - `knnSearch()` works identically after GPU setup
- **Explicit GPU setup** - call `buildCUDAKnnSearch(k)` once to enable GPU search
- **GPU v2.0 format** - pre-built GPU indices for fast cold starts

### Benchmark Results (December 2024)

#### Search Performance (k=10, checks=128)

| Algorithm | Dataset | CPU Search | GPU Search | Speedup | GPU Precision |
|-----------|---------|------------|------------|---------|---------------|
| Hierarchical | 77K × 64 binary | 17.8 ms | 9.8 ms | **1.8x** | 90.2% |
| K-Means | 100K × 128 float | 92.4 ms | 9.7 ms | **9.5x** | 89.4% |

#### Index Loading Performance

| Format | Binary Hierarchical | Float KMeans | Notes |
|--------|---------------------|--------------|-------|
| CPU-only load | 9.5 ms | 98 ms | No GPU setup |
| CPU-to-GPU | 37 ms | 251 ms | Includes GPU upload |
| **GPU v2.0** | **9 ms** | **121 ms** | Pre-built GPU arrays |
| **Speedup** | **4.2x** | **2.1x** | vs CPU-to-GPU |

#### Throughput Comparison

| Scenario | Binary Hierarchical | Float KMeans |
|----------|---------------------|--------------|
| CPU-only | 56,000 q/s | 11,000 q/s |
| GPU | 102,000 q/s | 103,000 q/s |
| **Speedup** | **1.8x** | **9.4x** |

**Test Coverage:** All CUDA tests passing

### CUDA vs OpenCL Comparison (December 2024)

Both CUDA and OpenCL backends are available for GPU acceleration. This section compares their performance using the same CPU index files with proper warmup queries.

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

#### Binary Hierarchical (brief100K - 100K × 32 bytes, Hamming)

| Metric | OpenCL | CUDA | Winner |
|--------|--------|------|--------|
| Build time (CPU) | 0.303s ± 0.008 | 0.305s ± 0.002 | ~Equal |
| **GPU Upload** | **0.357s** | **0.014s** | **CUDA 25x faster** |
| Search (warm) | 0.005s | 0.005s | ~Equal |
| **Precision** | **81.7%** | **89.7%** | **CUDA +8%** |
| Throughput | 208,928 q/s | 182,498 q/s | OpenCL slightly higher |

#### Float K-Means (sift100K - 99K × 128 floats, L2)

| Metric | OpenCL | CUDA | Winner |
|--------|--------|------|--------|
| Build time (CPU) | 10.74s ± 0.16 | 10.83s ± 0.11 | ~Equal |
| **GPU Upload** | **0.493s** | **0.152s** | **CUDA 3.2x faster** |
| Search (warm) | 0.012s | 0.010s | CUDA 17% faster |
| **Precision** | **77.2%** | **88.4%** | **CUDA +11%** |
| Throughput | 86,238 q/s | 97,194 q/s | CUDA 13% higher |

#### Key Takeaways

| Aspect | CUDA Advantage | Notes |
|--------|----------------|-------|
| **GPU Upload** | 3-25x faster | CUDA's optimized memory transfer |
| **Precision** | +8-11% higher | Better search coverage |
| **Search Speed** | ~Equal to 17% faster | Similar post-warmup |
| **Hardware** | NVIDIA only | OpenCL supports AMD/Intel |

**Recommendation:** Use CUDA when NVIDIA hardware is available; use OpenCL for multi-vendor support or AMD/Intel GPUs.

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

### Query Matrix Requirements

GPU search requires contiguous query matrices for efficient upload:

```cpp
// CORRECT - contiguous allocation
float* data = new float[num_queries * 128];
flann::Matrix<float> queries(data, num_queries, 128);  // stride = cols * sizeof(T)

// CORRECT - std::vector (always contiguous)
std::vector<float> data(num_queries * 128);
flann::Matrix<float> queries(data.data(), num_queries, 128);

// WRONG - non-contiguous will throw FLANNException
flann::Matrix<float> queries(ptr, rows, cols, custom_stride);  // stride != cols * sizeof(T)
```

**Dimension alignment:** For best performance, use dimensions that are multiples of 4 (e.g., 128, 256). Non-aligned dimensions (e.g., 127) work but require GPU padding and will produce a one-time warning:
```
[FLANN] Dimension 127 requires padding to 128. Use multiples of 4 for best performance.
```

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

**CUDA indices ARE thread-safe for concurrent searches.** Multiple threads can safely call `knnSearch()` or `knnSearchGPU()` on the same index instance simultaneously. This is achieved through:

1. **Reader-writer locking (`std::shared_mutex`):** Concurrent searches acquire shared (read) locks, allowing full parallelism. Mutations (addPoints, removePoint) acquire exclusive (write) locks and wait for all searches to complete.

2. **Per-thread CUDA streams:** Each thread gets its own CUDA stream via `thread_local` storage, enabling concurrent GPU kernel execution without stream contention.

3. **Per-search resource allocation:** Query buffers and result buffers are allocated per-search, eliminating data races on temporary storage.

**Concurrency Model:**

| Operation | Lock Type | Concurrent With |
|-----------|-----------|-----------------|
| `knnSearch()` | Shared (read) | Other searches |
| `addPoints()` | Exclusive (write) | Nothing |
| `removePoint()` | Exclusive (write) | Nothing |
| `uploadToGPU()` | Exclusive (write) | Nothing |
| `buildCUDAKnnSearch()` | Exclusive (write) | Nothing |

**Usage Examples:**

```cpp
// Thread-safe concurrent searches (automatically parallelized)
cuda::KMeansCUDAIndex<L2<float>> index(dataset, cuda::KMeansCUDAIndexParams(32));
index.buildIndex();
index.buildCUDAKnnSearch(k, SearchParams());

// Launch multiple search threads - fully concurrent
std::vector<std::thread> threads;
for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&index, &queries, k]() {
        Matrix<int> indices(new int[queries.rows * k], queries.rows, k);
        Matrix<float> dists(new float[queries.rows * k], queries.rows, k);

        // Safe to call from multiple threads simultaneously
        index.knnSearch(queries, indices, dists, k, SearchParams());

        // ... process results ...
        delete[] indices.ptr();
        delete[] dists.ptr();
    });
}

for (auto& t : threads) t.join();
```

**Performance Characteristics:**

| Aspect | Impact |
|--------|--------|
| Lock overhead | ~10ns per search (negligible vs GPU time) |
| Stream creation | ~0.5ms first call per thread, then cached |
| Memory per thread | ~100KB (result buffers) |
| GPU parallelism | Full concurrent kernel execution |

**Note:** While searches are thread-safe, mutations (addPoints, removePoint) will block all concurrent searches until complete. For workloads with frequent mutations, consider using separate index instances per thread.

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

## Benchmarking Tools

FLANN includes several benchmark utilities for measuring CPU vs GPU performance.

### benchmark_cpu_gpu_comparison

Comprehensive benchmark comparing CPU-only, CPU-to-GPU, and GPU v2.0 loading paths.

```bash
# Binary Hierarchical benchmark (real-world test data)
./test/benchmark_cpu_gpu_comparison \
  /path/to/cpu_index.db \
  /path/to/gpu_index.idx \
  --k=10 --runs=5 --csv=results.csv

# Float KMeans benchmark (from HDF5 dataset)
./test/benchmark_cpu_gpu_comparison --dataset=datasets/sift100K.h5

# Run all benchmarks
./test/benchmark_cpu_gpu_comparison --all --csv=full_results.csv
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--k=N` | 10 | Number of nearest neighbors |
| `--checks=N` | 128 | Search quality parameter |
| `--runs=N` | 5 | Number of timed runs |
| `--warmup=N` | 2 | Number of warmup runs |
| `--queries=N` | 1000 | Number of test queries |
| `--csv=FILE` | - | Export results to CSV |
| `--verbose` | - | Detailed output |

### benchmark_gpu_index_stats

Measures file size and loading time differences between CPU and GPU index formats.

```bash
./test/benchmark_gpu_index_stats --dataset=datasets/sift100K.h5
```

### flann_convert_to_gpu

Converts CPU format indices to GPU v2.0 format for faster loading.

```bash
./bin/flann_convert_to_gpu input.db output.gpu.idx --verify --verbose
```

---

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [FLANN Paper](https://www.cs.ubc.ca/research/flann/) - Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009
- [JetPack Documentation](https://developer.nvidia.com/embedded/jetpack)
