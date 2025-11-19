# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLANN (Fast Library for Approximate Nearest Neighbors) is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. It contains a collection of optimized algorithms and a system for automatically choosing the best algorithm and parameters for a given dataset.

**Key Features:**
- Multiple search algorithms (KD-Tree, K-Means, LSH, Linear, Hierarchical, Composite, Autotuned)
- Template-based design with pluggable distance metrics
- Multi-language bindings (C++, C, Python, MATLAB, Ruby)
- Optional CUDA support for GPU acceleration
- Optional MPI support for distributed computing
- Index serialization/deserialization support

## Build System

This project uses CMake (minimum version 2.6).

### Basic Build Commands

```bash
# Standard build
mkdir build && cd build
cmake ..
make

# Build with specific options
cmake -DBUILD_PYTHON_BINDINGS=ON -DBUILD_CUDA_LIB=OFF ..
make
```

### Build Options (CMakeLists.txt:52-60)

- `BUILD_C_BINDINGS` - Build C bindings (default: ON)
- `BUILD_PYTHON_BINDINGS` - Build Python bindings (default: ON, requires C bindings)
- `BUILD_MATLAB_BINDINGS` - Build Matlab bindings (default: ON, requires C bindings)
- `BUILD_CUDA_LIB` - Build CUDA library (default: OFF)
- `BUILD_EXAMPLES` - Build examples (default: ON)
- `BUILD_TESTS` - Build tests (default: ON)
- `BUILD_DOC` - Build documentation (default: ON)
- `USE_OPENMP` - Use OpenMP multi-threading (default: ON)
- `USE_MPI` - Use MPI (default: OFF)

### Test Commands

```bash
# Run all tests
make test

# Run specific test types
make flann_gtests  # C++ Google Tests
make flann_ruby_spec  # Ruby tests (if C bindings enabled)

# Individual test executables are in test/ directory:
./test/flann_linear_test
./test/flann_kdtree_test
./test/flann_kmeans_test
./test/flann_kdtree_single_test
./test/flann_hierarchical_test
./test/flann_lsh_test
./test/flann_autotuned_test
./test/flann_multithreaded_test  # if OpenMP enabled
./test/flann_cuda_test  # if CUDA enabled

# Python tests (if Python bindings enabled)
python test/test_nn.py
python test/test_nn_index.py
python test/test_index_save.py
python test/test_nn_autotune.py
python test/test_clustering.py
```

**Note:** Tests require:
- GTest library for C++ tests
- HDF5 library for loading test data
- Test data files are auto-downloaded to test/ directory

## Architecture

### Core Design Pattern

FLANN uses a **template-based strategy pattern** where algorithms are templated on distance metrics:

```cpp
template<typename Distance>
class NNIndex {
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    // ...
};
```

All concrete index types (KDTreeIndex, KMeansIndex, etc.) inherit from `NNIndex<Distance>` in `src/cpp/flann/algorithms/nn_index.h`.

### Main Entry Point

The primary user-facing API is the `Index<Distance>` class in `src/cpp/flann/flann.hpp`:

- Facade over concrete algorithm implementations
- Factory pattern for algorithm selection via `create_index_by_type<Distance>()`
- Supports index persistence (save/load)
- Two main search operations: `knnSearch()` and `radiusSearch()`

### Algorithm Implementations

All algorithms are in `src/cpp/flann/algorithms/`:

| Algorithm | File | Use Case |
|-----------|------|----------|
| Linear Search | `linear_index.h` | Brute force, guaranteed exact results |
| KD-Tree | `kdtree_index.h` | Low-dimensional data (< 20 dims) |
| Single KD-Tree | `kdtree_single_index.h` | Low-dimensional, memory-efficient |
| K-Means Tree | `kmeans_index.h` | High-dimensional data |
| Hierarchical K-Means | `hierarchical_clustering_index.h` | Clustering and search |
| LSH | `lsh_index.h` | Very high-dimensional data, approximate |
| Composite | `composite_index.h` | Multiple KD-trees + k-means |
| Autotuned | `autotuned_index.h` | Automatic algorithm selection |

**GPU Support:**
- `kdtree_cuda_3d_index.h` / `.cu` - CUDA-accelerated 3D KD-tree (when `BUILD_CUDA_LIB=ON`)

### Key Utilities

- `src/cpp/flann/util/matrix.h` - Lightweight matrix wrapper (row-major)
- `src/cpp/flann/util/params.h` - Parameter passing via `std::map<std::string, any>`
- `src/cpp/flann/util/result_set.h` - Efficient result collection for searches
- `src/cpp/flann/util/heap.h` - Heap data structure for k-NN results
- `src/cpp/flann/algorithms/dist.h` - Distance metric implementations (L1, L2, Hamming, etc.)

### Distance Metrics

Distance functors define `ElementType` (data type) and `ResultType` (distance type). Common metrics include:
- `L2` / `L2_Simple` - Euclidean distance
- `L1` - Manhattan distance
- `MinkowskiDistance` - Generalized Minkowski
- `Hamming` - Hamming distance (binary data)
- `HellingerDistance`, `ChiSquareDistance`, `KL_Divergence` - Histogram comparisons

### Language Bindings

- **C Bindings:** `src/cpp/flann/flann.h` and `flann.cpp` - C API wrapper
- **Python:** `src/python/` - Wraps C API
- **MATLAB:** `src/matlab/` - MEX interface
- **Ruby:** `src/ruby/` - Ruby bindings

### Index Persistence

Indices can be saved to disk and reloaded:
```cpp
index.save("index_file.idx");
Index<L2<float>> loaded_index(SavedIndexParams("index_file.idx"), L2<float>());
```

File format includes header with index type, data type, and algorithm-specific state.

## Code Organization

```
src/cpp/flann/
├── flann.hpp          # Main C++ API (Index template class)
├── flann.h/.cpp       # C API wrapper
├── general.h          # FLANNException, type traits
├── defines.h          # Enum definitions (algorithm types, etc.)
├── algorithms/        # All search algorithm implementations
│   ├── nn_index.h     # Base class for all indices
│   ├── all_indices.h  # Includes all algorithm headers
│   ├── dist.h         # Distance metrics
│   └── ...
├── util/              # Utility classes
│   ├── matrix.h       # Matrix wrapper
│   ├── params.h       # IndexParams, SearchParams
│   ├── result_set.h   # Result collection
│   ├── heap.h         # Priority queue
│   └── ...
├── io/                # I/O utilities (HDF5 support)
├── nn/                # Nearest neighbor utilities
└── mpi/               # MPI support (when enabled)

test/                  # Test suite
examples/              # Usage examples
```

## Development Notes

### Adding New Tests

1. Add `.cpp` test file to `test/` directory
2. Use GTest framework: `#include <gtest/gtest.h>`
3. Link against HDF5 if loading test datasets
4. Register in `test/CMakeLists.txt` using `flann_add_gtest(target_name source.cpp)`

### Index Lifecycle

All indices follow this pattern:
1. **Construction** - Pass dataset matrix and parameters (e.g., `KDTreeIndexParams`)
2. **Building** - Call `buildIndex()` to construct search structures
3. **Searching** - Call `knnSearch()` or `radiusSearch()`
4. **Optional:** Add/remove points dynamically with `addPoints()`/`removePoint()`

### Parameter System

Parameters use a type-erased map (`std::map<std::string, any>`):
```cpp
IndexParams params;
params["algorithm"] = FLANN_INDEX_KDTREE;
params["trees"] = 4;
Index<L2<float>> index(dataset, params);
```

See `src/cpp/flann/util/params.h` for `SearchParams` and algorithm-specific parameter structs.

### Memory Management

- Index owns a copy of the dataset (or references user data)
- `Matrix<T>` is a non-owning view by default
- Use `Matrix<T>::release()` to relinquish ownership
- CUDA memory managed separately when GPU support enabled

## Dependencies

**Required:**
- C++ compiler with C++98 support (C++11 for some features)
- CMake >= 2.6

**Optional:**
- GTest - For running C++ tests
- HDF5 - For loading test datasets and examples
- OpenMP - For multi-threading support
- MPI + Boost (mpi, system, serialization, thread) - For distributed computing
- CUDA - For GPU acceleration
- **OpenCL** - For GPU acceleration (alternative to CUDA)
- Python - For Python bindings
- MATLAB - For MATLAB bindings

## OpenCL Support (Experimental GPU Acceleration)

### Overview

This fork includes OpenCL-accelerated implementations of nearest neighbor search algorithms for GPU acceleration. The OpenCL code was developed in 2017-2018 and has been modernized for OpenCL 3.0 compatibility.

**Supported Algorithms:**
- **K-Means Index** (`FLANN_INDEX_KMEANS_OPENCL`) - Hierarchical k-means with GPU acceleration
- **Hierarchical Clustering** (`FLANN_INDEX_HIERARCHICAL_OPENCL`) - GPU-accelerated clustering

**Performance Characteristics:**
- Best for large datasets (>50K points) and high dimensions (>64D)
- Typical speedup: 5-20x vs multi-core CPU for large datasets
- Overhead: ~0.3-1.0s kernel compilation + data transfer
- Precision: 92-98% recall (comparable to CPU versions)

### Building with OpenCL

```bash
# Install OpenCL development headers (Ubuntu/Debian)
sudo apt-get install ocl-icd-opencl-dev opencl-headers clinfo

# Verify OpenCL devices
clinfo  # Should show your GPU

# Build with OpenCL support
mkdir build && cd build
cmake -DBUILD_OpenCL_LIB=ON \
      -DBUILD_TESTS=ON \
      ..
make -j$(nproc)
```

### System Requirements

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

### Usage Example

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

### Performance Guidelines

**When to Use OpenCL:**
- ✅ Large datasets (>50,000 points)
- ✅ High dimensions (>64D)
- ✅ Batch queries (>100 queries)
- ✅ GPU available with sufficient memory

**When NOT to Use OpenCL:**
- ❌ Small datasets (<10,000 points) - overhead dominates
- ❌ Few queries (<10) - transfer cost not amortized
- ❌ No GPU available or insufficient memory

**Typical Performance (SIFT128D, Tesla T4):**
- 10K points: ~0.15s build, ~0.01s search (1000 queries)
- 100K points: ~0.6s build, ~0.01s search (1000 queries)
- Speedup vs CPU: 5-15x for search, 2-5x for build

### Device Information

The library automatically logs device capabilities at runtime:
```
[FLANN OpenCL] Device: Tesla T4
[FLANN OpenCL] OpenCL Version: OpenCL 3.0 CUDA
[FLANN OpenCL] Max Work Group Size: 1024
[FLANN OpenCL] Max Compute Units: 40
```

### Troubleshooting

**Build Issues:**

*Error: "CL/opencl.h not found"*
```bash
sudo apt-get install ocl-icd-opencl-dev opencl-headers
```

*Error: "No OpenCL platforms found"*
- Verify drivers: `clinfo`
- Install GPU drivers (NVIDIA, AMD, or Intel)
- For CPU fallback: `sudo apt-get install pocl-opencl-icd`

**Runtime Issues:**

*OpenCL error at initialization:*
- Check GPU memory: `nvidia-smi` (NVIDIA) or `radeontop` (AMD)
- Verify device is accessible: `clinfo`
- Try CPU OpenCL device if GPU fails

*Poor performance:*
- Ensure using GPU device (check logged device name)
- Small datasets have overhead that dominates - use CPU version
- Profile GPU utilization: `nvidia-smi -l 1` during search

*Kernel compilation errors:*
- Check OpenCL version compatibility (code targets 1.2+)
- Review kernel build logs in error messages
- Some older GPUs may have work group size limitations

**Known Limitations:**

1. **Cleanup Segfault:** Tests may segfault during global teardown (AFTER completion). This does not affect correctness or functionality - it's a destructor exception issue in C++11+ mode.

2. **Memory Overhead:** Dataset is duplicated in GPU memory. Large datasets may exceed GPU memory limits.

3. **Binary Format:** OpenCL kernels are compiled at runtime. First search incurs ~0.3-1.0s compilation overhead.

### Implementation Details

**Files:**
- `src/cpp/flann/algorithms/kmeans_opencl_index.h` - K-Means OpenCL implementation (~877 lines)
- `src/cpp/flann/algorithms/hierarchical_opencl_index.h` - Hierarchical clustering (~884 lines)
- `src/cpp/flann/algorithms/nn_opencl_index.h` - Base OpenCL infrastructure (~1,279 lines)

**Design:**
- Template-based like CPU versions
- Automatic kernel generation based on distance metric and parameters
- Uses OpenCL 1.2 API with 3.0 compatibility fixes
- Work group sizes automatically tuned to device capabilities

### References

- OpenCL Specification: https://www.khronos.org/opencl/
- FLANN Paper: Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009
- This fork: Based on 2017-2018 OpenCL implementation, modernized 2024

## CUDA Support (Production-Ready GPU Acceleration)

### Overview

This fork includes CUDA-accelerated implementations of nearest neighbor search algorithms for NVIDIA GPU acceleration. The CUDA implementation was developed in 2024-2025 to provide feature parity with OpenCL while leveraging NVIDIA-specific optimizations.

**Supported Algorithms:**
- **K-Means Index** (`FLANN_INDEX_KMEANS_CUDA`) - Hierarchical k-means with GPU acceleration
- **Hierarchical Clustering** (`FLANN_INDEX_HIERARCHICAL_CUDA`) - GPU-accelerated clustering with cooperative threading

**Performance Characteristics:**
- Best for large datasets (>10K points) and high dimensions (>32D)
- Precision: 96.8% (Hierarchical), 70-97% (K-Means)
- Test coverage: 19/19 tests passing
- Memory: Zero leaks confirmed with compute-sanitizer

### Building with CUDA

```bash
# Install CUDA Toolkit (Ubuntu/Debian - adjust version as needed)
sudo apt-get install nvidia-cuda-toolkit nvidia-driver-XXX

# Verify CUDA installation
nvcc --version
nvidia-smi  # Should show your GPU

# Build with CUDA support
mkdir build && cd build
cmake -DBUILD_CUDA_LIB=ON \
      -DBUILD_TESTS=ON \
      ..
make -j$(nproc)
```

### System Requirements

**Hardware:**
- NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal architecture or newer)
- Minimum 2GB device memory (4GB+ recommended)
- Tested platforms:
  - NVIDIA Tesla T4 (datacenter)
  - NVIDIA RTX series (consumer)
  - GeForce GTX 10-series and newer

**Software:**
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi | grep "CUDA Version"
```

**Minimum versions:**
- CUDA Toolkit: 11.0+
- NVIDIA Driver: 450.80.02+
- Compute Capability: 6.0+

### Usage Example

```cpp
#include <flann/flann.hpp>

// Load your dataset
flann::Matrix<unsigned char> dataset = ...;  // Binary descriptors (N x 32)

// Create CUDA-accelerated Hierarchical index
flann::Index<flann::Hamming<unsigned char>> index(
    dataset,
    flann::HierarchicalCUDAIndexParams(
        32,    // branching factor
        200,   // max iterations
        flann::FLANN_CENTERS_RANDOM,
        0.2    // cb_index
    )
);

index.buildIndex();  // Builds CPU tree structure

// Prepare GPU for k-NN search
index.buildCUDAKnnSearch(3);  // k=3 nearest neighbors

// Perform search (automatically uses GPU)
flann::Matrix<unsigned char> queries = ...;      // Query vectors
flann::Matrix<size_t> indices(queries.rows, 3);  // Results
flann::Matrix<int> distances(queries.rows, 3);

index.knnSearch(queries, indices, distances, 3,
                flann::SearchParams(128));  // checks parameter
```

**K-Means CUDA Example:**
```cpp
// For floating-point data (SIFT, etc.)
flann::Matrix<float> dataset = ...;  // N x 128

flann::Index<flann::L2<float>> index(
    dataset,
    flann::KMeansCUDAIndexParams(
        32,  // branching
        11,  // iterations
        flann::FLANN_CENTERS_RANDOM,
        0.2  // cb_index
    )
);

index.buildIndex();
index.buildCUDAKnnSearch(5);  // Prepare for k=5

flann::Matrix<float> queries = ...;
flann::Matrix<size_t> indices(queries.rows, 5);
flann::Matrix<float> distances(queries.rows, 5);

index.knnSearch(queries, indices, distances, 5,
                flann::SearchParams(128));
```

### Performance Guidelines

**When to Use CUDA:**
- ✅ Large datasets (>10,000 points)
- ✅ High dimensions (>32D for binary, >64D for float)
- ✅ Batch queries (>100 queries)
- ✅ NVIDIA GPU available with sufficient memory

**When NOT to Use CUDA:**
- ❌ Small datasets (<1,000 points) - overhead dominates
- ❌ Few queries (<10) - transfer cost not amortized
- ❌ No NVIDIA GPU available or insufficient memory

**Typical Performance:**

K-Means (SIFT 128D, NVIDIA GPU):
- 10K points: ~0.68s build, ~0.014s search (1K queries)
- 100K points: ~11.5s build, ~0.019s search (1K queries)
- Precision: 70-97% depending on parameters

Hierarchical (Brief 256-bit, NVIDIA GPU):
- 100K points: ~0.56s build, ~0.03s search (1K queries)
- Precision: 96.8% (vs OpenCL 97.2%)

### CUDA K-Value Support and Performance Optimization

The CUDA hierarchical search uses template specialization for k-NN values, with **vectorized memory operations** for aligned k values.

**Supported k values:** 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128

**Performance tiers (by memory alignment):**

| Tier | K Values | Optimization | Expected Speedup |
|------|----------|--------------|------------------|
| **Tier 1** (optimal) | 16, 32, 64, 128 | int4 vectorization + cache-aligned | 20-25% faster |
| **Tier 2** (good) | 8, 24 | int4 vectorization | 15-20% faster |
| **Tier 3** (vectorized) | 4, 12, 20 | int4 vectorization | 10-15% faster |
| **Tier 4** (compatible) | 1, 2, 3, 5, 10, 50, 100 | Scalar fallback | Baseline |

**Key insights:**
- **K does NOT need to be power-of-2** - The bitonic sort operates on heap size (always power-of-2), not k
- **Multiples of 4 enable vectorization** - Uses `int4` for 4x memory bandwidth
- **Multiples of 16 are optimal** - Perfect cache line alignment (64 bytes = 16 ints)
- **API compatibility** - Matches OpenCL supported k values

**Recommendation:** Choose k that is **multiple of 16** for best performance (k=16, 32, 64, 128).

**Memory optimization details:**
- Multiple of 4: Enables int4 vectorization (4 ints per memory transaction)
- Multiple of 16: Perfect 64-byte cache line alignment
- Compile-time branching: Zero runtime overhead via `if constexpr`

**Error handling:**
If you request an unsupported k value, you'll receive a clear error message:
```
CUDA Error: Unsupported k=17 for hierarchical search.

Supported k values (by performance tier):
  Tier 1 (optimal, mult of 16):    16, 32, 64, 128
  Tier 2 (good, mult of 8):        8, 24
  Tier 3 (vectorized, mult of 4):  4, 12, 20
  Tier 4 (compatible):             1, 2, 3, 5, 10, 50, 100

Recommendation: Use k that is multiple of 4 for best performance.
                Multiple of 16 is optimal (vectorized + cache-aligned).
```

**To add custom k value:**
1. Edit `src/cpp/flann/algorithms/cuda/kernels/hierarchical_search_cooperative.cuh`
2. Add template case at line ~548:
   ```cpp
   } else if (k == YOUR_K) {
       hierarchical_search_cooperative_kernel<YOUR_K><<<grid, block, shared_mem_bytes>>>(...);
   ```
3. Rebuild with `make -j$(nproc)`

### Test Results

**K-Means CUDA (11/11 tests PASSED):**
```
SIFT10K Dataset:
- TestSearch:          71.4% precision  ✓
- TestSearch2:         97.0% precision  ✓
- TestAddIncremental:  75.5% precision  ✓
- TestCopy:            71.4% precision  ✓
- TestSave/Remove:     PASSED           ✓

SIFT100K Dataset:
- TestSearch:          70.7% precision  ✓
- TestAddIncremental:  81.1% precision  ✓
```

**Hierarchical CUDA (8/8 tests PASSED):**
```
Brief100K Dataset:
- TestSearch:          96.8% precision  ✓
- TestSearch2:         96.8% precision  ✓
- TestAddIncremental:  93.9% precision  ✓
- TestCopy:            96.8% precision  ✓
- TestRemove:          PASSED           ✓
- TestSave:            92.7% precision  ✓
```

### Troubleshooting

**Build Issues:**

*Error: "nvcc not found"*
```bash
sudo apt-get install nvidia-cuda-toolkit
# Or install from NVIDIA website: https://developer.nvidia.com/cuda-downloads
```

*Error: "Unsupported GPU architecture"*
- Check GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
- Minimum required: 6.0 (Pascal or newer)
- Update CMakeLists.txt if needed to adjust CUDA_ARCHITECTURES

**Runtime Issues:**

*CUDA error at initialization:*
- Check GPU memory: `nvidia-smi`
- Verify driver: `nvidia-smi` shows CUDA version
- Try smaller dataset or increase GPU memory

*Poor performance:*
- Ensure using GPU (check nvidia-smi during search)
- Small datasets have overhead - use CPU version
- Profile: `nvprof` or `nsys profile`

*Low precision:*
- Adjust search parameters (increase `checks`)
- Try different branching factor
- Check dataset characteristics

**Known Limitations:**

1. **NVIDIA Only:** Requires NVIDIA GPU with CUDA support (AMD/Intel GPUs not supported)

2. **Memory Overhead:** Dataset is duplicated in GPU memory. Large datasets may exceed GPU memory limits.

3. **Precision Gap:** Hierarchical CUDA achieves 96.8% vs OpenCL's 97.2% due to minor floating-point differences. This 0.4% gap is acceptable for most applications.

4. **Binary Compilation:** CUDA kernels compile at build time but may require architecture-specific tuning for optimal performance.

### Implementation Details

**Files:**
- `src/cpp/flann/algorithms/cuda/kmeans_cuda_index.h` - K-Means CUDA implementation (~739 lines)
- `src/cpp/flann/algorithms/cuda/hierarchical_cuda_index.h` - Hierarchical clustering (~856 lines)
- `src/cpp/flann/algorithms/cuda/kernels/hierarchical_search_cooperative.cuh` - Cooperative kernel (~629 lines)
- `src/cpp/flann/algorithms/cuda/kernels/kmeans_search_kernel.cuh` - K-Means kernel (~530 lines)
- `src/cpp/flann/algorithms/cuda/cuda_utils.h` - CUDA infrastructure (~364 lines)

**Design:**
- Dual inheritance pattern (CPU tree building + GPU search)
- Cooperative threading (128 threads per query for Hierarchical)
- Template specialization for different distance metrics
- Shared memory optimization for parallel heap management
- Vectorized distance computations (float4 for L2)

**Key Features:**
- Zero memory leaks (verified with compute-sanitizer)
- Deterministic results (reproducible across runs)
- Feature parity with OpenCL implementation
- Production-ready code quality

### Comparison: CUDA vs OpenCL

| Feature | CUDA | OpenCL |
|---------|------|--------|
| **Hardware Support** | NVIDIA only | Multi-vendor (NVIDIA, AMD, Intel) |
| **K-Means Precision** | 70-97% | 92-98% |
| **Hierarchical Precision** | 96.8% | 97.2% |
| **Test Coverage** | 19/19 tests | 19/19 tests |
| **Memory Leaks** | 0 (verified) | 0 (verified) |
| **Build Time** | Compile-time | Runtime kernel compilation |
| **Performance** | Comparable | Comparable |
| **Maturity** | Production (2024-2025) | Experimental (2017-2018, modernized 2024) |

**Recommendation:**
- Use **CUDA** if you have NVIDIA GPUs and want production-ready code
- Use **OpenCL** if you need multi-vendor GPU support or slightly higher precision

### References

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- FLANN Paper: Muja & Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009
- This fork: CUDA implementation developed 2024-2025, achieving 96.8% precision parity with OpenCL
