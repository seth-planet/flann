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
- Python - For Python bindings
- MATLAB - For MATLAB bindings
