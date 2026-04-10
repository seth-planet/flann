# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

FLANN (Fast Library for Approximate Nearest Neighbors) - fast approximate nearest neighbor searches in high dimensional spaces.

**Key Features:** KD-Tree, K-Means, LSH, Hierarchical algorithms | C++/C/Python/MATLAB bindings | CUDA & OpenCL GPU support

## Quick Reference

### Build
```bash
mkdir build && cd build
cmake .. && make -j$(nproc)

# With GPU support
cmake -DBUILD_CUDA_LIB=ON ..     # NVIDIA CUDA
cmake -DBUILD_OpenCL_LIB=ON ..   # OpenCL (multi-vendor)
```

### Test
```bash
make test                              # All tests
./test/flann_kmeans_test               # Specific test
./test/flann_kmeans_cuda_test          # CUDA tests
./test/flann_kmeans_opencl_test        # OpenCL tests
```

### Key Build Options
| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_CUDA_LIB` | OFF | NVIDIA GPU support |
| `BUILD_OpenCL_LIB` | OFF | Multi-vendor GPU support |
| `BUILD_TESTS` | ON | Build test suite |
| `USE_OPENMP` | ON | Multi-threading |

## Code Organization

```
src/cpp/flann/
├── flann.hpp              # Main C++ API (Index template class)
├── algorithms/            # Search algorithm implementations
│   ├── nn_index.h         # Base class for all indices
│   ├── kdtree_index.h     # KD-Tree (low dimensions)
│   ├── kmeans_index.h     # K-Means tree (high dimensions)
│   ├── lsh_index.h        # LSH (very high dimensions)
│   └── cuda/              # CUDA implementations
├── util/
│   ├── matrix.h           # Matrix wrapper (row-major)
│   ├── params.h           # IndexParams, SearchParams
│   └── result_set.h       # Result collection
└── io/                    # HDF5 support

test/                      # GTest-based test suite
examples/                  # Usage examples
```

### Algorithm Selection
| Algorithm | File | Best For |
|-----------|------|----------|
| KD-Tree | `kdtree_index.h` | Low dimensions (<20D) |
| K-Means | `kmeans_index.h` | High dimensions |
| LSH | `lsh_index.h` | Very high dimensions, approximate |
| Linear | `linear_index.h` | Small datasets, exact results |

## Development Patterns

### Index Lifecycle
```cpp
// 1. Create with dataset and params
flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));

// 2. Build search structure
index.buildIndex();

// 3. Search
index.knnSearch(queries, indices, dists, k, flann::SearchParams(128));

// 4. Optional: dynamic updates
index.addPoints(new_points);
index.removePoint(id);
```

### Template Pattern
All algorithms are templated on distance metrics:
```cpp
template<typename Distance>
class NNIndex {
    typedef typename Distance::ElementType ElementType;  // Data type
    typedef typename Distance::ResultType DistanceType;  // Distance type
};
```

### Adding Tests
1. Create `test/flann_mytest.cpp`
2. Use GTest: `#include <gtest/gtest.h>`
3. Register in `test/CMakeLists.txt`: `flann_add_gtest(flann_mytest flann_mytest.cpp)`

## GPU Support

GPU acceleration available for K-Means and Hierarchical indices:

| Backend | Hardware | Guide |
|---------|----------|-------|
| **CUDA** | NVIDIA GPUs | [CUDA_GUIDE.md](CUDA_GUIDE.md) |
| **OpenCL** | NVIDIA/AMD/Intel | [OPENCL_GUIDE.md](OPENCL_GUIDE.md) |

**Quick comparison:**
- CUDA: 89-99% precision, production-ready, NVIDIA only
- OpenCL: 92-98% precision, multi-vendor support

### GPU Index Format (v2.0)

Pre-built GPU indices for fast cold starts:
```bash
# Convert CPU index to GPU v2.0 format
./bin/flann_convert_to_gpu input.db output.gpu.idx --verify

# Load GPU index (4x faster startup than CPU-to-GPU conversion)
flann::Index<L2<float>> index(empty, SavedIndexParams("output.gpu.idx"));
```

### Benchmarking Tools
```bash
# Compare CPU vs GPU loading and search performance
./test/benchmark_cpu_gpu_comparison --all --csv=results.csv

# GPU format statistics
./test/benchmark_gpu_index_stats --dataset=datasets/sift100K.h5
```

## Dependencies

**Required:** C++ compiler, CMake >= 3.20

**Optional:**
- GTest - C++ tests
- HDF5 - Test datasets
- CUDA Toolkit - NVIDIA GPU support
- OpenCL - Multi-vendor GPU support
- OpenMP - Multi-threading

## References

- [FLANN Paper](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_visapp09.pdf) - Muja & Lowe, VISAPP 2009
- [CUDA Guide](CUDA_GUIDE.md) - Detailed CUDA documentation
- [OpenCL Guide](OPENCL_GUIDE.md) - Detailed OpenCL documentation
