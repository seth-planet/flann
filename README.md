FLANN - Fast Library for Approximate Nearest Neighbors
======================================================

FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. It contains a collection of algorithms we found to work best for nearest neighbor search and a system for automatically choosing the best algorithm and optimum parameters depending on the dataset.

FLANN is written in C++ and contains bindings for the following languages: C, MATLAB, Python, and Ruby.

> **GPU Acceleration:** This fork includes GPU-accelerated implementations for K-Means and Hierarchical clustering:
> - **CUDA** (NVIDIA GPUs): Production-ready, 97-99% precision, ~10µs/query. See [CUDA_GUIDE.md](CUDA_GUIDE.md) for details.
> - **OpenCL** (Multi-vendor): Experimental, supports NVIDIA/AMD/Intel GPUs. See [OPENCL_GUIDE.md](OPENCL_GUIDE.md) for details.


Quick Start
-----------

### C++

```cpp
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

int main()
{
    int nn = 3;
    flann::Matrix<float> dataset;
    flann::Matrix<float> query;
    flann::load_from_file(dataset, "dataset.hdf5", "dataset");
    flann::load_from_file(query, "dataset.hdf5", "query");

    flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
    flann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

    // Construct a randomized kd-tree index using 4 kd-trees
    flann::Index<flann::L2<float>> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();

    // Do a knn search, using 128 checks
    index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

    delete[] dataset.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
    return 0;
}
```

### Python

```python
from pyflann import *
from numpy import *
from numpy.random import *

dataset = rand(10000, 128)
testset = rand(1000, 128)

flann = FLANN()
result, dists = flann.nn(dataset, testset, 5, algorithm="kmeans",
                         branching=32, iterations=7, checks=16)
```


Building
--------

FLANN uses CMake as its build system.

```bash
# Linux
mkdir build && cd build
cmake ..
make

# Windows
mkdir build && cd build
cmake ..
nmake
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_C_BINDINGS` | ON | Build C bindings |
| `BUILD_PYTHON_BINDINGS` | ON | Build Python bindings |
| `BUILD_MATLAB_BINDINGS` | ON | Build MATLAB bindings |
| `BUILD_CUDA_LIB` | OFF | Build CUDA GPU library |
| `BUILD_OpenCL_LIB` | OFF | Build OpenCL GPU library |
| `BUILD_EXAMPLES` | ON | Build examples |
| `BUILD_TESTS` | ON | Build tests |
| `USE_OPENMP` | ON | Enable OpenMP multi-threading |

For GPU support, see [CUDA_GUIDE.md](CUDA_GUIDE.md) or [OPENCL_GUIDE.md](OPENCL_GUIDE.md).


Algorithms
----------

FLANN supports multiple index types, each suited for different scenarios:

| Algorithm | Best For | Key Parameters |
|-----------|----------|----------------|
| **Linear** | Exact search, small datasets | None |
| **KDTree** | Low dimensions (<20D) | `trees` (1-16) |
| **KMeans** | High dimensions | `branching`, `iterations` |
| **Hierarchical** | Binary features, clustering | `branching`, `trees` |
| **LSH** | Very high dimensions, binary | `table_number`, `key_size` |
| **Composite** | KDTree + KMeans combined | Both parameters |
| **Autotuned** | Unknown data characteristics | `target_precision` |

### Autotuned Index

For automatic parameter selection:

```cpp
flann::Index<flann::L2<float>> index(dataset,
    flann::AutotunedIndexParams(
        0.9,    // target_precision (90%)
        0.01,   // build_weight
        0,      // memory_weight
        0.1     // sample_fraction
    )
);
```


Search Parameters
-----------------

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checks` | int | 32 | Max leaf nodes to visit (-1 = unlimited) |
| `eps` | float | 0 | Epsilon for approximate search |
| `sorted` | bool | true | Sort results by distance |
| `max_neighbors` | int | -1 | Max neighbors for radius search |
| `cores` | int | 0 | OpenMP threads (0 = auto) |
| `use_heap` | bool | auto | Use heap for large k values |


Distance Metrics
----------------

| Distance | Type | Use Case |
|----------|------|----------|
| `L2<T>` | Euclidean | General purpose (default) |
| `L1<T>` | Manhattan | Sparse data |
| `MinkowskiDistance<T>` | Generalized | Custom p-norm |
| `Hamming<T>` | Hamming | Binary features (ORB, BRIEF) |
| `HellingerDistance<T>` | Hellinger | Histograms |
| `ChiSquareDistance<T>` | Chi-Square | Histograms |
| `KL_Divergence<T>` | KL | Probability distributions |


Language Bindings
-----------------

- **C++**: Include `<flann/flann.hpp>` - full API access
- **C**: Include `<flann/flann.h>` - functions for float, double, int, unsigned char
- **Python**: `from pyflann import *` - requires numpy
- **MATLAB**: `flann_build_index`, `flann_search`, `flann_free_index`

See `examples/` directory for complete examples in each language.


GPU Acceleration
----------------

For large datasets (>50K points) and high dimensions (>64D), GPU acceleration provides 5-20x speedup:

- **CUDA** (NVIDIA GPUs): [CUDA_GUIDE.md](CUDA_GUIDE.md) - Production-ready, 97-99% precision
- **OpenCL** (Multi-vendor): [OPENCL_GUIDE.md](OPENCL_GUIDE.md) - Experimental, AMD/Intel/NVIDIA


Documentation
-------------

- **User Manual**: `doc/manual.pdf` - Complete API reference
- **Web Page**: [http://www.cs.ubc.ca/research/flann](http://www.cs.ubc.ca/research/flann)
- **Paper**: Marius Muja and David G. Lowe, "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration", VISAPP 2009 [(PDF)](https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_visapp09.pdf)


Getting FLANN
-------------

Clone the repository: `git clone git://github.com/mariusmuja/flann.git`

Browse the source: [https://github.com/mariusmuja/flann](https://github.com/mariusmuja/flann)


License
-------

FLANN is distributed under the terms of the [BSD License](https://github.com/mariusmuja/flann/blob/master/COPYING).


Bug Reporting
-------------

Please report bugs or feature requests using [GitHub Issues](http://github.com/mariusmuja/flann/issues).
