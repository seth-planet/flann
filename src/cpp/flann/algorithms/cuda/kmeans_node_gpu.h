/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024 (CUDA node structure)
 *
 * THE BSD LICENSE
 *
 * See kmeans_cuda_index.h for full license text.
 *************************************************************************/

#ifndef FLANN_CUDA_KMEANS_NODE_GPU_H_
#define FLANN_CUDA_KMEANS_NODE_GPU_H_

#include <cstdint>

namespace flann {
namespace cuda {

/**
 * @brief GPU-friendly K-Means tree node representation
 *
 * Layout optimized for coalesced memory access (16-byte aligned).
 * Stored in breadth-first order for better cache locality during traversal.
 */
struct alignas(16) KMeansNodeGPU {
    int pivot_index;      // Index into pivots array (dataset index for leaves)
    int child_start;      // First child index in flat array (-1 for leaves)
    uint16_t child_count; // Number of children (0 for leaves)
    uint16_t level;       // Tree depth (0 = root)
    float radius;         // Bounding sphere radius
    float variance;       // Cluster variance (for CB_INDEX heuristic)
};

} // namespace cuda
} // namespace flann

#endif // FLANN_CUDA_KMEANS_NODE_GPU_H_
