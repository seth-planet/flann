/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024 (migrated from OpenCL to CUDA)
 * Copyright 2017  Seth Price (seth@planet.com). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_CUDA_HEAP_UTILS_CUH_
#define FLANN_CUDA_HEAP_UTILS_CUH_

#ifdef FLANN_USE_CUDA

#include <cuda_runtime.h>
#include <float.h>

namespace flann {
namespace cuda {

/**
 * @brief Sift down operation for max-heap
 *
 * Used for k-NN result heap where root contains the largest (worst) distance.
 * Maintains max-heap property after replacing root.
 *
 * Complexity: O(log K), fully unrolled for small K (≤20)
 *
 * @param dists Distance array (heap)
 * @param indices Index array (parallel to dists)
 * @param size Size of heap
 * @param pos Starting position to sift down from
 */
__device__ inline void sift_down_max_heap(
    float* dists,
    int* indices,
    int size,
    int pos
) {
    while (2 * pos + 1 < size) {
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int largest = pos;

        if (dists[left] > dists[largest])
            largest = left;
        if (right < size && dists[right] > dists[largest])
            largest = right;

        if (largest == pos) break;

        // Swap
        float tmp_dist = dists[pos];
        int tmp_idx = indices[pos];
        dists[pos] = dists[largest];
        indices[pos] = indices[largest];
        dists[largest] = tmp_dist;
        indices[largest] = tmp_idx;

        pos = largest;
    }
}

/**
 * @brief Sift up operation for max-heap
 *
 * Used when building initial heap. Bubbles element up until max-heap property restored.
 *
 * @param dists Distance array (heap)
 * @param indices Index array (parallel to dists)
 * @param pos Starting position to sift up from
 */
__device__ inline void sift_up_max_heap(
    float* dists,
    int* indices,
    int pos
) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (dists[parent] >= dists[pos]) break;

        // Swap with parent
        float tmp_dist = dists[pos];
        int tmp_idx = indices[pos];
        dists[pos] = dists[parent];
        indices[pos] = indices[parent];
        dists[parent] = tmp_dist;
        indices[parent] = tmp_idx;

        pos = parent;
    }
}

/**
 * @brief Sift down operation for min-heap
 *
 * Used for priority queue where root contains the smallest (best) distance.
 * Maintains min-heap property after removing root.
 *
 * @param dists Distance array (heap)
 * @param nodes Node ID array (parallel to dists)
 * @param size Size of heap
 * @param pos Starting position to sift down from
 */
__device__ inline void sift_down_min_heap(
    float* dists,
    int* nodes,
    int size,
    int pos
) {
    while (2 * pos + 1 < size) {
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int smallest = pos;

        if (dists[left] < dists[smallest])
            smallest = left;
        if (right < size && dists[right] < dists[smallest])
            smallest = right;

        if (smallest == pos) break;

        // Swap
        float tmp_dist = dists[pos];
        int tmp_node = nodes[pos];
        dists[pos] = dists[smallest];
        nodes[pos] = nodes[smallest];
        dists[smallest] = tmp_dist;
        nodes[smallest] = tmp_node;

        pos = smallest;
    }
}

/**
 * @brief Sift up operation for min-heap
 *
 * Used when inserting into priority queue. Bubbles element up until min-heap property restored.
 *
 * @param dists Distance array (heap)
 * @param nodes Node ID array (parallel to dists)
 * @param pos Starting position to sift up from
 */
__device__ inline void sift_up_min_heap(
    float* dists,
    int* nodes,
    int pos
) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (dists[parent] <= dists[pos]) break;

        // Swap with parent
        float tmp_dist = dists[pos];
        int tmp_node = nodes[pos];
        dists[pos] = dists[parent];
        nodes[pos] = nodes[parent];
        dists[parent] = tmp_dist;
        nodes[parent] = tmp_node;

        pos = parent;
    }
}

/**
 * @brief Initialize k-NN result heap with worst possible values
 *
 * Heap is a max-heap where root contains the k-th nearest neighbor (worst of k best).
 *
 * @param dists Distance array to initialize
 * @param indices Index array to initialize
 * @param k Number of nearest neighbors
 */
template<int K>
__device__ inline void init_result_heap(
    float* dists,
    int* indices
) {
    #pragma unroll
    for (int i = 0; i < K; i++) {
        dists[i] = FLT_MAX;  // Worst possible distance
        indices[i] = -1;     // Invalid index
    }
}

/**
 * @brief Insert candidate into k-NN result heap if better than k-th neighbor
 *
 * Uses max-heap property: root contains largest distance among k neighbors.
 * If new distance is better (smaller), replace root and sift down.
 *
 * @param dists Distance heap (max-heap)
 * @param indices Index array (parallel to dists)
 * @param k Number of nearest neighbors
 * @param new_dist Distance of new candidate
 * @param new_idx Index of new candidate
 * @return true if candidate was inserted, false otherwise
 */
template<int K>
__device__ inline bool insert_into_heap(
    float* dists,
    int* indices,
    float new_dist,
    int new_idx
) {
    if (new_dist < dists[0]) {
        // Replace root with new candidate
        dists[0] = new_dist;
        indices[0] = new_idx;

        // Sift down to maintain max-heap property
        sift_down_max_heap(dists, indices, K, 0);
        return true;
    }
    return false;
}

// ============================================================================
// Integer versions for Hamming distance (Hierarchical Clustering)
// ============================================================================

/**
 * @brief Sift down operation for min-heap (integer distances)
 */
__device__ inline void sift_down_min_heap_int(
    int* dists,
    int* nodes,
    int size,
    int pos
) {
    while (2 * pos + 1 < size) {
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int smallest = pos;

        if (dists[left] < dists[smallest])
            smallest = left;
        if (right < size && dists[right] < dists[smallest])
            smallest = right;

        if (smallest == pos) break;

        // Swap
        int tmp_dist = dists[pos];
        int tmp_node = nodes[pos];
        dists[pos] = dists[smallest];
        nodes[pos] = nodes[smallest];
        dists[smallest] = tmp_dist;
        nodes[smallest] = tmp_node;

        pos = smallest;
    }
}

/**
 * @brief Sift up operation for min-heap (integer distances)
 */
__device__ inline void sift_up_min_heap_int(
    int* dists,
    int* nodes,
    int pos
) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (dists[parent] <= dists[pos]) break;

        // Swap with parent
        int tmp_dist = dists[pos];
        int tmp_node = nodes[pos];
        dists[pos] = dists[parent];
        nodes[pos] = nodes[parent];
        dists[parent] = tmp_dist;
        nodes[parent] = tmp_node;

        pos = parent;
    }
}

/**
 * @brief Pop minimum from min-heap (integer distances)
 */
__device__ inline void pop_min_heap_int(
    int* dists,
    int* nodes,
    int& size
) {
    if (size <= 0) return;

    // Move last element to root
    dists[0] = dists[size - 1];
    nodes[0] = nodes[size - 1];

    // Sift down from root
    sift_down_min_heap_int(dists, nodes, size - 1, 0);
}

/**
 * @brief Sift up operation for max-heap (integer distances)
 */
__device__ inline void sift_up_max_heap_int(
    int* dists,
    int* nodes,
    int pos
) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (dists[parent] >= dists[pos]) break;

        // Swap with parent
        int tmp_dist = dists[pos];
        int tmp_node = nodes[pos];
        dists[pos] = dists[parent];
        nodes[pos] = nodes[parent];
        dists[parent] = tmp_dist;
        nodes[parent] = tmp_node;

        pos = parent;
    }
}

/**
 * @brief Sift down operation for max-heap (integer distances)
 */
__device__ inline void sift_down_max_heap_int(
    int* dists,
    int* indices,
    int size,
    int pos
) {
    while (2 * pos + 1 < size) {
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int largest = pos;

        if (dists[left] > dists[largest])
            largest = left;
        if (right < size && dists[right] > dists[largest])
            largest = right;

        if (largest == pos) break;

        // Swap
        int tmp_dist = dists[pos];
        int tmp_idx = indices[pos];
        dists[pos] = dists[largest];
        indices[pos] = indices[largest];
        dists[largest] = tmp_dist;
        indices[largest] = tmp_idx;

        pos = largest;
    }
}

/**
 * @brief Initialize k-NN result heap with worst possible values (integer distances)
 */
template<int K>
__device__ inline void init_result_heap_int(
    int* dists,
    int* indices
) {
    #pragma unroll
    for (int i = 0; i < K; i++) {
        dists[i] = INT_MAX;  // Worst possible Hamming distance
        indices[i] = -1;     // Invalid index
    }
}

/**
 * @brief Insert candidate into k-NN result heap if better than k-th neighbor (integer distances)
 */
template<int K>
__device__ inline bool insert_into_heap_int(
    int* dists,
    int* indices,
    int new_dist,
    int new_idx
) {
    if (new_dist < dists[0]) {
        // Replace root with new candidate
        dists[0] = new_dist;
        indices[0] = new_idx;

        // Sift down to maintain max-heap property
        sift_down_max_heap_int(dists, indices, K, 0);
        return true;
    }
    return false;
}

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_CUDA_HEAP_UTILS_CUH_
