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

#ifndef FLANN_CUDA_KMEANS_SEARCH_KERNEL_CUH_
#define FLANN_CUDA_KMEANS_SEARCH_KERNEL_CUH_

#ifdef FLANN_USE_CUDA

#include <cuda_runtime.h>
#include <float.h>
#include "../kmeans_node_gpu.h"
#include "distance_kernels.cuh"
#include "heap_utils.cuh"

namespace flann {
namespace cuda {

/**
 * @brief Check if node is too far away to contain better neighbors
 *
 * Uses triangle inequality: if ||q - center||² - radius² > worst_dist²,
 * and the discriminant is positive, then the node cannot improve results.
 *
 * @param node_pivots Flat array of node pivot coordinates
 * @param node Tree node metadata
 * @param query Query vector
 * @param worst_dist Distance to k-th nearest neighbor (worst of k best)
 * @param dim Vector dimension
 * @return true if node can be pruned
 */
__device__ inline bool is_too_far(
    const float* __restrict__ node_pivots,
    const KMeansNodeGPU& node,
    const float* __restrict__ query,
    float worst_dist,
    int dim
) {
    // Compute distance to node pivot
    float bsq = compute_l2_distance(query, node_pivots + node.pivot_index * dim, dim);

    // Node radius squared
    float rsq = node.radius * node.radius;

    // Worst distance squared (k-th neighbor)
    float wsq = worst_dist;

    // Triangle inequality test
    float val = bsq - rsq - wsq;
    if (val > 0.0f) {
        float discriminant = val * val - 4.0f * rsq * wsq;
        return discriminant > 0.0f;
    }
    return false;
}

/**
 * @brief Explore node branches, find closest child and add rest to priority queue
 *
 * For an internal node, computes distances to all children, finds the closest,
 * and pushes the remaining children onto the priority queue for later exploration.
 * Uses CB_INDEX variance adjustment to prioritize tighter clusters.
 *
 * @param tree_nodes Flat array of tree nodes
 * @param tree_pivots Flat array of pivot coordinates
 * @param node_index Index of current node
 * @param query Query vector
 * @param pq_dists Priority queue distances (min-heap)
 * @param pq_nodes Priority queue node indices
 * @param pq_size Current PQ size
 * @param dim Vector dimension
 * @param max_pq_size Maximum PQ size
 * @param cb_index CB_INDEX parameter for variance adjustment
 * @return Index of closest child node
 */
__device__ inline int explore_node_branches(
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const float* __restrict__ tree_pivots,
    int node_index,
    const float* __restrict__ query,
    float* __restrict__ pq_dists,
    int* __restrict__ pq_nodes,
    int& pq_size,
    int dim,
    int max_pq_size,
    float cb_index
) {
    const KMeansNodeGPU& node = tree_nodes[node_index];
    int child_start = node.child_start;
    int child_count = node.child_count;

    // Compute distances to all children
    float dist_arr[64];  // Max branching factor
    int best_idx = 0;
    float best_dist = FLT_MAX;

    for (int i = 0; i < child_count; ++i) {
        int child_idx = child_start + i;
        // Get the child node to access its pivot_index field
        const KMeansNodeGPU& child_node = tree_nodes[child_idx];
        // Use the node's pivot_index (NOT the node index itself) to access pivots
        const float* child_pivot = tree_pivots + child_node.pivot_index * dim;
        float dist = compute_l2_distance(query, child_pivot, dim);
        dist_arr[i] = dist;

        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    // Add non-best children to priority queue (min-heap)
    for (int i = 0; i < child_count; ++i) {
        if (i == best_idx) continue;

        int child_idx = child_start + i;
        const KMeansNodeGPU& child_node = tree_nodes[child_idx];

        // Apply CB_INDEX variance adjustment to prioritize tighter clusters
        // This matches OpenCL implementation (nn_opencl_index.h:1247)
        float dist = dist_arr[i];
        float adjusted_dist = dist - cb_index * child_node.variance;

        // Insert into PQ if not full, or if better than worst
        if (pq_size < max_pq_size) {
            pq_dists[pq_size] = adjusted_dist;
            pq_nodes[pq_size] = child_idx;
            sift_up_min_heap(pq_dists, pq_nodes, pq_size);
            pq_size++;
        } else if (adjusted_dist < pq_dists[pq_size - 1]) {
            // Replace worst element
            pq_dists[pq_size - 1] = adjusted_dist;
            pq_nodes[pq_size - 1] = child_idx;
            // Restore min-heap property (sift up from end)
            int pos = pq_size - 1;
            while (pos > 0) {
                int parent = (pos - 1) / 2;
                if (pq_dists[parent] <= pq_dists[pos]) break;

                float tmp_dist = pq_dists[pos];
                int tmp_node = pq_nodes[pos];
                pq_dists[pos] = pq_dists[parent];
                pq_nodes[pos] = pq_nodes[parent];
                pq_dists[parent] = tmp_dist;
                pq_nodes[parent] = tmp_node;

                pos = parent;
            }
        }
    }

    // Return closest child for immediate exploration
    return child_start + best_idx;
}

/**
 * @brief Process a single node: compute distances for leaves or explore children
 *
 * If node is a leaf, computes distances to all dataset points it contains
 * and inserts them into the k-NN result heap. If node is internal, explores
 * its children.
 *
 * @param tree_nodes Flat array of tree nodes
 * @param tree_pivots Flat array of pivot coordinates
 * @param dataset Full dataset
 * @param node_index Index of node to process
 * @param query Query vector
 * @param result_dists k-NN result heap (max-heap)
 * @param result_indices k-NN result indices
 * @param checks Counter for number of distance computations
 * @param pq_dists Priority queue distances
 * @param pq_nodes Priority queue node indices
 * @param pq_size Current PQ size
 * @param dim Vector dimension
 * @param k Number of nearest neighbors
 * @param max_pq_size Maximum PQ size
 * @return Next node to explore (-1 if leaf)
 */
template<int K>
__device__ inline int find_nearest_neighbor(
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const float* __restrict__ tree_pivots,
    const int* __restrict__ dataset_indices,
    const float* __restrict__ dataset,
    int node_index,
    const float* __restrict__ query,
    float* __restrict__ result_dists,
    int* __restrict__ result_indices,
    int& checks,
    int max_checks,
    float* __restrict__ pq_dists,
    int* __restrict__ pq_nodes,
    int& pq_size,
    int dim,
    int max_pq_size,
    float cb_index
) {
    const KMeansNodeGPU& node = tree_nodes[node_index];

    // Check if node is too far (prune)
    if (is_too_far(tree_pivots, node, query, result_dists[0], dim)) {
        return -1;
    }

    // Leaf node check: negative child_start indicates leaf
    // For leaves, child_start = -(offset + 1) where offset is into dataset_indices
    if (node.child_start < 0) {
        // This is a leaf node - decode offset and iterate through all dataset points
        int leaf_start = (-node.child_start) - 1;  // Decode: -(offset + 1) -> offset
        int leaf_size = node.child_count;

        // Process all points in this leaf
        for (int i = 0; i < leaf_size && checks < max_checks; ++i) {
            int dataset_idx = dataset_indices[leaf_start + i];
            const float* point = dataset + dataset_idx * dim;

            float dist = compute_l2_distance(query, point, dim);
            checks++;

            // Insert into k-NN heap if better than k-th neighbor
            insert_into_heap<K>(result_dists, result_indices, dist, dataset_idx);
        }

        return -1;  // Leaf processed, no children to explore
    }

    // Internal node: explore branches
    return explore_node_branches(
        tree_nodes, tree_pivots, node_index, query,
        pq_dists, pq_nodes, pq_size, dim, max_pq_size,
        cb_index
    );
}

/**
 * @brief Main K-Means search kernel (thread-per-query design)
 *
 * Each thread processes one query using best-first search through the K-Means tree.
 * Maintains a priority queue (min-heap) for nodes to explore and a result heap
 * (max-heap) for k-nearest neighbors.
 *
 * Algorithm:
 * 1. Initialize result heap with FLT_MAX
 * 2. Start from root node (index 0)
 * 3. Best-first search:
 *    - Explore current node (find_nearest_neighbor)
 *    - If leaf: compute distances, update results
 *    - If internal: explore_node_branches, get closest child
 *    - When branch ends: pop next node from priority queue
 * 4. Continue until checks >= max_checks or PQ empty
 * 5. Result heap already sorted (max-heap property)
 *
 * @tparam K Number of nearest neighbors
 * @tparam MAX_CHECKS Maximum distance computations per query
 * @param dataset Full dataset [N x dim]
 * @param queries Query vectors [num_queries x dim]
 * @param tree_nodes Flat array of tree nodes (breadth-first)
 * @param tree_pivots Flat array of pivot coordinates [num_nodes x dim]
 * @param result_indices Output k-NN indices [num_queries x K]
 * @param result_dists Output k-NN distances [num_queries x K]
 * @param num_queries Number of queries
 * @param dim Vector dimension
 * @param num_nodes Total number of tree nodes
 */
template<int K, int MAX_CHECKS>
__global__ void kmeans_search_kernel(
    const float* __restrict__ dataset,
    const float* __restrict__ queries,
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const float* __restrict__ tree_pivots,
    const int* __restrict__ dataset_indices,
    int* __restrict__ result_indices,
    float* __restrict__ result_distances,
    int num_queries,
    int dim,
    int num_nodes,
    float cb_index
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (gid >= num_queries) return;

    // Query vector for this thread
    const float* query = queries + gid * dim;

    // Per-thread result heap (k-NN, max-heap)
    float result_dists[K];
    int result_ids[K];
    init_result_heap<K>(result_dists, result_ids);

    // Per-thread priority queue (nodes to explore, min-heap)
    const int MAX_PQ_SIZE = MAX_CHECKS;
    float pq_dists[MAX_PQ_SIZE];
    int pq_nodes[MAX_PQ_SIZE];
    int pq_size = 0;

    int checks = 0;
    int next_node = 0;  // Start from root

    // Best-first search
    while (next_node != -1 && checks < MAX_CHECKS) {
        // Process current node
        next_node = find_nearest_neighbor<K>(
            tree_nodes, tree_pivots, dataset_indices, dataset,
            next_node, query,
            result_dists, result_ids,
            checks, MAX_CHECKS,
            pq_dists, pq_nodes, pq_size,
            dim, MAX_PQ_SIZE,
            cb_index
        );

        // If branch ended, pop next node from priority queue
        if (next_node == -1 && pq_size > 0) {
            // Pop minimum from priority queue
            next_node = pq_nodes[0];

            // Remove root: move last element to root, sift down
            pq_nodes[0] = pq_nodes[pq_size - 1];
            pq_dists[0] = pq_dists[pq_size - 1];
            pq_size--;

            if (pq_size > 0) {
                sift_down_min_heap(pq_dists, pq_nodes, pq_size, 0);
            }
        }
    }

    // Sort results by distance (ascending)
    // The max-heap contains the k smallest distances but NOT in sorted order
    for (int i = 0; i < K - 1; ++i) {
        for (int j = i + 1; j < K; ++j) {
            if (result_dists[j] < result_dists[i]) {
                // Swap distances
                float tmp_dist = result_dists[i];
                result_dists[i] = result_dists[j];
                result_dists[j] = tmp_dist;

                // Swap indices
                int tmp_idx = result_ids[i];
                result_ids[i] = result_ids[j];
                result_ids[j] = tmp_idx;
            }
        }
    }

    // Write sorted results to global memory
    for (int i = 0; i < K; ++i) {
        result_indices[gid * K + i] = result_ids[i];
        result_distances[gid * K + i] = result_dists[i];
    }
}

/**
 * @brief Kernel dispatcher with runtime parameter selection
 *
 * Dispatches to appropriate template-specialized kernel based on k value.
 * Supports k ∈ {1, 5, 10, 20, 50, 100}.
 *
 * @param dataset Full dataset
 * @param queries Query vectors
 * @param tree_nodes Tree node array
 * @param tree_pivots Pivot array
 * @param result_indices Output indices
 * @param result_distances Output distances
 * @param num_queries Number of queries
 * @param dim Vector dimension
 * @param num_nodes Number of tree nodes
 * @param k Number of nearest neighbors
 * @param max_checks Maximum distance computations
 * @param grid Grid dimensions
 * @param block Block dimensions
 * @return true if kernel launched successfully, false if unsupported k
 */
bool launch_kmeans_search(
    const float* dataset,
    const float* queries,
    const KMeansNodeGPU* tree_nodes,
    const float* tree_pivots,
    const int* dataset_indices,
    int* result_indices,
    float* result_distances,
    size_t num_queries,
    size_t dim,
    size_t num_nodes,
    int k,
    int max_checks,
    dim3 grid,
    dim3 block,
    float cb_index
) {
    // Dispatch based on k and max_checks
    // Common configurations: k ∈ {1, 5, 10, 20, 50, 100}, max_checks ∈ {32, 64, 128, 256}

    if (k == 1) {
        if (max_checks <= 32) {
            kmeans_search_kernel<1, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 64) {
            kmeans_search_kernel<1, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 128) {
            kmeans_search_kernel<1, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else {
            kmeans_search_kernel<1, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        }
    } else if (k == 5) {
        if (max_checks <= 32) {
            kmeans_search_kernel<5, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 64) {
            kmeans_search_kernel<5, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 128) {
            kmeans_search_kernel<5, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else {
            kmeans_search_kernel<5, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        }
    } else if (k == 10) {
        if (max_checks <= 32) {
            kmeans_search_kernel<10, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 64) {
            kmeans_search_kernel<10, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 128) {
            kmeans_search_kernel<10, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else {
            kmeans_search_kernel<10, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        }
    } else if (k == 20) {
        if (max_checks <= 32) {
            kmeans_search_kernel<20, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 64) {
            kmeans_search_kernel<20, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else if (max_checks <= 128) {
            kmeans_search_kernel<20, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else {
            kmeans_search_kernel<20, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        }
    } else if (k == 50) {
        if (max_checks <= 128) {
            kmeans_search_kernel<50, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else {
            kmeans_search_kernel<50, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        }
    } else if (k == 100) {
        if (max_checks <= 128) {
            kmeans_search_kernel<100, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        } else {
            kmeans_search_kernel<100, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices,
                result_indices, result_distances,
                num_queries, dim, num_nodes, cb_index);
        }
    } else {
        // Unsupported k value
        return false;
    }

    return true;
}

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_CUDA_KMEANS_SEARCH_KERNEL_CUH_
