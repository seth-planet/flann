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

#ifndef FLANN_CUDA_HIERARCHICAL_SEARCH_KERNEL_CUH_
#define FLANN_CUDA_HIERARCHICAL_SEARCH_KERNEL_CUH_

#ifdef FLANN_USE_CUDA

#include <cuda_runtime.h>
#include <limits.h>
#include <cstdio>
#include "../kmeans_node_gpu.h"
#include "distance_kernels.cuh"
#include "heap_utils.cuh"

namespace flann {
namespace cuda {

/**
 * @brief Check if node is too far away to contain better neighbors (Hamming distance)
 *
 * For binary descriptors with Hamming distance, we use a simpler bound check
 * since we don't have radius information like in K-Means.
 *
 * @param node_pivots Flat array of node pivot descriptors
 * @param node Tree node metadata
 * @param query Query descriptor
 * @param worst_dist Hamming distance to k-th nearest neighbor
 * @param bytes Descriptor length in bytes
 * @return true if node can be pruned (always false for hierarchical - no radius)
 */
__device__ inline bool is_too_far_hamming(
    const unsigned char* __restrict__ node_pivots,
    const KMeansNodeGPU& node,
    const unsigned char* __restrict__ query,
    int worst_dist,
    int bytes
) {
    // For hierarchical clustering with Hamming distance, we don't have radius bounds
    // So we cannot prune nodes early - always return false
    return false;
}

/**
 * @brief Explore virtual root node branches (IMPLICIT roots - no node structure)
 *
 * CRITICAL: Roots are implicit (not stored in tree_nodes array). This function directly
 * accesses pivots at tree*branching offset without reading any node structure.
 *
 * This is the key difference from explore_node_branches_hamming which reads node
 * structures. Roots don't have node structures, so we compute distances directly
 * from pivot array using tree_id * branching as the starting offset.
 *
 * @param tree_pivots Flat array of pivot binary descriptors
 * @param tree_id Tree identifier (0, 1, 2, 3, ...)
 * @param branching Branching factor (typically 32)
 * @param query Query binary descriptor
 * @param pq_dists Priority queue distances (min-heap)
 * @param pq_nodes Priority queue node indices
 * @param pq_size Current PQ size
 * @param padded_bytes Descriptor stride (padded length)
 * @param actual_bytes Actual descriptor length for distance computation
 * @param max_pq_size Max PQ capacity
 * @return Index of closest child node
 */
__device__ inline int explore_root_branches_hamming(
    const unsigned char* __restrict__ tree_pivots,
    const int* __restrict__ device_node_index,  // CRITICAL FIX: nodeIndex indirection
    int tree_id,
    int branching,
    const unsigned char* __restrict__ query,
    int* __restrict__ pq_dists,
    int* __restrict__ pq_nodes,
    int& pq_size,
    int padded_bytes,
    int actual_bytes,
    int max_pq_size
) {
    // CRITICAL FIX: Roots are IMPLICIT (not stored in arrays)
    // First-level children occupy pre-reserved slots: tree_id * branching
    // Tree 0 → slots 0-31, Tree 1 → slots 32-63, etc. (matches OpenCL)
    int child_start = tree_id * branching;

    // Debug: Log root exploration for first thread/block
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[ROOT DEBUG] Exploring tree %d root (implicit): children at pivot indices %d-%d\n",
               tree_id, child_start, child_start + branching - 1);
        // Show first child's pivot bytes
        const unsigned char* first_pivot = tree_pivots + child_start * padded_bytes;
        printf("[ROOT DEBUG]   First child pivot[%d] bytes: %02x %02x %02x %02x\n",
               child_start, first_pivot[0], first_pivot[1], first_pivot[2], first_pivot[3]);
    }

    // Compute distances to all root's children
    int dist_arr[64];  // Max branching factor
    int best_idx = 0;
    int best_dist = INT_MAX;

    for (int i = 0; i < branching; ++i) {
        int pivot_idx = child_start + i;
        const unsigned char* child_pivot = tree_pivots + pivot_idx * padded_bytes;
        int dist = compute_hamming_distance(query, child_pivot, actual_bytes);
        dist_arr[i] = dist;

        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    // Add non-best children to priority queue
    for (int i = 0; i < branching; ++i) {
        if (i == best_idx) continue;

        int child_node_idx = child_start + i;  // Node index = pivot index for first level
        int dist = dist_arr[i];

        if (pq_size < max_pq_size) {
            pq_dists[pq_size] = dist;
            pq_nodes[pq_size] = child_node_idx;  // REVERT: Direct index (identity mapping makes this equivalent)
            sift_up_min_heap_int(pq_dists, pq_nodes, pq_size);
            pq_size++;
        }
    }

    // ========================================================================
    // PHASE 2: Root Exploration Verification (Debug - Query 0 only)
    // ========================================================================
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[PHASE 2 ROOT] Tree %d exploration:\n", tree_id);
        printf("  Child range: pivot indices %d-%d\n", child_start, child_start + branching - 1);
        printf("  First 8 children distances:\n");
        for (int i = 0; i < branching && i < 8; ++i) {
            int pivot_idx = child_start + i;
            const unsigned char* pivot = tree_pivots + pivot_idx * padded_bytes;
            printf("    Child %d (node %d): dist=%d, pivot[0-3]=%02x %02x %02x %02x\n",
                   i, pivot_idx, dist_arr[i],
                   pivot[0], pivot[1], pivot[2], pivot[3]);
        }
        printf("  Best child: index %d (node %d), dist=%d\n\n",
               best_idx, child_start + best_idx, best_dist);
    }
    // ========================================================================

    // Return closest child node index (REVERT: Direct index)
    return child_start + best_idx;
}

/**
 * @brief Explore node branches, find closest child and add rest to priority queue
 *
 * For an internal node, computes Hamming distances to all children, finds the closest,
 * and pushes the remaining children onto the priority queue for later exploration.
 *
 * @param tree_nodes Flat array of tree nodes
 * @param tree_pivots Flat array of pivot binary descriptors
 * @param node_index Index of current node
 * @param query Query binary descriptor
 * @param pq_dists Priority queue Hamming distances (min-heap)
 * @param pq_nodes Priority queue node indices (parallel to pq_dists)
 * @param pq_size Current PQ size
 * @param bytes Descriptor length in bytes
 * @return Index of closest child node
 */
__device__ inline int explore_node_branches_hamming(
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const unsigned char* __restrict__ tree_pivots,
    const int* __restrict__ device_node_index,  // CRITICAL FIX: nodeIndex indirection
    int node_index,
    const unsigned char* __restrict__ query,
    int* __restrict__ pq_dists,
    int* __restrict__ pq_nodes,
    int& pq_size,
    int padded_bytes,
    int actual_bytes,
    int max_pq_size
) {
    const KMeansNodeGPU& node = tree_nodes[node_index];
    // CRITICAL FIX: Use hybrid array for parent traversal (matching OpenCL algorithm)
    // OpenCL dereferences nodeIndex before exploring: nodePtr = nodeIndexArr[nodeId]
    int child_start = device_node_index[node_index];  // Read from hybrid array, NOT struct
    int child_count = node.child_count;

    // Debug: Log pivot access for first thread exploring roots
    if (threadIdx.x == 0 && blockIdx.x == 0 && (node_index <= 3)) {
        printf("[PIVOT DEBUG] Exploring node %d: child_start=%d, child_count=%d\n",
               node_index, child_start, child_count);
        if (child_count > 0) {
            printf("[PIVOT DEBUG]   Will access pivots at indices %d-%d\n",
                   child_start, child_start + child_count - 1);
            // Show first pivot's first 4 bytes
            const unsigned char* first_pivot = tree_pivots + child_start * padded_bytes;
            printf("[PIVOT DEBUG]   First child pivot[%d] bytes: %02x %02x %02x %02x\n",
                   child_start, first_pivot[0], first_pivot[1], first_pivot[2], first_pivot[3]);
        }
    }

    // Compute distances to all children
    int dist_arr[64];  // Max branching factor
    int best_idx = 0;
    int best_dist = INT_MAX;

    for (int i = 0; i < child_count; ++i) {
        int child_idx = child_start + i;
        // Access pivot using node array index (NOT pivot_index field which is dataset index)
        const unsigned char* child_pivot = tree_pivots + child_idx * padded_bytes;
        int dist = compute_hamming_distance(query, child_pivot, actual_bytes);
        dist_arr[i] = dist;

        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    // Add non-best children to priority queue (min-heap for best-first search)
    // Min-heap property: root (index 0) contains smallest distance = best unexplored node
    for (int i = 0; i < child_count; ++i) {
        if (i == best_idx) continue;

        int child_idx = child_start + i;
        int dist = dist_arr[i];

        // Add all non-best children to PQ (no bounding - let max_checks limit exploration)
        if (pq_size < max_pq_size) {
            pq_dists[pq_size] = dist;
            pq_nodes[pq_size] = child_idx;  // REVERT: Direct index (identity mapping makes this equivalent)
            sift_up_min_heap_int(pq_dists, pq_nodes, pq_size);
            pq_size++;
        }
        // If PQ full, we've hit the max_checks limit organically, stop adding
    }

    // Return closest child for immediate exploration (REVERT: Direct index)
    return child_start + best_idx;
}

/**
 * @brief Process a single node: compute distances for leaves or explore children
 *
 * If node is a leaf, computes Hamming distances to all binary descriptors it contains
 * and inserts them into the k-NN result heap. If node is internal, explores
 * its children.
 *
 * **CRITICAL**: Uses duplicate detection to avoid examining the same dataset point
 * multiple times across different trees/branches (matches CPU implementation).
 *
 * @param tree_nodes Flat array of tree nodes
 * @param tree_pivots Flat array of pivot binary descriptors
 * @param dataset_indices Leaf node point indices into dataset
 * @param dataset Full dataset of binary descriptors
 * @param node_index Current node index
 * @param query Query binary descriptor
 * @param result_dists k-NN result heap (Hamming distances, max-heap)
 * @param result_indices k-NN result indices
 * @param pq_dists Priority queue distances
 * @param pq_nodes Priority queue node indices
 * @param pq_size Current PQ size
 * @param checks Number of distance computations so far
 * @param bytes Descriptor length in bytes
 * @param max_checks Maximum distance computations allowed
 * @param max_pq_size Maximum priority queue size
 * @param checked_indices Array tracking visited dataset indices (duplicate detection)
 * @param num_checked Count of checked indices
 * @return Index of next node to explore (-1 if leaf)
 */
template<int K>
__device__ inline int find_nearest_neighbor_hamming(
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const unsigned char* __restrict__ tree_pivots,
    const int* __restrict__ dataset_indices,
    const int* __restrict__ device_node_index,  // CRITICAL FIX: nodeIndex indirection
    const unsigned char* __restrict__ dataset,
    int node_index,
    const unsigned char* __restrict__ query,
    int* __restrict__ result_dists,
    int* __restrict__ result_indices,
    int* __restrict__ pq_dists,
    int* __restrict__ pq_nodes,
    int& pq_size,
    int& checks,
    int padded_bytes,
    int actual_bytes,
    int max_checks,
    int max_pq_size,
    int* __restrict__ checked_indices,
    int& num_checked
) {
    const KMeansNodeGPU& node = tree_nodes[node_index];

    // Leaf node check: child_start < 0 encodes -(offset + 1)
    if (node.child_start < 0) {
        // CRITICAL FIX: Use hybrid array with double-indirection (OpenCL-style)
        // Indirection 1: Get pointer to leaf data region in hybrid array
        int leaf_ptr = device_node_index[node_index];

        // Indirection 2: Read leaf count from first element of leaf data region
        int leaf_size = device_node_index[leaf_ptr];

        // Debug: Log first leaf access for query 0
        if (threadIdx.x == 0 && blockIdx.x == 0 && num_checked == 0) {
            printf("[LEAF DEBUG] First leaf: node_index=%d, leaf_ptr=%d, leaf_size=%d\n",
                   node_index, leaf_ptr, leaf_size);
            printf("[LEAF DEBUG]   First 3 indices: %d %d %d\n",
                   device_node_index[leaf_ptr + 1],
                   device_node_index[leaf_ptr + 2],
                   device_node_index[leaf_ptr + 3]);
        }

        // Process all points in this leaf (indices are at leaf_ptr+1, leaf_ptr+2, ...)
        for (int i = 0; i < leaf_size && checks < max_checks; ++i) {
            // CRITICAL FIX: Read dataset index from hybrid array (double-indirection)
            int dataset_idx = device_node_index[leaf_ptr + 1 + i];

            // Check if we've already examined this dataset point
            bool already_checked = false;
            for (int j = 0; j < num_checked; ++j) {
                if (checked_indices[j] == dataset_idx) {
                    already_checked = true;
                    break;
                }
            }

            if (already_checked) continue;  // Skip duplicate

            // Mark as checked
            checked_indices[num_checked++] = dataset_idx;

            const unsigned char* point = dataset + dataset_idx * padded_bytes;

            int dist = compute_hamming_distance(query, point, actual_bytes);
            checks++;

            // ========================================================================
            // PHASE 4: Heap & Leaf Processing Instrumentation (Debug - Query 0 only)
            // ========================================================================
            if (threadIdx.x == 0 && blockIdx.x == 0 && num_checked <= 30) {
                // Log first 30 leaf visits
                int prev_worst = result_dists[0];  // Max-heap root = worst distance
                bool will_insert = (dist < prev_worst);

                printf("[PHASE 4 LEAF %d] Dataset idx=%d, dist=%d, will_insert=%s\n",
                       num_checked - 1, dataset_idx, dist,
                       will_insert ? "YES" : "NO");
                printf("  Heap before: ids=[%d,%d,%d] dists=[%d,%d,%d]\n",
                       result_indices[0], result_indices[1], result_indices[2],
                       result_dists[0], result_dists[1], result_dists[2]);
            }

            // Insert into k-NN heap if better than k-th neighbor
            insert_into_heap_int<K>(result_dists, result_indices, dist, dataset_idx);

            // Log heap state AFTER insertion if it changed
            if (threadIdx.x == 0 && blockIdx.x == 0 && num_checked <= 30 && dist < result_dists[0]) {
                printf("  Heap after:  ids=[%d,%d,%d] dists=[%d,%d,%d]\n\n",
                       result_indices[0], result_indices[1], result_indices[2],
                       result_dists[0], result_dists[1], result_dists[2]);
            }
        }

        return -1;  // Leaf processed, no children to explore
    }

    // Internal node: explore branches
    return explore_node_branches_hamming(
        tree_nodes, tree_pivots, device_node_index, node_index, query,
        pq_dists, pq_nodes, pq_size, padded_bytes, actual_bytes, max_pq_size
    );
}

/**
 * @brief Main Hierarchical search kernel (thread-per-query design)
 *
 * Each thread processes one query using best-first search through the Hierarchical tree.
 * Maintains a priority queue (min-heap) for nodes to explore and a result heap
 * (max-heap) for k-nearest neighbors.
 *
 * **Key differences from K-Means:**
 * - Uses Hamming distance instead of L2
 * - Works with binary descriptors (unsigned char*)
 * - Distance type is int instead of float
 * - No radius-based pruning (hierarchical doesn't store radii)
 *
 * Algorithm:
 * 1. Initialize result heap with INT_MAX
 * 2. Start from root node (index 0)
 * 3. Best-first search:
 *    - Explore current node (find_nearest_neighbor_hamming)
 *    - If leaf: compute distances, update results
 *    - If internal: explore_node_branches_hamming, get closest child
 *    - When branch ends: pop next node from priority queue
 * 4. Continue until checks >= max_checks or PQ empty
 * 5. Result heap already sorted (max-heap property)
 *
 * @tparam K Number of nearest neighbors
 * @tparam MAX_CHECKS Maximum distance computations per query
 * @param dataset Full dataset [N x bytes] (binary descriptors)
 * @param queries Query descriptors [num_queries x bytes]
 * @param tree_nodes Flat array of tree nodes (breadth-first)
 * @param tree_pivots Flat array of pivot descriptors [num_nodes x bytes]
 * @param dataset_indices Leaf node dataset indices
 * @param result_indices Output k-NN indices [num_queries x K]
 * @param result_dists Output k-NN Hamming distances [num_queries x K]
 * @param num_queries Number of queries
 * @param bytes Descriptor length in bytes
 * @param num_nodes Total number of tree nodes
 */
template<int K, int MAX_CHECKS>
__global__ void hierarchical_search_kernel(
    const unsigned char* __restrict__ dataset,
    const unsigned char* __restrict__ queries,
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const unsigned char* __restrict__ tree_pivots,
    const int* __restrict__ dataset_indices,
    const int* __restrict__ device_node_index,  // CRITICAL FIX: nodeIndex indirection array
    int* __restrict__ result_indices,
    int* __restrict__ result_distances,
    int num_queries,
    int padded_bytes,      // Stride for array indexing (with padding)
    int actual_bytes,      // Actual descriptor length for Hamming distance
    int num_nodes,
    int num_trees,         // Number of trees (roots at indices 0..num_trees-1)
    int branching          // Branching factor (tree N's children start at N*branching)
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (gid >= num_queries) return;

    // Query descriptor for this thread
    const unsigned char* query = queries + gid * padded_bytes;

    // Per-thread result heap (k-NN, max-heap)
    int result_dists[K];
    int result_ids[K];
    init_result_heap_int<K>(result_dists, result_ids);

    // Per-thread priority queue (nodes to explore, min-heap)
    const int MAX_PQ_SIZE = MAX_CHECKS;
    int pq_dists[MAX_PQ_SIZE];
    int pq_nodes[MAX_PQ_SIZE];
    int pq_size = 0;

    // Per-thread duplicate detection array
    // Tracks which dataset indices have been examined (prevents duplicate checks)
    int checked_indices[MAX_CHECKS];
    int num_checked = 0;

    int checks = 0;
    int current_tree = 1;  // Start at 1 (tree 0 explored before loop, like OpenCL)

    // Explore tree 0 root (IMPLICIT root - use special function)
    // Tree 0's children are at pivot indices 0*branching to (0+1)*branching-1
    int next_node = explore_root_branches_hamming(
        tree_pivots, device_node_index, 0, branching, query,
        pq_dists, pq_nodes, pq_size, padded_bytes, actual_bytes, MAX_PQ_SIZE
    );

    // Best-first search with multiple trees
    // Pattern from OpenCL: explore each tree root first, then fall back to PQ
    // CRITICAL: Use do-while to ensure tree exploration happens before budget check
    // This ensures all trees are explored even if tree 0 exhausts the check budget
    do {
        // Process current node
        next_node = find_nearest_neighbor_hamming<K>(
            tree_nodes, tree_pivots, dataset_indices, device_node_index, dataset,
            next_node, query,
            result_dists, result_ids,
            pq_dists, pq_nodes, pq_size,
            checks, padded_bytes, actual_bytes, MAX_CHECKS, MAX_PQ_SIZE,
            checked_indices, num_checked
        );

        // If current branch exhausted
        if (next_node == -1) {
            // First, try exploring remaining tree roots (trees 1, 2, 3, ...)
            // Roots are IMPLICIT - tree N's children at pivot indices N*branching
            if (current_tree < num_trees) {
                next_node = explore_root_branches_hamming(
                    tree_pivots, device_node_index, current_tree, branching, query,
                    pq_dists, pq_nodes, pq_size, padded_bytes, actual_bytes, MAX_PQ_SIZE
                );
                current_tree++;
            } else if (pq_size > 0) {
                // All trees explored, pop minimum (best) from priority queue
                next_node = pq_nodes[0];
                pop_min_heap_int(pq_dists, pq_nodes, pq_size);
                pq_size--;
            }
        }
    } while (next_node != -1 && (checks < MAX_CHECKS || checks < K));

    // ========================================================================
    // PHASE 4: Final Heap State Verification (Debug - Query 0 only)
    // ========================================================================
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n[PHASE 4 FINAL] Search complete for query 0:\n");
        printf("  Total checks performed: %d\n", checks);
        printf("  Total unique dataset indices examined: %d\n", num_checked);
        printf("  Heap BEFORE sorting: ids=[%d,%d,%d] dists=[%d,%d,%d]\n",
               result_ids[0], result_ids[1], result_ids[2],
               result_dists[0], result_dists[1], result_dists[2]);
    }

    // CRITICAL: Sort the max-heap before copying to output
    // Max-heap property does NOT guarantee sorted order - elements can be in arbitrary order
    // Example: [45, 43, 45] is a valid max-heap but NOT sorted
    // This matches OpenCL behavior (nn_opencl_index.h line 971: sortHeap())
    sort_result_heap_int<K>(result_dists, result_ids);

    // Log final sorted results for query 0
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("  Heap AFTER sorting:  ids=[%d,%d,%d] dists=[%d,%d,%d]\n\n",
               result_ids[0], result_ids[1], result_ids[2],
               result_dists[0], result_dists[1], result_dists[2]);
    }

    // Copy sorted results to output (now in ascending distance order)
    for (int i = 0; i < K; ++i) {
        result_indices[gid * K + i] = result_ids[i];
        result_distances[gid * K + i] = result_dists[i];
    }
}

/**
 * @brief Kernel launcher (dispatches to template instantiations)
 *
 * This function provides runtime dispatch to compile-time template parameters.
 * Different k and max_checks values require different kernel instantiations
 * for optimal performance.
 *
 * @param dataset Dataset descriptors (GPU)
 * @param queries Query descriptors (GPU)
 * @param tree_nodes Tree structure (GPU)
 * @param tree_pivots Pivot descriptors (GPU)
 * @param dataset_indices Leaf indices (GPU)
 * @param result_indices Output indices (GPU)
 * @param result_distances Output distances (GPU)
 * @param num_queries Number of queries
 * @param bytes Descriptor length in bytes
 * @param num_nodes Number of tree nodes
 * @param k Number of nearest neighbors
 * @param max_checks Max distance computations
 * @param grid CUDA grid dimensions
 * @param block CUDA block dimensions
 * @return true if kernel launched successfully, false if unsupported k
 */
bool launch_hierarchical_search(
    const unsigned char* dataset,
    const unsigned char* queries,
    const KMeansNodeGPU* tree_nodes,
    const unsigned char* tree_pivots,
    const int* dataset_indices,
    const int* device_node_index,  // CRITICAL FIX: nodeIndex indirection array
    int* result_indices,
    int* result_distances,
    size_t num_queries,
    size_t padded_bytes,
    size_t actual_bytes,
    size_t num_nodes,
    int k,
    int max_checks,
    int num_trees,
    int branching,
    dim3 grid,
    dim3 block
) {
    // Dispatch based on k and max_checks
    // Common configurations: k ∈ {1, 5, 10, 20, 50, 100}, max_checks ∈ {32, 64, 128, 256}

    if (k == 1) {
        if (max_checks <= 32) {
            hierarchical_search_kernel<1, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 64) {
            hierarchical_search_kernel<1, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 128) {
            hierarchical_search_kernel<1, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else {
            hierarchical_search_kernel<1, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        }
    } else if (k == 3) {
        if (max_checks <= 32) {
            hierarchical_search_kernel<3, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 64) {
            hierarchical_search_kernel<3, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 128) {
            hierarchical_search_kernel<3, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 256) {
            hierarchical_search_kernel<3, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 512) {
            hierarchical_search_kernel<3, 512><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 1024) {
            hierarchical_search_kernel<3, 1024><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else {
            hierarchical_search_kernel<3, 2048><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        }
    } else if (k == 5) {
        if (max_checks <= 32) {
            hierarchical_search_kernel<5, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 64) {
            hierarchical_search_kernel<5, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 128) {
            hierarchical_search_kernel<5, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else {
            hierarchical_search_kernel<5, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        }
    } else if (k == 10) {
        if (max_checks <= 32) {
            hierarchical_search_kernel<10, 32><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 64) {
            hierarchical_search_kernel<10, 64><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else if (max_checks <= 128) {
            hierarchical_search_kernel<10, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else {
            hierarchical_search_kernel<10, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        }
    } else if (k == 20) {
        if (max_checks <= 128) {
            hierarchical_search_kernel<20, 128><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        } else {
            hierarchical_search_kernel<20, 256><<<grid, block>>>(
                dataset, queries, tree_nodes, tree_pivots, dataset_indices, device_node_index,
                result_indices, result_distances,
                num_queries, padded_bytes, actual_bytes, num_nodes, num_trees, branching);
        }
    } else {
        return false;  // Unsupported k value
    }

    return true;
}

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_CUDA_HIERARCHICAL_SEARCH_KERNEL_CUH_
