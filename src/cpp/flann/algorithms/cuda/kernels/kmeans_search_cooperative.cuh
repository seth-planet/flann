/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024-2025 (CUDA cooperative kernel implementation)
 * Copyright 2017  Seth Price (seth@planet.com) - original OpenCL version
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

#ifndef FLANN_CUDA_KMEANS_SEARCH_COOPERATIVE_CUH_
#define FLANN_CUDA_KMEANS_SEARCH_COOPERATIVE_CUH_

#ifdef FLANN_USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>  // For fprintf error reporting
#include <float.h>
#include <limits>
#include "../kmeans_node_gpu.h"
#include "distance_kernels.cuh"

namespace flann {
namespace cuda {

/**
 * @brief Cooperative K-Means search kernel
 *
 * This kernel implements multi-threaded cooperative search matching OpenCL's
 * findNeighborsLocal architecture. LOC_SIZE threads (32-128) cooperate on a
 * single query using shared memory for heap operations and parallel tree descent.
 *
 * Architecture:
 * - One thread block per query (grid.x = num_queries)
 * - LOC_SIZE threads per block (block.x = 32, 64, or 128)
 * - Shared heap of size N_HEAP = LOC_SIZE * 2
 * - Three phases:
 *   1. Tree descent: Find leaf nodes (cooperative)
 *   2. Leaf processing: Compute distances to leaf points (parallel)
 *   3. Result extraction: Store top-K results
 *
 * Performance: Best when heap_size <= LOC_SIZE (small trees, high precision queries)
 * Falls back to single-threaded kernel for large heaps.
 */

//=============================================================================
// SHARED MEMORY STRUCTURE
//=============================================================================

/**
 * @brief Shared memory layout for cooperative search
 *
 * @tparam LOC_SIZE Number of threads per block (32, 64, or 128)
 */
template<int LOC_SIZE>
struct CooperativeSearchShared {
    // Heap for node/leaf exploration (N_HEAP = LOC_SIZE * 2)
    // Lower half [0...LOC_SIZE-1]: Will contain final k-NN results
    // Upper half [LOC_SIZE...N_HEAP-1]: Working space for new nodes/leaves
    int heap_ids[LOC_SIZE * 2];
    float heap_dists[LOC_SIZE * 2];

    // Query vector copied to shared memory for fast access
    float query[256];  // Max descriptor size (will be trimmed by compiler if smaller)

    // Atomic work counters
    int loc_ptr;       // Next available slot in heap
    int loc_done;      // Termination flag (1 = done, 0 = continue)
    int leaf_count;    // Number of leaves processed
};

//=============================================================================
// HELPER DEVICE FUNCTIONS
//=============================================================================

/**
 * @brief Initialize shared memory heap to invalid state
 *
 * Each thread initializes its position and position+LOC_SIZE to:
 * - Distance: FLT_MAX (maximum distance)
 * - ID: INT_MAX (invalid leaf marker)
 *
 * @param shared Shared memory structure
 * @param num_nodes Total number of tree nodes (for leaf detection)
 */
template<int LOC_SIZE>
__device__ inline void init_heap(CooperativeSearchShared<LOC_SIZE>* shared, int num_nodes)
{
    int tid = threadIdx.x;
    shared->heap_dists[tid] = FLT_MAX;
    shared->heap_ids[tid] = INT_MAX;
    shared->heap_dists[tid + LOC_SIZE] = FLT_MAX;
    shared->heap_ids[tid + LOC_SIZE] = INT_MAX;
}

/**
 * @brief Check if all threads are done (AND reduction)
 *
 * If ANY thread has done=false, sets shared->loc_done = 0.
 * This implements a cooperative AND reduction across all threads.
 *
 * @param done Per-thread done flag (true if this thread is done)
 * @param shared Shared memory structure
 */
template<int LOC_SIZE>
__device__ inline void check_done(bool done, CooperativeSearchShared<LOC_SIZE>* shared)
{
    if (!done) {
        shared->loc_done = 0;
    }
    __syncthreads();
}

//=============================================================================
// BITONIC SORT (Parallel Sorting Network)
//=============================================================================

/**
 * @brief Bitonic merge step
 *
 * Parallel compare-exchange operation for bitonic sort.
 * Each thread compares two elements and swaps if out of order.
 *
 * @param dists Distance array (shared memory)
 * @param ids ID array (shared memory)
 * @param size Current merge size
 * @param dir Sort direction (0 = ascending, 1 = descending)
 */
template<int N_HEAP>
__device__ inline void bitonic_merge(
    float* __restrict__ dists,
    int* __restrict__ ids,
    int size,
    int dir
) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
        __syncthreads();

        int tid = threadIdx.x;
        int pos = 2 * tid - (tid & (stride - 1));

        if (pos + stride < N_HEAP) {
            float keyA = dists[pos];
            float keyB = dists[pos + stride];

            // Swap if out of order (ascending: keyA < keyB, descending: keyA > keyB)
            // Fixed: Use < instead of > to match OpenCL bitonic sort semantics
            if ((keyA < keyB) == dir) {
                int valA = ids[pos];
                int valB = ids[pos + stride];
                dists[pos] = keyB;
                dists[pos + stride] = keyA;
                ids[pos] = valB;
                ids[pos + stride] = valA;
            }
        }
    }
}

/**
 * @brief Bitonic sort for shared heap
 *
 * Sorts heap in ascending order by distance using bitonic sort network.
 * Complexity: O(log²N) parallel steps, where N = LOC_SIZE * 2
 *
 * After sorting, lower half [0...LOC_SIZE-1] contains closest nodes/leaves.
 *
 * @param dists Distance array (shared memory)
 * @param ids ID array (shared memory)
 */
template<int LOC_SIZE>
__device__ inline void sort_heap(
    float* __restrict__ dists,
    int* __restrict__ ids
) {
    const int N_HEAP = LOC_SIZE * 2;

    // Build bitonic sequence from small sizes to N_HEAP
    for (int size = 2; size < N_HEAP; size <<= 1) {
        // Determine direction based on position
        int ddd = (threadIdx.x & (size / 2)) != 0;
        bitonic_merge<N_HEAP>(dists, ids, size, ddd);
    }

    // Final merge to full ascending order
    bitonic_merge<N_HEAP>(dists, ids, N_HEAP, 0);
    __syncthreads();
}

//=============================================================================
// TREE DESCENT (Phase 1: Find Leaf Nodes)
//=============================================================================

/**
 * @brief Compute distances to children and insert into heap (COOPERATIVE)
 *
 * Uses strided access pattern where multiple threads cooperate to process
 * all children of all parent nodes in the lower heap.
 *
 * Thread assignment:
 * - loc_id = tid / BRANCHING  (which parent slot to process)
 * - child_offset = tid % BRANCHING  (which child of that parent)
 *
 * This implements the OpenCL findNewNodeDist() function with correct
 * cooperative semantics matching the original algorithm.
 *
 * @param tree_nodes Flat array of tree nodes
 * @param tree_pivots Flat array of pivot coordinates
 * @param shared Shared memory structure
 * @param num_nodes Total number of tree nodes
 * @param dim Vector dimension
 * @param cb_index CB_INDEX parameter for variance adjustment
 */
template<int LOC_SIZE, int BRANCHING>
__device__ inline void find_new_node_dist(
    const int* __restrict__ node_index,          // NEW: nodeIndex array
    const float* __restrict__ node_pivots,       // Renamed from tree_pivots
    const float* __restrict__ node_variance,     // NEW: separate variance
    CooperativeSearchShared<LOC_SIZE>* shared,
    int num_nodes,
    int dim,
    float cb_index
) {
    int tid = threadIdx.x;
    const int N_HEAP = LOC_SIZE * 2;

    // ========================================================================
    // OpenCL PARITY: Search FULL N_HEAP for parents (not just LOC_SIZE)
    // ========================================================================
    // Determine which parent slot this thread will process
    unsigned int loc_id = tid / BRANCHING;

    // Skip leaf entries - simplified detection
    // In OpenCL architecture: heap_ids[x] >= num_nodes means it's a leaf pointer (data region)
    while (loc_id < N_HEAP && shared->heap_ids[loc_id] >= num_nodes) {
        loc_id += LOC_SIZE / BRANCHING;
    }

    // ========================================================================
    // OpenCL PARITY: Indirect pointer arithmetic for child computation
    // ========================================================================
    int child_offset = tid % BRANCHING;

    // heap_ids[loc_id] contains BASE OFFSET (first child BFS index)
    // Add child_offset to get actual child node ID
    int parent_ptr = (loc_id < N_HEAP) ? shared->heap_ids[loc_id] : -1;
    int child_node_id = (parent_ptr >= 0 && parent_ptr < num_nodes) ?
                        (parent_ptr + child_offset) : -1;

    __syncthreads();

    // ========================================================================
    // Invalidate parent slots (OpenCL parity: now searches N_HEAP)
    // ========================================================================
    if (loc_id < N_HEAP && child_offset == 0) {
        shared->heap_dists[loc_id] = FLT_MAX;
        shared->heap_ids[loc_id] = INT_MAX;
    }

    // ========================================================================
    // OpenCL PARITY: Compute distance and store INDIRECT pointer
    // ========================================================================
    if (child_node_id >= 0 && child_node_id < num_nodes) {
        // Load pivot for this child node
        const float* child_pivot = node_pivots + child_node_id * dim;

        // Compute distance
        float dist = compute_l2_distance(
            shared->query,
            child_pivot,
            dim
        );

        // Apply CB_INDEX variance adjustment
        float adjusted_dist = dist - cb_index * node_variance[child_node_id];

        // Store in upper half of heap
        int heap_idx = LOC_SIZE + tid;
        if (heap_idx < N_HEAP) {
            shared->heap_dists[heap_idx] = adjusted_dist;

            // CRITICAL: Store INDIRECT POINTER from nodeIndex, not direct node ID
            // This is what enables the OpenCL parity architecture
            shared->heap_ids[heap_idx] = node_index[child_node_id];
        }
    }
}

/**
 * @brief Cooperative tree descent to find leaf nodes
 *
 * Starting from the root, all threads cooperatively explore the tree
 * by repeatedly:
 * 1. Processing parent nodes in lower heap
 * 2. Computing distances to children
 * 3. Inserting children into upper heap
 * 4. Sorting heap to prioritize closest nodes
 * 5. Repeating until only leaf nodes remain in lower heap
 *
 * This implements the OpenCL findNodes() function.
 *
 * @param node_index OpenCL-style nodeIndex array
 * @param node_pivots Flat array of pivot coordinates
 * @param node_variance Flat array of node variance values
 * @param shared Shared memory structure
 * @param num_nodes Total number of tree nodes
 * @param dim Vector dimension
 * @param cb_index CB_INDEX parameter for variance adjustment
 */
template<int LOC_SIZE, int BRANCHING = 32>
__device__ void find_nodes_cooperative(
    const int* __restrict__ node_index,
    const float* __restrict__ node_pivots,
    const float* __restrict__ node_variance,
    CooperativeSearchShared<LOC_SIZE>* shared,
    int num_nodes,
    int dim,
    float cb_index
) {
    int tid = threadIdx.x;

    // Initialize heap
    init_heap<LOC_SIZE>(shared, num_nodes);
    __syncthreads();

    // Initialize with root's first child index (OpenCL parity)
    // CUDA tree layout: root at index 0, children at indices 1..BRANCHING
    // heap_ids stores FIRST CHILD INDEX, so we start with 1 (root's first child)
    // This matches OpenCL where heapId[0] = 0 points to root's children at 0..BRANCHING-1
    if (tid == 0) {
        shared->heap_ids[0] = 1;  // Root's first child index (CUDA has explicit root at 0)
        shared->heap_dists[0] = 0.0f;
    }
    __syncthreads();

    // Find closest children of root
    find_new_node_dist<LOC_SIZE, BRANCHING>(node_index, node_pivots, node_variance, shared, num_nodes, dim, cb_index);

    // Sort to prioritize closest nodes
    sort_heap<LOC_SIZE>(shared->heap_dists, shared->heap_ids);
    __syncthreads();

    // Iteratively descend tree until only leaves remain
    int iteration = 0;
    const int MAX_ITERATIONS = 100;  // Safety limit

    do {
        iteration++;

        // Explore parent nodes to discover children
        find_new_node_dist<LOC_SIZE, BRANCHING>(node_index, node_pivots, node_variance, shared, num_nodes, dim, cb_index);

        // Initialize termination flag (OpenCL PARITY: no barrier before sort)
        if (tid == 0) {
            shared->loc_done = 1;  // Assume done until proven otherwise
        }
        // NOTE: sortHeap has its own barriers, no extra barrier needed here

        // Sort to bring closest nodes to lower heap
        sort_heap<LOC_SIZE>(shared->heap_dists, shared->heap_ids);

        // Check if we still have parent nodes in lower heap (OpenCL-style)
        // heap_ids stores FIRST CHILD INDEX:
        //   - >= num_nodes means leaf pointer (points to data region)
        //   - < num_nodes means valid first-child-index, not a leaf
        // IMPORTANT: Do NOT lookup node_index[my_id] because my_id is a first-child-index,
        // not a node index. The slot at my_id might not even have valid data!
        int my_id = shared->heap_ids[tid];
        bool is_leaf = (my_id >= num_nodes) || (my_id < 0);

        check_done<LOC_SIZE>(is_leaf, shared);

    } while (!shared->loc_done && iteration < MAX_ITERATIONS);

    __syncthreads();
}

//=============================================================================
// LEAF PROCESSING (Phase 2: Compute Distances to Leaf Points)
//=============================================================================

/**
 * @brief Process leaf nodes to find k nearest neighbors (COOPERATIVE)
 *
 * ALL threads cooperatively process ALL leaves in the lower heap.
 * Each thread extracts its leaf's dataset range, then cooperatively
 * fills the heap with points from all leaves using atomic insertion.
 *
 * This implements the OpenCL findLeaves() function with correct
 * cooperative semantics.
 *
 * @param tree_nodes Flat array of tree nodes
 * @param dataset_indices Flat array mapping tree leaf slots to dataset indices
 * @param dataset Full dataset (all points)
 * @param shared Shared memory structure
 * @param num_nodes Total number of tree nodes
 * @param dim Vector dimension
 */
template<int LOC_SIZE>
__device__ void find_leaves_cooperative(
    const int* __restrict__ node_index,          // OpenCL PARITY: unified nodeIndex array
    const float* __restrict__ dataset,
    CooperativeSearchShared<LOC_SIZE>* shared,
    int num_nodes,
    int dim
) {
    int tid = threadIdx.x;
    const int N_HEAP = LOC_SIZE * 2;

    // ========================================================================
    // OpenCL PARITY: Extract leaf information from heap (NO DEDUPLICATION)
    // ========================================================================
    // After find_nodes_cooperative, heap contains INDIRECT POINTERS
    // For leaf nodes: heap_ids[tid] >= num_nodes means it points to data region
    //
    // NOTE: OpenCL does NOT use deduplication for single-tree K-Means.
    // Removed bitmap deduplication to match OpenCL and reduce overhead (-10%).

    int leaf_ptr = shared->heap_ids[tid];
    int leaf_count = 0;
    bool should_process_leaf = false;

    // Direct leaf pointer extraction - simple validity check (OpenCL parity)
    if (leaf_ptr >= num_nodes && leaf_ptr != INT_MAX) {
        // This is a leaf pointer - read count from unified nodeIndex array
        // Data region format: [count, id1, id2, ..., idN]
        leaf_count = node_index[leaf_ptr];
        should_process_leaf = (leaf_count > 0);  // Process if has valid points
    }

    // ========================================================================
    // STEP 2: Reset heap for distance computation
    // ========================================================================
    init_heap<LOC_SIZE>(shared, num_nodes);
    __syncthreads();

    // ========================================================================
    // STEP 3: Fill heap with dataset indices from ALL leaf nodes
    // ========================================================================
    // Each thread with a valid leaf contributes its points to the heap

    if (tid == 0) {
        shared->loc_ptr = 0;
        shared->loc_done = 0;
    }
    __syncthreads();

    // ========================================================================
    // FIX 2: Track pre-fill progress to avoid reprocessing same leaves
    // ========================================================================
    // OpenCL PARITY: Use absolute offset (not relative index) to avoid duplicate reads
    // This matches OpenCL's leafPtr increment pattern
    int my_leaf_offset = leaf_ptr + 1;  // Start at first dataset ID (skip count at leaf_ptr)
    int leaf_end = leaf_ptr + leaf_count;  // End of this leaf's data

    // First pass: Fill both halves of heap
    // DEDUPLICATION: Only process if this thread claimed ownership of the leaf
    if (should_process_leaf && leaf_count > 0) {
        while (my_leaf_offset <= leaf_end) {
            int slot = atomicAdd(&shared->loc_ptr, 1);
            if (slot < N_HEAP) {
                // OpenCL PARITY: Read dataset ID from unified array using absolute offset
                // Data format: [count, id1, id2, ..., idN]
                // my_leaf_offset increments through: leaf_ptr+1, leaf_ptr+2, ..., leaf_ptr+count
                int dataset_idx = node_index[my_leaf_offset++];  // Post-increment moves to next
                shared->heap_ids[slot] = dataset_idx;
            } else {
                break;  // Heap full
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // STEP 4: Compute distances for pre-filled points (LOWER HALF ONLY)
    // ========================================================================
    // OpenCL PARITY: Only compute lower half distances in pre-fill.
    // The main loop will compute upper half distances at its START (before sort).
    // This avoids computing upper half distances TWICE (once here, once in loop).

    // Lower half only (each thread handles one slot in lower half)
    if (tid < LOC_SIZE && shared->heap_ids[tid] != INT_MAX) {
        int dataset_idx = shared->heap_ids[tid];
        shared->heap_dists[tid] = compute_l2_distance(
            shared->query,
            dataset + dataset_idx * dim,
            dim
        );
    }
    // NOTE: Upper half distances are computed by main loop's first iteration
    __syncthreads();

    // ========================================================================
    // STEP 5: Main loop - continue until all leaf points processed
    // ========================================================================
    // OpenCL PARITY: Structure loop to match OpenCL's findLeaves() exactly:
    //   1. Compute distances for upper half (from previous iteration's insertions)
    //   2. Sort heap
    //   3. Reset loc_ptr to LOC_SIZE
    //   4. Insert IDs only (no distance computation)
    //   5. Check done
    // This order eliminates one barrier per iteration compared to previous CUDA.

    int leaf_iteration = 0;
    int locI = 0;  // Track last atomic result for termination check

    do {
        leaf_iteration++;

        // ========================================================================
        // OpenCL PARITY: Compute distances at START of loop (before sort)
        // ========================================================================
        // This computes distances for IDs inserted in the PREVIOUS iteration.
        // First iteration: distances already computed in pre-fill section.
        if (shared->heap_ids[LOC_SIZE + tid] != INT_MAX) {
            int dataset_idx = shared->heap_ids[LOC_SIZE + tid];
            shared->heap_dists[LOC_SIZE + tid] = compute_l2_distance(
                shared->query,
                dataset + dataset_idx * dim,
                dim
            );
        }

        // Sort heap to bring closest points to bottom
        sort_heap<LOC_SIZE>(shared->heap_dists, shared->heap_ids);

        // Reset for next batch (barrier inside sortHeap handles sync)
        if (tid == 0) {
            shared->loc_ptr = LOC_SIZE;  // Start filling from middle
            shared->loc_done = 1;         // Assume done until proven otherwise
        }
        __syncthreads();

        // ========================================================================
        // OpenCL PARITY: Insert IDs ONLY (no distance computation)
        // ========================================================================
        locI = 0;
        while (should_process_leaf && my_leaf_offset <= leaf_end) {
            locI = atomicAdd(&shared->loc_ptr, 1);
            if (locI < N_HEAP) {
                int dataset_idx = node_index[my_leaf_offset++];
                shared->heap_ids[locI] = dataset_idx;
                // Distance computed at START of NEXT iteration
            } else {
                break;  // Heap full
            }
        }

        // ========================================================================
        // OpenCL PARITY: checkDone has barrier inside, no extra barrier needed
        // ========================================================================
        check_done<LOC_SIZE>(locI == 0, shared);

    } while (!shared->loc_done);

    __syncthreads();
}

//=============================================================================
// MAIN COOPERATIVE KERNEL
//=============================================================================

/**
 * @brief Cooperative K-Means search kernel
 *
 * One thread block per query, LOC_SIZE threads cooperate.
 *
 * @tparam K Number of nearest neighbors to find
 * @tparam LOC_SIZE Number of threads per block (32, 64, or 128)
 * @tparam VECLEN Vector dimension (compile-time constant for optimization)
 * @tparam BRANCHING Tree branching factor (32 or 64)
 *
 * @param dataset Full dataset (all points)
 * @param queries Query vectors
 * @param tree_nodes Flat array of tree nodes (breadth-first order)
 * @param tree_pivots Flat array of pivot coordinates
 * @param dataset_indices Mapping from tree leaf slots to dataset indices
 * @param result_indices Output k-NN indices [num_queries x K]
 * @param result_distances Output k-NN distances [num_queries x K]
 * @param num_queries Number of query vectors
 * @param dim Vector dimension (runtime parameter, must match VECLEN)
 * @param num_nodes Total number of tree nodes
 * @param cb_index CB_INDEX parameter for variance adjustment
 */
template<int K, int LOC_SIZE, int VECLEN, int BRANCHING = 32>
__global__ void kmeans_search_cooperative_kernel(
    const float* __restrict__ dataset,
    const float* __restrict__ queries,
    const int* __restrict__ node_index,        // OpenCL PARITY: unified nodeIndex array
    const float* __restrict__ node_pivots,     // Node pivot coordinates
    const float* __restrict__ node_variance,   // Node variance values
    int* __restrict__ result_indices,
    float* __restrict__ result_distances,
    int num_queries,
    int dim,
    int num_nodes,
    float cb_index
) {
    // Shared memory allocation
    __shared__ CooperativeSearchShared<LOC_SIZE> shared;

    int query_id = blockIdx.x;
    int tid = threadIdx.x;

    if (query_id >= num_queries) return;

    //-------------------------------------------------------------------------
    // Phase 1: Copy query to shared memory (parallel)
    //-------------------------------------------------------------------------
    const float* query_global = queries + query_id * dim;
    for (int i = tid; i < dim; i += LOC_SIZE) {
        shared.query[i] = query_global[i];
    }
    __syncthreads();

    //-------------------------------------------------------------------------
    // Phase 2: Find leaf nodes (cooperative tree descent)
    //-------------------------------------------------------------------------
    find_nodes_cooperative<LOC_SIZE, BRANCHING>(
        node_index, node_pivots, node_variance, &shared, num_nodes, dim, cb_index
    );

    //-------------------------------------------------------------------------
    // Phase 3: Process leaf points (cooperative distance computation)
    //-------------------------------------------------------------------------
    find_leaves_cooperative<LOC_SIZE>(
        node_index, dataset, &shared, num_nodes, dim
    );

    //-------------------------------------------------------------------------
    // Phase 4: Store results (threads 0...K-1 write)
    //-------------------------------------------------------------------------
    // Remove duplicates if necessary (K-Means with single tree shouldn't have duplicates)
    // For multi-tree K-Means, would need duplicate removal here

    // Copy top-K results from shared heap to global memory
    for (int i = tid; i < K; i += LOC_SIZE) {
        int out_idx = query_id * K + i;
        result_indices[out_idx] = shared.heap_ids[i];
        result_distances[out_idx] = shared.heap_dists[i];
    }
}

//=============================================================================
// KERNEL LAUNCHER
//=============================================================================

/**
 * @brief Launch cooperative K-Means search kernel
 *
 * Dispatches to appropriate template instantiation based on K and LOC_SIZE.
 *
 * @tparam K Number of nearest neighbors
 * @param dataset Full dataset
 * @param queries Query vectors
 * @param tree_nodes Tree structure
 * @param tree_pivots Node pivots
 * @param dataset_indices Leaf-to-dataset mapping
 * @param result_indices Output indices
 * @param result_distances Output distances
 * @param num_queries Number of queries
 * @param dim Vector dimension
 * @param num_nodes Total tree nodes
 * @param heap_size Calculated heap size (for validation)
 * @param loc_size Device capability (32, 64, or 128)
 * @param cb_index CB_INDEX parameter
 * @return true if launched successfully
 */
template<int K>
bool launch_kmeans_search_cooperative(
    const float* dataset,
    const float* queries,
    const int* node_index,           // OpenCL PARITY: unified nodeIndex array
    const float* node_pivots,        // Renamed from tree_pivots
    const float* node_variance,      // Separate variance array
    int* result_indices,
    float* result_distances,
    size_t num_queries,
    size_t dim,
    size_t num_nodes,
    int heap_size,
    int loc_size,
    int branching,  // Add branching factor parameter
    float cb_index
) {
    dim3 grid(num_queries);   // One block per query
    dim3 block(loc_size);     // LOC_SIZE threads per block

    // Dispatch based on LOC_SIZE, VECLEN, and BRANCHING
    // For simplicity, use 128 as default VECLEN (SIFT descriptors)
    // Support branching=32 and branching=64

    if (loc_size == 32) {
        if (dim <= 128) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 32, 128, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 32, 128, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;  // Unsupported branching factor
            }
        } else if (dim <= 256) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 32, 256, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 32, 256, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;  // Unsupported branching factor
            }
        } else {
            return false;  // Unsupported dimension
        }
    } else if (loc_size == 64) {
        if (dim <= 128) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 64, 128, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 64, 128, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else if (dim <= 256) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 64, 256, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 64, 256, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else if (loc_size == 128) {
        if (dim <= 128) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 128, 128, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 128, 128, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else if (dim <= 256) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 128, 256, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 128, 256, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else if (loc_size == 256) {
        if (dim <= 128) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 256, 128, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 256, 128, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else if (dim <= 256) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 256, 256, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 256, 256, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else if (loc_size == 512) {
        if (dim <= 128) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 512, 128, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 512, 128, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else if (dim <= 256) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 512, 256, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 512, 256, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else if (loc_size == 1024) {
        if (dim <= 128) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 1024, 128, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 1024, 128, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else if (dim <= 256) {
            if (branching == 32) {
                kmeans_search_cooperative_kernel<K, 1024, 256, 32><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else if (branching == 64) {
                kmeans_search_cooperative_kernel<K, 1024, 256, 64><<<grid, block>>>(
                    dataset, queries, node_index, node_pivots, node_variance,
                    result_indices, result_distances, num_queries, dim, num_nodes, cb_index
                );
            } else {
                return false;
            }
        } else {
            return false;
        }
    } else {
        return false;  // Unsupported LOC_SIZE
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA K-Means] Kernel launch error: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA

#endif // FLANN_CUDA_KMEANS_SEARCH_COOPERATIVE_CUH_
