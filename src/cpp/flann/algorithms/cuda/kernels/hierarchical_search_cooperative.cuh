#ifndef FLANN_HIERARCHICAL_SEARCH_COOPERATIVE_CUH
#define FLANN_HIERARCHICAL_SEARCH_COOPERATIVE_CUH

#include "bitonic_sort.cuh"
#include "distance_kernels.cuh"
#include "../kmeans_node_gpu.h"

namespace flann {
namespace cuda {

/**
 * @file hierarchical_search_cooperative.cuh
 * @brief Cooperative multi-threaded hierarchical k-NN search kernel
 *
 * Implements hierarchical clustering search using cooperative thread blocks,
 * matching the OpenCL implementation in nn_opencl_index.h.
 *
 * Key differences from single-threaded CUDA version:
 * - One thread BLOCK per query (not one thread per query)
 * - Shared heap visible to all threads in block
 * - Parallel bitonic sort for heap maintenance
 * - Cooperative node exploration (multiple threads explore multiple parents)
 * - Matches OpenCL precision (0.97+)
 */

/**
 * @brief Initialize local heap to invalid values
 *
 * Matches OpenCL initLoc (nn_opencl_index.h:947-954)
 *
 * @param heap_dists Shared memory array for distances
 * @param heap_ids Shared memory array for IDs
 * @param local_id Thread's local ID
 * @param local_size Block size (number of threads)
 */
__device__ inline void init_local_heap(
    int* heap_dists,
    int* heap_ids,
    int local_id,
    int local_size
) {
    // Init both halves of the heap to invalid
    // OpenCL: heapDist[get_local_id(0)] = MAX_DIST;
    heap_dists[local_id] = INT_MAX;
    heap_ids[local_id] = INT_MAX;
    heap_dists[local_id + local_size] = INT_MAX;
    heap_ids[local_id + local_size] = INT_MAX;
}

/**
 * @brief Find new node distances - cooperative exploration
 *
 * Matches OpenCL findNewNodeDist (nn_opencl_index.h:866-899)
 *
 * Each thread handles ONE child of MULTIPLE parents.
 * Multiple threads cooperate to explore all children of parent nodes
 * currently in the heap.
 *
 * @param device_node_index Hybrid node index array (nodeIndex)
 * @param tree_pivots Flat array of pivot descriptors
 * @param query Query descriptor
 * @param heap_dists Shared heap distances
 * @param heap_ids Shared heap node IDs
 * @param num_nodes Total number of nodes in tree
 * @param branching Branching factor
 * @param local_id Thread's local ID
 * @param local_size Block size
 * @param actual_bytes Actual descriptor length in bytes
 * @param padded_bytes Padded descriptor length (multiple of 4)
 */
__device__ inline void find_new_node_dist(
    const int* __restrict__ device_node_index,
    const unsigned char* __restrict__ tree_pivots,
    const unsigned char* __restrict__ query,
    int* heap_dists,
    int* heap_ids,
    int num_nodes,
    int branching,
    int local_id,
    int local_size,
    int actual_bytes,
    int padded_bytes
) {
    // Find the next valid node pointer
    // OpenCL: unsigned int locId = get_local_id(0) / BRANCHING;
    int loc_id = local_id / branching;

    // Skip ids marked as pointers to leaves
    // OpenCL: while (locId < N_HEAP && heapId[locId] >= nNodes)
    while (loc_id < local_size && heap_ids[loc_id] >= num_nodes) {
        loc_id += local_size / branching;
    }

    // Check if we were able to find a valid node ptr
    // Go from pointer-in-heap to actual node ID
    // OpenCL: int nodeId = (locId < N_HEAP) ? (heapId[locId] + get_local_id(0) % BRANCHING) : 0;
    int node_id = (loc_id < local_size)
        ? (heap_ids[loc_id] + local_id % branching)
        : 0;

    // Ensure all node IDs are retrieved before adjusting the heap
    // OpenCL: barrier(CLK_LOCAL_MEM_FENCE);
    __syncthreads();

    if (loc_id < local_size) {
        // Invalidate this pointer if it's been used
        // OpenCL: heapDist[locId] = MAX_DIST; heapId[locId] = INT_MAX;
        heap_dists[loc_id] = INT_MAX;
        heap_ids[loc_id] = INT_MAX;

        // Store adjusted distance to node and its pointer
        // OpenCL: int i = LOC_SIZE + get_local_id(0);
        int i = local_size + local_id;

        // Compute distance to pivot
        // OpenCL: heapDist[i] = vecDistLoc(vec, nodePivots, nodeId*N_VECLEN);
        const unsigned char* child_pivot = tree_pivots + node_id * padded_bytes;
        heap_dists[i] = compute_hamming_distance(query, child_pivot, actual_bytes);

        // Store dereferenced node index
        // OpenCL: heapId[i] = nodeIndex[nodeId];
        heap_ids[i] = device_node_index[node_id];
    }
}

/**
 * @brief Find nodes phase - cooperative exploration until heap contains only leaves
 *
 * Matches OpenCL findNodes (nn_opencl_index.h:762-799)
 *
 * Explores the tree from roots to leaves, maintaining a heap of candidate nodes.
 * All threads cooperate to explore multiple parent nodes in parallel.
 * Continues until heap contains only leaf pointers.
 *
 * @param tree_nodes Flat array of tree node structures
 * @param device_node_index Hybrid node index array
 * @param tree_pivots Flat array of pivot descriptors
 * @param query Query descriptor
 * @param heap_dists Shared heap distances
 * @param heap_ids Shared heap node IDs
 * @param done_flag Shared flag indicating completion
 * @param num_nodes Total number of tree nodes
 * @param num_trees Number of parallel trees
 * @param branching Branching factor
 * @param local_id Thread's local ID
 * @param local_size Block size
 * @param actual_bytes Actual descriptor length
 * @param padded_bytes Padded descriptor length
 */
__device__ inline void find_nodes_cooperative(
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const int* __restrict__ device_node_index,
    const unsigned char* __restrict__ tree_pivots,
    const unsigned char* __restrict__ query,
    int* heap_dists,
    int* heap_ids,
    int* done_flag,
    int num_nodes,
    int num_trees,
    int branching,
    int local_id,
    int local_size,
    int actual_bytes,
    int padded_bytes
) {
    // Initialize local heap
    // OpenCL: initLoc(heapDist, heapId, nNodes);
    init_local_heap(heap_dists, heap_ids, local_id, local_size);
    __syncthreads();

    // Init the node query array with pointer to root_'s children
    // Ensure that all trees are descended
    // OpenCL: for (int i = 0; i < N_TREES; i++) { heapId[i] = i*BRANCHING; heapDist[i] = 0; }
    if (local_id < num_trees) {
        heap_ids[local_id] = local_id * branching;
        heap_dists[local_id] = 0;
    }
    __syncthreads();

    // Find the closest children to the root
    // OpenCL: findNewNodeDist(...);
    find_new_node_dist(
        device_node_index, tree_pivots, query,
        heap_dists, heap_ids,
        num_nodes, branching,
        local_id, local_size,
        actual_bytes, padded_bytes
    );

    // Sort to mix the new nodes into the correct positions in the heap
    // OpenCL: sortHeap(heapDist, heapId);
    sort_heap(heap_dists, heap_ids, local_size * 2, local_id);

    // Main exploration loop
    // OpenCL: do { ... } while (!(*locDone));
    do {
        // Use the closest parent nodes to discover additional nodes
        // OpenCL: findNewNodeDist(...);
        find_new_node_dist(
            device_node_index, tree_pivots, query,
            heap_dists, heap_ids,
            num_nodes, branching,
            local_id, local_size,
            actual_bytes, padded_bytes
        );

        // Init shared vars
        // OpenCL: if (get_local_id(0) == 0) { (*locDone) = 1; }
        if (local_id == 0) {
            *done_flag = 1;  // Set as if it's our last pass, will be flipped back if not done
        }

        // Sort to mix the new nodes into the correct positions in the heap
        // OpenCL: sortHeap(heapDist, heapId);
        sort_heap(heap_dists, heap_ids, local_size * 2, local_id);

        // While there is a pointer to a parent node in the heap, keep going
        // OpenCL: checkDone((heapId[get_local_id(0)] >= nNodes), locDone);
        if (heap_ids[local_id] < num_nodes) {
            *done_flag = 0;  // Found a parent node, not done yet
        }
        __syncthreads();

    } while (*done_flag == 0);
}

/**
 * @brief Find leaves phase - cooperative leaf processing
 *
 * Matches OpenCL findLeaves (nn_opencl_index.h:805-863)
 *
 * Processes leaf nodes to find actual k-nearest neighbors.
 * Each thread stores its leaf pointer, then cooperatively fills
 * the heap with dataset points and sorts to find best candidates.
 *
 * @param device_node_index Hybrid node index array
 * @param query Query descriptor
 * @param dataset Full dataset of descriptors
 * @param heap_dists Shared heap distances
 * @param heap_ids Shared heap IDs (dataset indices)
 * @param loc_ptr Shared pointer for atomic increments
 * @param done_flag Shared completion flag
 * @param local_id Thread's local ID
 * @param local_size Block size
 * @param actual_bytes Actual descriptor length
 * @param padded_bytes Padded descriptor length
 */
__device__ inline void find_leaves_cooperative(
    const int* __restrict__ device_node_index,
    const unsigned char* __restrict__ query,
    const unsigned char* __restrict__ dataset,
    int* heap_dists,
    int* heap_ids,
    int* loc_ptr,
    int* done_flag,
    int local_id,
    int local_size,
    int actual_bytes,
    int padded_bytes
) {
    // Store pointer to list before init local mem
    // OpenCL: int leafPtr = heapId[get_local_id(0)];
    int leaf_ptr = heap_ids[local_id];

    // OpenCL: int lastPtr = (leafPtr < INT_MAX) ? (nodeIndex[leafPtr] + leafPtr) : 0;
    int last_ptr = (leaf_ptr < INT_MAX)
        ? (device_node_index[leaf_ptr] + leaf_ptr)
        : 0;

    // Reset heap
    // OpenCL: initLoc(heapDist, heapId, nNodes);
    init_local_heap(heap_dists, heap_ids, local_id, local_size);

    // 'Unroll' this ahead of the main loop to fill both halves of the heap before
    // first sort, and because we don't really need to checkDone() so soon.
    // Init shared vars
    // OpenCL: if (get_local_id(0) == 0) (*locPtr) = 0;
    if (local_id == 0) {
        *loc_ptr = 0;
    }
    __syncthreads();

    // Find the next dataset index to check
    // OpenCL: int locI = 0; while (leafPtr < lastPtr && (locI = atomic_inc(locPtr)) < N_HEAP)
    int loc_i = 0;
    while (leaf_ptr < last_ptr && (loc_i = atomicAdd(loc_ptr, 1)) < local_size * 2) {
        // OpenCL: heapId[locI] = nodeIndex[++leafPtr];
        heap_ids[loc_i] = device_node_index[++leaf_ptr];
    }
    __syncthreads();

    // Note that the heapId may be uninitialized at this location if we're
    // only descending one tree or it's a small/wide tree.
    // OpenCL: if (heapId[get_local_id(0)] < INT_MAX)
    //             heapDist[get_local_id(0)] = vecDistLoc(query, dataset, heapId[get_local_id(0)]*N_VECLEN);
    if (heap_ids[local_id] < INT_MAX) {
        const unsigned char* point = dataset + heap_ids[local_id] * padded_bytes;
        heap_dists[local_id] = compute_hamming_distance(query, point, actual_bytes);
    }

    // Main loop
    // OpenCL: do { ... } while (!(*locDone));
    do {
        // Set the leaf distances for the given IDs.
        // OpenCL: if (heapId[LOC_SIZE + get_local_id(0)] < INT_MAX)
        if (heap_ids[local_size + local_id] < INT_MAX) {
            const unsigned char* point = dataset + heap_ids[local_size + local_id] * padded_bytes;
            heap_dists[local_size + local_id] = compute_hamming_distance(query, point, actual_bytes);
        }

        // Sort heap to leave the closest distances in the bottom of the heap
        // OpenCL: sortHeap(heapDist, heapId);
        sort_heap(heap_dists, heap_ids, local_size * 2, local_id);

        // Init shared vars
        // OpenCL: if (get_local_id(0) == 0) { (*locPtr) = LOC_SIZE; (*locDone) = 1; }
        if (local_id == 0) {
            *loc_ptr = local_size;  // Pointer starts in the middle of the heap
            *done_flag = 1;         // Assume we're done until proven otherwise
        }
        __syncthreads();

        // Find the next dataset index to check
        // OpenCL: locI = 0; while (leafPtr < lastPtr && (locI = atomic_inc(locPtr)) < N_HEAP)
        loc_i = 0;
        while (leaf_ptr < last_ptr && (loc_i = atomicAdd(loc_ptr, 1)) < local_size * 2) {
            heap_ids[loc_i] = device_node_index[++leaf_ptr];
        }

        // We're done if there's no more leaf indices to search
        // OpenCL: checkDone(locI == 0, locDone);
        if (loc_i == 0) {
            // No more indices retrieved, keep done_flag = 1
        } else {
            *done_flag = 0;  // Got more indices, not done
        }
        __syncthreads();

    } while (*done_flag == 0);
}

/**
 * @brief Store results - remove duplicates and copy to global memory
 *
 * Matches OpenCL storeResult (nn_opencl_index.h:957-979)
 *
 * After leaf processing, the bottom of the heap contains the best k results.
 * Remove duplicates (if multiple trees), sort again, and copy to output.
 *
 * @param heap_dists Shared heap distances
 * @param heap_ids Shared heap IDs
 * @param result_dists Global output distances
 * @param result_ids Global output indices
 * @param k Number of neighbors to return
 * @param num_trees Number of parallel trees
 * @param local_id Thread's local ID
 * @param local_size Block size
 * @param query_id Query index
 */
template<int K>
__device__ inline void store_results(
    int* heap_dists,
    int* heap_ids,
    int* result_dists,
    int* result_ids,
    int num_trees,
    int local_id,
    int local_size,
    int query_id
) {
    // Get rid of duplicates if there are multiple trees searched
    // OpenCL: if (N_TREES > 1) { ... }
    if (num_trees > 1) {
        // OpenCL: int thisHeapId = heapId[get_local_id(0)];
        int this_heap_id = heap_ids[local_id];

        // OpenCL: for (int i = get_local_id(0)+1; i < N_HEAP; i++)
        //             if (heapId[i] == thisHeapId) heapDist[i] = MAX_DIST;
        // CRITICAL: Must match OpenCL exactly - NO extra condition
        for (int i = local_id + 1; i < local_size * 2; i++) {
            if (heap_ids[i] == this_heap_id) {
                heap_dists[i] = INT_MAX;  // Mark duplicate
            }
        }

        // CRITICAL FIX: Ensure all threads finish marking duplicates before sorting
        // Without this barrier, threads race: some enter sort_heap() while others
        // are still marking duplicates, causing heap corruption
        __syncthreads();

        // Send duplicates to the back of the heap
        // OpenCL: sortHeap(heapDist, heapId);
        sort_heap(heap_dists, heap_ids, local_size * 2, local_id);
    }

    // Copy from heap to global result mem
    // OpenCL: for (int i = get_local_id(0); i < N_RESULT; i += LOC_SIZE)
    int output_offset = query_id * K;
    for (int i = local_id; i < K; i += local_size) {
        result_dists[output_offset + i] = heap_dists[i];
        result_ids[output_offset + i] = heap_ids[i];
    }
}

/**
 * @brief Main cooperative hierarchical search kernel
 *
 * Implements hierarchical k-NN search using cooperative thread blocks.
 * Matches OpenCL implementation (nn_opencl_index.h:732-756).
 *
 * Kernel launch configuration:
 * - Grid: num_queries (one block per query)
 * - Block: 128 threads (cooperative workgroup)
 * - Shared memory: (128 * 2 * 2 + 2) * sizeof(int)
 *
 * @param tree_nodes Flat array of tree node structures
 * @param device_node_index Hybrid node index array
 * @param tree_pivots Flat array of pivot descriptors
 * @param dataset Full dataset of descriptors
 * @param queries Query descriptors
 * @param result_indices Output k-NN indices [num_queries x K]
 * @param result_distances Output k-NN distances [num_queries x K]
 * @param num_queries Number of queries
 * @param num_nodes Total number of tree nodes
 * @param num_trees Number of parallel trees
 * @param branching Branching factor
 * @param actual_bytes Actual descriptor length in bytes
 * @param padded_bytes Padded descriptor length (multiple of 4)
 */
template<int K>
__global__ void hierarchical_search_cooperative_kernel(
    const KMeansNodeGPU* __restrict__ tree_nodes,
    const int* __restrict__ device_node_index,
    const unsigned char* __restrict__ tree_pivots,
    const unsigned char* __restrict__ dataset,
    const unsigned char* __restrict__ queries,
    int* __restrict__ result_indices,
    int* __restrict__ result_distances,
    int num_queries,
    int num_nodes,
    int num_trees,
    int branching,
    int actual_bytes,
    int padded_bytes
) {
    // blockIdx.x = query ID (one block per query)
    // threadIdx.x = local thread ID within block
    int query_id = blockIdx.x;
    int local_id = threadIdx.x;
    int local_size = blockDim.x;  // Should be 128

    if (query_id >= num_queries) return;

    // Shared memory layout
    extern __shared__ int shared_mem[];
    int* heap_dists = shared_mem;                      // [local_size * 2]
    int* heap_ids = &shared_mem[local_size * 2];       // [local_size * 2]
    int* loc_ptr = &shared_mem[local_size * 4];        // [1]
    int* done_flag = &shared_mem[local_size * 4 + 1];  // [1]

    // Get query descriptor
    const unsigned char* query = queries + query_id * padded_bytes;

    // Phase 1: Find nodes (explore tree until heap contains only leaves)
    // OpenCL: findNodes(...)
    find_nodes_cooperative(
        tree_nodes, device_node_index, tree_pivots, query,
        heap_dists, heap_ids, done_flag,
        num_nodes, num_trees, branching,
        local_id, local_size,
        actual_bytes, padded_bytes
    );

    // Phase 2: Process leaves (fill heap with dataset points)
    // OpenCL: findLeaves(...)
    find_leaves_cooperative(
        device_node_index, query, dataset,
        heap_dists, heap_ids, loc_ptr, done_flag,
        local_id, local_size,
        actual_bytes, padded_bytes
    );

    // DEBUG: Print entire heap for query 0 to see all neighbors found
    if (query_id == 0 && local_id == 0) {
        printf("\n[CUDA HEAP DEBUG] Query 0 - Full heap contents (sorted by distance):\n");
        printf("  Top 20 neighbors:\n");
        for (int i = 0; i < 20 && i < local_size * 2; i++) {
            printf("    [%2d] ID=%6d dist=%3d\n", i, heap_ids[i], heap_dists[i]);
        }
        printf("  Looking for expected neighbors:\n");
        printf("    Expected: ID=53249 (dist should be 34)\n");
        printf("    Expected: ID=72652 (dist should be 37)\n");
        printf("    Expected: ID=74891 (dist should be 37)\n");
        printf("  Searching heap for these IDs...\n");
        for (int i = 0; i < local_size * 2; i++) {
            if (heap_ids[i] == 53249 || heap_ids[i] == 72652 || heap_ids[i] == 74891) {
                printf("    FOUND: ID=%6d at heap[%3d] with dist=%3d\n", heap_ids[i], i, heap_dists[i]);
            }
        }
    }
    __syncthreads();

    // Phase 3: Store results (remove duplicates, copy to global)
    // OpenCL: storeResult(...)
    store_results<K>(
        heap_dists, heap_ids,
        result_distances, result_indices,
        num_trees, local_id, local_size, query_id
    );
}

/**
 * @brief Launch cooperative hierarchical search kernel
 *
 * This function provides the kernel launch with appropriate configuration.
 * Unlike the single-threaded version, this uses one block per query
 * with cooperative threads within each block.
 *
 * @param dataset Dataset descriptors (GPU)
 * @param queries Query descriptors (GPU)
 * @param tree_nodes Tree structure (GPU)
 * @param tree_pivots Pivot descriptors (GPU)
 * @param device_node_index Hybrid node index array (GPU)
 * @param result_indices Output indices (GPU)
 * @param result_distances Output distances (GPU)
 * @param num_queries Number of queries
 * @param padded_bytes Padded descriptor length
 * @param actual_bytes Actual descriptor length
 * @param num_nodes Number of tree nodes
 * @param k Number of nearest neighbors
 * @param num_trees Number of parallel trees
 * @param branching Branching factor
 * @return true if kernel launched successfully
 */
bool launch_hierarchical_search_cooperative(
    const unsigned char* dataset,
    const unsigned char* queries,
    const KMeansNodeGPU* tree_nodes,
    const unsigned char* tree_pivots,
    const int* device_node_index,
    int* result_indices,
    int* result_distances,
    size_t num_queries,
    size_t padded_bytes,
    size_t actual_bytes,
    size_t num_nodes,
    int k,
    int num_trees,
    int branching
) {
    // Cooperative kernel launch configuration
    // TESTING: Increased from 128 to 256 to allow visiting more leaves
    // Hypothesis: 128 threads = max 128 leaves, but single-threaded visits 180 leaves
    const int local_size = 256;  // Was 128 (OpenCL LOC_SIZE)

    // Grid: one block per query (matching OpenCL one workgroup per query)
    dim3 grid(num_queries);
    dim3 block(local_size);

    // Calculate shared memory size
    // heap_dists[local_size * 2] + heap_ids[local_size * 2] + loc_ptr[1] + done_flag[1]
    size_t shared_mem_bytes =
        (local_size * 2) * sizeof(int) +  // heap_dists
        (local_size * 2) * sizeof(int) +  // heap_ids
        1 * sizeof(int) +                 // loc_ptr
        1 * sizeof(int);                  // done_flag

    // Dispatch based on k (common configurations)
    if (k == 1) {
        hierarchical_search_cooperative_kernel<1><<<grid, block, shared_mem_bytes>>>(
            tree_nodes, device_node_index, tree_pivots, dataset, queries,
            result_indices, result_distances,
            num_queries, num_nodes, num_trees, branching,
            actual_bytes, padded_bytes);
    } else if (k == 3) {
        hierarchical_search_cooperative_kernel<3><<<grid, block, shared_mem_bytes>>>(
            tree_nodes, device_node_index, tree_pivots, dataset, queries,
            result_indices, result_distances,
            num_queries, num_nodes, num_trees, branching,
            actual_bytes, padded_bytes);
    } else if (k == 5) {
        hierarchical_search_cooperative_kernel<5><<<grid, block, shared_mem_bytes>>>(
            tree_nodes, device_node_index, tree_pivots, dataset, queries,
            result_indices, result_distances,
            num_queries, num_nodes, num_trees, branching,
            actual_bytes, padded_bytes);
    } else if (k == 10) {
        hierarchical_search_cooperative_kernel<10><<<grid, block, shared_mem_bytes>>>(
            tree_nodes, device_node_index, tree_pivots, dataset, queries,
            result_indices, result_distances,
            num_queries, num_nodes, num_trees, branching,
            actual_bytes, padded_bytes);
    } else if (k == 20) {
        hierarchical_search_cooperative_kernel<20><<<grid, block, shared_mem_bytes>>>(
            tree_nodes, device_node_index, tree_pivots, dataset, queries,
            result_indices, result_distances,
            num_queries, num_nodes, num_trees, branching,
            actual_bytes, padded_bytes);
    } else if (k == 50) {
        hierarchical_search_cooperative_kernel<50><<<grid, block, shared_mem_bytes>>>(
            tree_nodes, device_node_index, tree_pivots, dataset, queries,
            result_indices, result_distances,
            num_queries, num_nodes, num_trees, branching,
            actual_bytes, padded_bytes);
    } else if (k == 100) {
        hierarchical_search_cooperative_kernel<100><<<grid, block, shared_mem_bytes>>>(
            tree_nodes, device_node_index, tree_pivots, dataset, queries,
            result_indices, result_distances,
            num_queries, num_nodes, num_trees, branching,
            actual_bytes, padded_bytes);
    } else {
        // Unsupported k
        return false;
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

} // namespace cuda
} // namespace flann

#endif  // FLANN_HIERARCHICAL_SEARCH_COOPERATIVE_CUH
