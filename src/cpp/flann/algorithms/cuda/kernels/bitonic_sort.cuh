#ifndef FLANN_BITONIC_SORT_CUH
#define FLANN_BITONIC_SORT_CUH

namespace flann {
namespace cuda {

/**
 * @file bitonic_sort.cuh
 * @brief Parallel bitonic sort for shared memory arrays
 *
 * Implements parallel bitonic sort matching OpenCL implementation
 * in nn_opencl_index.h lines 902-933.
 *
 * Used for maintaining sorted heap in cooperative hierarchical search.
 */

/**
 * @brief Bitonic merge operation
 *
 * Merges a bitonic sequence into a sorted sequence.
 * Matches OpenCL bitonicMerge (nn_opencl_index.h:914-933)
 *
 * @param heap_dists Shared memory array of distances
 * @param heap_ids Shared memory array of IDs
 * @param size Size of the sequence to merge
 * @param dir Direction: 0 = ascending, 1 = descending
 * @param local_id Thread's local ID within block
 */
__device__ inline void bitonic_merge(
    int* heap_dists,
    int* heap_ids,
    int size,
    int dir,
    int local_id
) {
    for (int stride = size / 2; stride > 0; stride >>= 1) {
        __syncthreads();

        // Calculate position for this thread
        // OpenCL: int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
        int pos = 2 * local_id - (local_id & (stride - 1));

        // CRITICAL FIX: OpenCL has NO bounds check here!
        // The heap is always allocated with full capacity (local_size * 2),
        // so all memory accesses are valid. The bitonic sort algorithm requires
        // ALL threads to participate in every compare-and-swap operation.
        // Previous bounds check "if (pos + stride < size)" was WRONG and broke
        // the algorithm by excluding threads during intermediate merges.

        int keyA = heap_dists[pos];
        int keyB = heap_dists[pos + stride];
        int valA = heap_ids[pos];
        int valB = heap_ids[pos + stride];

        // Match OpenCL behavior: simple comparison without tie-breaking
        // OpenCL: if ((keyA < keyB) == dir) { swap }
        // Note: valA, valB loaded but not used for tie-breaking (matches OpenCL)
        (void)valA;  // Suppress unused variable warning
        (void)valB;

        if ((keyA < keyB) == dir) {
            // Swap distances
            heap_dists[pos] = keyB;
            heap_dists[pos + stride] = keyA;

            // Swap IDs
            heap_ids[pos] = valB;
            heap_ids[pos + stride] = valA;
        }
    }
}

/**
 * @brief Sort shared heap using parallel bitonic sort
 *
 * Sorts heap by ascending distance using bitonic sort algorithm.
 * Matches OpenCL sortHeap (nn_opencl_index.h:902-912)
 *
 * The heap is sorted in-place with all threads cooperating.
 * After sorting, smallest distances are at the beginning.
 *
 * @param heap_dists Shared memory array of distances to sort
 * @param heap_ids Shared memory array of IDs to sort (parallel to dists)
 * @param heap_size Total size of heap to sort
 * @param local_id Thread's local ID within block
 */
__device__ inline void sort_heap(
    int* heap_dists,
    int* heap_ids,
    int heap_size,
    int local_id
) {
    // Iterate from small sizes to N_HEAP
    // OpenCL: for (int size = 2; size < N_HEAP; size <<= 1)
    for (int size = 2; size < heap_size; size <<= 1) {
        // Bitonic merge
        // OpenCL: int ddd = (get_local_id(0) & (size / 2)) != 0;
        int ddd = (local_id & (size / 2)) != 0;
        bitonic_merge(heap_dists, heap_ids, size, ddd, local_id);
    }

    // Final merge with direction 0 (ascending)
    // OpenCL: bitonicMerge(heapDist, heapId, N_HEAP, 0);
    bitonic_merge(heap_dists, heap_ids, heap_size, 0, local_id);

    // Synchronize all threads
    // OpenCL: barrier(CLK_LOCAL_MEM_FENCE);
    __syncthreads();
}

} // namespace cuda
} // namespace flann

#endif  // FLANN_BITONIC_SORT_CUH
