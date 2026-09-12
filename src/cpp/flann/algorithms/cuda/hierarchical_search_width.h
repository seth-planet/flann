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


#ifndef FLANN_HIERARCHICAL_SEARCH_WIDTH_H_
#define FLANN_HIERARCHICAL_SEARCH_WIDTH_H_

/**
 * @file hierarchical_search_width.h
 * @brief The block width the cooperative hierarchical k-NN search runs at, and the
 *        bounds a width has to satisfy.
 *
 * Host-side and free of CUDA, so both sides of the search can include it:
 * hierarchical_cuda_index.h declares the launch and is compiled by the host compiler, and
 * kernels/hierarchical_search_cooperative.cuh defines it and is compiled by nvcc. It is a
 * header of its own rather than an addition to cuda_utils.h because that file is nearly
 * 700 lines of host machinery -- CUDABuffer, PinnedBuffer, CUDAStream, a thread-local
 * stream pool -- and the kernel translation unit needs none of it to state a bound.
 *
 */

namespace flann {
namespace cuda {

/// Block width the search runs at when a caller states no preference.
///
/// 128 is the width the CUDA port inherited by copying a sibling kmeans kernel's value,
/// and it is the default so a caller who states no width searches at the width every
/// pipeline measurement was taken at. SearchParams::checks is accepted and ignored on
/// this path; cuda_utils.h's getCUDALocSize is the kmeans width and returns the same 128,
/// and the 97.4% precision figure recorded beside it is a kmeans measurement.
///
/// The OpenCL implementation this kernel was translated from carries FLANN's
/// SearchParams(checks) accuracy/speed dial in its workgroup size, under the name
/// LOC_SIZE -- "avg checks per node needed for MAX_CHECKS", whose only stated lower bound
/// is LOC_SIZE >= the neighbour count. Two kernels with different heaps, different barrier
/// counts and different bounds need separate dials, which is why this one is not
/// getCUDALocSize.
inline constexpr int kDefaultSearchWidth = 128;

/**
 * @brief Why a block width cannot run this kernel, or nullptr if it can.
 *
 * The width is refused rather than clamped because a clamp would silently answer a
 * different search than the caller asked for, and because each bound below fails
 * quietly or not at all:
 *
 * - Below @p branching the kernel HANGS. find_new_node_dist walks the heap in groups of
 *   `branching` consecutive threads, one group per heap entry, advancing by
 *   `loc_id += local_size / branching`; below branching that step truncates to zero and
 *   the while loop around it cannot advance.
 * - A width that is not a MULTIPLE of @p branching is the quiet half of the same
 *   arithmetic. The group count is ceil(local_size / branching) while the step is the
 *   floor, so the last group is a short one and the groups either side of that
 *   disagreement rescan each other's heap positions. At width 128 with branching 48 the
 *   last group holds 32 threads, expands 32 of its node's 48 children and never visits
 *   the other 16: fewer candidates, no error. Costless for every width shipped so far,
 *   since 32 divides each power of two from 32 to 1024, and reachable today through
 *   HierarchicalCUDAIndexParams' branching argument or the branching loadIndexV2 restores
 *   from a saved index.
 * - Below @p k, store_results' duplicate scan probes with heap_ids[local_id] over
 *   thread ids alone, so heap entries past the width are never used as a probe and
 *   duplicates from the overlapping trees survive into the returned neighbours. Past
 *   twice the width it is a shared-memory overrun instead: store_results' copy-out loop
 *   runs to k over a heap of local_size * 2 entries, so width 32 with k 128 reads
 *   heap_ids[127], which starts 244 bytes past the end of the 130-int allocation and in
 *   range of no fault.
 * - Below @p num_trees, find_nodes_cooperative seeds one root per thread under
 *   `if (local_id < num_trees)`, so the roots past the width never enter the heap and
 *   those trees are never descended -- fewer candidates, no error. Unreachable while
 *   branching (32) exceeds the tree count (4), and checked because the floor is
 *   max(branching, k, num_trees) rather than any one of them.
 * - A non-power-of-two HANGS, which is the second hang mode rather than a sorting one.
 *   sort_heap doubles `size` from 2 while `size < heap_size`, so a heap whose width is not
 *   a power of two is never reached by the doubling loop; bitonic_merge then partners with
 *   `local_id & (stride - 1)` over a sequence that was never made bitonic, and `pos +
 *   stride` runs past heap_dists into the heap_ids half of the same shared allocation --
 *   in range of the allocation, so no fault to report. Measured by removing this check and
 *   launching width 100: 100% GPU utilisation, no progress, killed at 150 s.
 * - Above the device limit the launch fails, which reads as a driver problem rather
 *   than as the configuration mistake it is. 1024 is admitted and sits on two sm_75
 *   limits at once: 16 KB of a block's 48 KB of shared memory, and 64 registers a thread
 *   for 65536, exactly the per-block register file. Measured on a T4 at 465.136 ms
 *   against 38.068 at width 128 in the same interleaved pass, returning the true
 *   neighbour for 8000 of 8000 queries against 7957.
 *
 * @param local_size Requested block width
 * @param k          Neighbours the caller asked for
 * @param branching  Branching factor of the index being searched, which must be positive
 *                   because the width is checked against it by division
 * @param num_trees  Trees in the index, which the root seeding needs a thread each for
 * @return nullptr when the width is usable, otherwise a literal naming the bound it broke
 */
inline const char* hierarchical_search_width_error(int local_size, int k, int branching,
                                                   int num_trees)
{
    if (local_size <= 0) {
        return "search width must be positive";
    }
    if ((local_size & (local_size - 1)) != 0) {
        return "search width must be a power of two (the heap is sorted by a bitonic network)";
    }
    if (local_size > 1024) {
        return "search width exceeds the 1024-thread CUDA block limit";
    }
    if (branching <= 0) {
        return "index branching factor is not positive (the width is checked against it "
               "by division)";
    }
    if (local_size < branching) {
        return "search width is below the index's branching factor (the kernel would hang)";
    }
    if (local_size % branching != 0) {
        return "search width is not a multiple of the index's branching factor (the last "
               "thread group would expand part of its node's children)";
    }
    if (local_size < k) {
        return "search width is below k (duplicate neighbours would survive the scan)";
    }
    if (local_size < num_trees) {
        return "search width is below the index's tree count (trees past the width are "
               "never descended)";
    }
    return nullptr;
}

}  // namespace cuda
}  // namespace flann

#endif  // FLANN_HIERARCHICAL_SEARCH_WIDTH_H_
