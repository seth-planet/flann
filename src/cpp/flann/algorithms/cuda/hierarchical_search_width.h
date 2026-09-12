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
 * hierarchical_cuda_index.h declares the launch and is compiled by the host compiler,
 * kernels/hierarchical_search_cooperative.cuh defines it and is compiled by nvcc, and the
 * nvcc translation unit includes neither the index header nor cuda_utils.h.
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
static const int kDefaultSearchWidth = 128;

/**
 * @brief Why a block width cannot run this kernel, or nullptr if it can.
 *
 * The width is refused rather than clamped because a clamp would silently answer a
 * different search than the caller asked for, and because each bound below fails
 * quietly or not at all:
 *
 * - Below @p branching, `loc_id += local_size / branching` in find_nodes_cooperative
 *   truncates to zero and the while loop around it cannot advance: the kernel HANGS.
 * - Below @p k, store_results' duplicate scan probes with heap_ids[local_id] over
 *   thread ids alone, so heap entries past the width are never used as a probe and
 *   duplicates from the overlapping trees survive into the returned neighbours.
 * - Below @p num_trees, find_nodes_cooperative seeds one root per thread under
 *   `if (local_id < num_trees)`, so the roots past the width never enter the heap and
 *   those trees are never descended -- fewer candidates, no error. Unreachable while
 *   branching (32) exceeds the tree count (4), and checked because the floor is
 *   max(branching, k, num_trees) rather than any one of them.
 * - A non-power-of-two sorts wrong rather than failing: bitonic_merge steps
 *   `stride >>= 1` from size/2 over a heap of local_size*2, and a bitonic network only
 *   sorts when its width is a power of two.
 * - Above the device limit the launch fails, which reads as a driver problem rather
 *   than as the configuration mistake it is. 1024 is admitted: measured on a T4 at
 *   464.5 ms against 37.9 at 128, returning the true neighbour for 8000 of 8000 queries.
 *   cuda_utils.h's getCUDALocSize records "1024: FAILED (exceeds GPU resource limits)",
 *   which is the kmeans kernel's limit -- this kernel's shared memory is
 *   (width * 4 + 2) ints, 16 KB at width 1024 and well inside a block's 48 KB.
 *
 * @param local_size Requested block width
 * @param k          Neighbours the caller asked for
 * @param branching  Branching factor of the index being searched
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
    if (local_size < branching) {
        return "search width is below the index's branching factor (the kernel would hang)";
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
