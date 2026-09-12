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
 * Host-side and free of CUDA, because the two sides of the search share nothing else:
 * hierarchical_cuda_index.h declares the launch and is compiled by the host compiler,
 * kernels/hierarchical_search_cooperative.cuh defines it and is compiled by nvcc.
 */

namespace flann {
namespace cuda {

/// Block width the search runs at when a caller states no preference.
///
/// The OpenCL implementation this kernel was translated from carried FLANN's
/// SearchParams(checks) accuracy/speed dial in its workgroup size, under the name
/// LOC_SIZE -- "avg checks per node needed for MAX_CHECKS", whose only stated lower
/// bound was LOC_SIZE >= the neighbour count. The CUDA port set the width to match a
/// sibling kmeans kernel and dropped the connection, so the dial became unreachable
/// and SearchParams::checks has been accepted and ignored since. 128 is that inherited
/// value, kept as the default so a caller who states no width gets the same neighbours
/// as before.
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
 * - A non-power-of-two sorts wrong rather than failing: bitonic_merge steps
 *   `stride >>= 1` from size/2 over a heap of local_size*2, and a bitonic network only
 *   sorts when its width is a power of two.
 * - Above the device limit the launch fails, which reads as a driver problem rather
 *   than as the configuration mistake it is.
 *
 * @param local_size Requested block width
 * @param k          Neighbours the caller asked for
 * @param branching  Branching factor of the index being searched
 * @return nullptr when the width is usable, otherwise a literal naming the bound it broke
 */
inline const char* hierarchical_search_width_error(int local_size, int k, int branching)
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
    return nullptr;
}

}  // namespace cuda
}  // namespace flann

#endif  // FLANN_HIERARCHICAL_SEARCH_WIDTH_H_
