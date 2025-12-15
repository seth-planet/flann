/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024. All rights reserved.
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

#ifndef FLANN_CUDA_UTILITY_KERNELS_CUH_
#define FLANN_CUDA_UTILITY_KERNELS_CUH_

#ifdef FLANN_USE_CUDA

#include <cuda_runtime.h>

namespace flann {
namespace cuda {

/**
 * @brief Kernel to pad query vectors from veclen to padded_veclen
 *
 * Each thread block handles one query. Threads cooperate to copy
 * the original dimensions and fill padding with zeros.
 *
 * Launch config: grid=num_queries, block=256 (or padded_veclen if smaller)
 *
 * @tparam T Element type (float for K-Means, unsigned char for Hierarchical)
 * @param src Source queries [num_queries x veclen]
 * @param dst Destination queries [num_queries x padded_veclen]
 * @param num_queries Number of query vectors
 * @param veclen Original vector length
 * @param padded_veclen Padded vector length (must be >= veclen)
 */
template<typename T>
__global__ void pad_queries_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int num_queries,
    int veclen,
    int padded_veclen
) {
    int query_id = blockIdx.x;
    int local_id = threadIdx.x;
    int local_size = blockDim.x;

    if (query_id >= num_queries) return;

    // Source and destination pointers for this query
    const T* src_query = src + query_id * veclen;
    T* dst_query = dst + query_id * padded_veclen;

    // Each thread handles multiple dimensions (strided access)
    for (int d = local_id; d < padded_veclen; d += local_size) {
        T val = (d < veclen) ? src_query[d] : T(0);
        dst_query[d] = val;
    }
}

/**
 * @brief Launch the query padding kernel
 *
 * Pads queries from veclen to padded_veclen dimensions.
 * Extra dimensions are filled with zeros (which don't affect L2/Hamming distance).
 *
 * @tparam T Element type
 * @param src Raw queries on GPU [num_queries x veclen]
 * @param dst Padded queries on GPU [num_queries x padded_veclen]
 * @param num_queries Number of queries
 * @param veclen Original vector length
 * @param padded_veclen Padded vector length
 * @param stream CUDA stream (nullptr = default stream)
 * @return true on success, false on failure
 */
template<typename T>
bool launch_pad_queries(
    const T* src,
    T* dst,
    size_t num_queries,
    size_t veclen,
    size_t padded_veclen,
    cudaStream_t stream = nullptr
) {
    if (num_queries == 0) return true;
    if (padded_veclen < veclen) return false;

    // Skip if no padding needed
    if (veclen == padded_veclen) {
        cudaError_t err;
        if (stream) {
            err = cudaMemcpyAsync(dst, src, num_queries * veclen * sizeof(T),
                                  cudaMemcpyDeviceToDevice, stream);
        } else {
            err = cudaMemcpy(dst, src, num_queries * veclen * sizeof(T),
                            cudaMemcpyDeviceToDevice);
        }
        return (err == cudaSuccess);
    }

    // One block per query, 256 threads per block
    int block_size = 256;
    if (padded_veclen < 256) {
        // For small vectors, use fewer threads
        block_size = ((padded_veclen + 31) / 32) * 32;  // Round up to warp size
        if (block_size < 32) block_size = 32;
    }

    dim3 grid(num_queries);
    dim3 block(block_size);

    pad_queries_kernel<T><<<grid, block, 0, stream>>>(
        src, dst, num_queries, veclen, padded_veclen
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] pad_queries_kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    return true;
}

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA

#endif // FLANN_CUDA_UTILITY_KERNELS_CUH_
