/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024 (CUDA kernel compilation unit)
 *
 * THE BSD LICENSE
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

/**
 * @file kmeans_cuda_kernels.cu
 * @brief CUDA kernel compilation unit for K-Means search
 *
 * This file serves as the CUDA compilation unit that compiles the
 * K-Means cooperative search kernel. The kernel implementation is in
 * kmeans_search_cooperative.cuh, included here and compiled by nvcc.
 *
 * This pattern separates:
 * - .cuh files: Device code (kernels) - included in both CPU and GPU compilation
 * - .cu files: CUDA compilation units - compiled only by nvcc
 * - .h files: Host-side declarations - compiled by C++ compiler
 */

// Note: FLANN_USE_CUDA is defined via CMake target_compile_definitions

// Include the cooperative kernel implementation (production kernel)
#include "kernels/kmeans_search_cooperative.cuh"

// Explicit template instantiations for cooperative kernel launcher
// Required because the launcher is called from host code via template dispatch
namespace flann {
namespace cuda {

// Instantiate for common k values (OpenCL parity: unified nodeIndex array)
// Supported k values: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100
template bool launch_kmeans_search_cooperative<1>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<2>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<4>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<5>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<7>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<8>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<10>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<16>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<20>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<32>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<50>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<64>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

template bool launch_kmeans_search_cooperative<100>(
    const float*, const float*, const int*, const float*, const float*,
    int*, float*, size_t, size_t, size_t, int, int, int, float);

} // namespace cuda
} // namespace flann
