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
 * @file hierarchical_cuda_kernels.cu
 * @brief CUDA kernel compilation unit for Hierarchical search
 *
 * This file serves as the CUDA compilation unit that compiles the
 * Hierarchical search kernels. The actual kernel implementations are in
 * hierarchical_search_kernel.cuh, which is included here and compiled by nvcc.
 *
 * This pattern separates:
 * - .cuh files: Device code (kernels) - included in both CPU and GPU compilation
 * - .cu files: CUDA compilation units - compiled only by nvcc
 * - .h files: Host-side declarations - compiled by C++ compiler
 */

#define FLANN_USE_CUDA

// Include the kernel implementations
#include "kernels/hierarchical_search_kernel.cuh"
#include "kernels/hierarchical_search_cooperative.cuh"

// This .cu file exists purely to trigger CUDA compilation of the kernels.
// The launch_hierarchical_search() function is inline in the .cuh header and
// will be instantiated when this file is compiled by nvcc.

// No additional code needed - the kernel templates are instantiated
// on-demand when launch_hierarchical_search() calls them with specific k and
// max_checks values.
