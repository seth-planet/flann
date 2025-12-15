/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024 (CUDA implementation)
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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

#ifndef FLANN_NN_CUDA_INDEX_H_
#define FLANN_NN_CUDA_INDEX_H_

#include <cuda_runtime.h>
#include <string>
#include <flann/general.h>

namespace flann {
namespace cuda {

/**
 * Base class for CUDA-accelerated indices.
 *
 * Provides common infrastructure for CUDA indices, including
 * device management and stream handling. Uses RAII pattern
 * (via CUDABuffer) for all GPU resource management.
 *
 * This class follows the same pattern as OpenCLIndex to enable
 * dual inheritance for seamless CPU→GPU index conversion.
 */
class CUDAIndex
{
public:
    CUDAIndex()
        : cuda_device_(-1)
        , cuda_stream_(nullptr)
    {
    }

    virtual ~CUDAIndex()
    {
        // CUDA resources are managed via RAII (CUDABuffer)
        // No explicit cleanup needed here
        if (cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
            cuda_stream_ = nullptr;
        }
    }

    /**
     * Get the CUDA device ID being used (-1 if not initialized)
     */
    int getCUDADevice() const {
        return cuda_device_;
    }

    /**
     * Get the CUDA stream being used (nullptr if default stream)
     */
    cudaStream_t getCUDAStream() const {
        return cuda_stream_;
    }

protected:
    /**
     * Set the CUDA device to use for this index
     * @param device Device ID (0-based)
     * @throws FLANNException if device selection fails
     */
    void setCUDADevice(int device) {
        if (device != cuda_device_) {
            cudaError_t err = cudaSetDevice(device);
            if (err != cudaSuccess) {
                throw FLANNException(
                    std::string("Failed to set CUDA device ") + std::to_string(device) +
                    ": " + cudaGetErrorString(err)
                );
            }
            cuda_device_ = device;
        }
    }

    /**
     * Create a CUDA stream for this index
     */
    void createCUDAStream() {
        if (!cuda_stream_) {
            cudaStreamCreate(&cuda_stream_);
        }
    }

private:
    int cuda_device_;           ///< CUDA device ID (-1 = not set)
    cudaStream_t cuda_stream_;  ///< CUDA stream (nullptr = default stream)
};

} // namespace cuda
} // namespace flann

#endif // FLANN_NN_CUDA_INDEX_H_
