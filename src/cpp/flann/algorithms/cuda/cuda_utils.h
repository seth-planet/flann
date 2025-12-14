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

#ifndef FLANN_CUDA_UTILS_H_
#define FLANN_CUDA_UTILS_H_

#ifdef FLANN_USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>  // std::move
#include <flann/general.h>

namespace flann {
namespace cuda {

/**
 * @brief Check CUDA runtime API call and throw on error
 *
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw flann::FLANNException( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + \
                cudaGetErrorString(err) + \
                " (error code: " + std::to_string(static_cast<int>(err)) + ")" \
            ); \
        } \
    } while (0)

/**
 * @brief Check for errors from last kernel launch
 *
 * Use after kernel invocation to catch launch configuration errors.
 * Must be called BEFORE cudaDeviceSynchronize() to get launch errors.
 *
 * Usage:
 *   my_kernel<<<grid, block>>>(args);
 *   CUDA_CHECK_LAST();  // Check launch errors
 *   CUDA_CHECK(cudaDeviceSynchronize());  // Check execution errors
 */
#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            throw flann::FLANNException( \
                std::string("CUDA kernel launch error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + \
                cudaGetErrorString(err) \
            ); \
        } \
    } while (0)

/**
 * @brief Check kernel execution errors (combines launch + sync)
 *
 * Usage:
 *   my_kernel<<<grid, block>>>(args);
 *   CUDA_CHECK_KERNEL();  // Checks launch AND execution
 */
#define CUDA_CHECK_KERNEL() \
    do { \
        CUDA_CHECK_LAST(); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

/**
 * @brief Synchronous error checking for debugging
 *
 * In debug builds, synchronize after every kernel to catch errors immediately.
 * In release builds, this is a no-op.
 */
#ifdef FLANN_CUDA_DEBUG
    #define CUDA_DEBUG_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
#else
    #define CUDA_DEBUG_SYNC() do {} while(0)
#endif

/**
 * @brief RAII wrapper for CUDA device memory
 *
 * Manages lifecycle of device memory allocations with automatic cleanup.
 * Supports move semantics but prohibits copying to prevent double-free.
 *
 * @tparam T Element type
 *
 * Example:
 * @code
 * CUDABuffer<float> device_data(num_elements);
 * device_data.upload(host_data, num_elements);
 * my_kernel<<<grid, block>>>(device_data.get(), num_elements);
 * device_data.download(host_data, num_elements);
 * // Automatic cleanup on scope exit
 * @endcode
 */
template<typename T>
class CUDABuffer {
public:
    /**
     * @brief Allocate device memory for 'count' elements
     * @param count Number of elements (not bytes)
     * @throws FLANNException if allocation fails
     */
    explicit CUDABuffer(size_t count = 0)
        : ptr_(nullptr), count_(count), size_bytes_(count * sizeof(T))
    {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, size_bytes_);
            if (err != cudaSuccess) {
                throw FLANNException(
                    std::string("CUDA allocation failed for ") +
                    std::to_string(size_bytes_) + " bytes: " +
                    cudaGetErrorString(err)
                );
            }
        }
    }

    /**
     * @brief Destructor: free device memory
     */
    ~CUDABuffer() {
        if (ptr_) {
            cudaFree(ptr_);  // Error ignored (destructor can't throw)
        }
    }

    // Disable copy (prevent double-free)
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    // Enable move
    CUDABuffer(CUDABuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), size_bytes_(other.size_bytes_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.size_bytes_ = 0;
    }

    CUDABuffer& operator=(CUDABuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            size_bytes_ = other.size_bytes_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.size_bytes_ = 0;
        }
        return *this;
    }

    /**
     * @brief Upload data from host to device
     * @param host_data Host memory pointer
     * @param count Number of elements to upload
     * @param stream CUDA stream for async operation (nullptr = sync)
     */
    void upload(const T* host_data, size_t count, cudaStream_t stream = nullptr) {
        if (count > count_) {
            throw FLANNException("Upload count exceeds buffer size");
        }
        cudaError_t err;
        if (stream) {
            err = cudaMemcpyAsync(ptr_, host_data, count * sizeof(T),
                                  cudaMemcpyHostToDevice, stream);
        } else {
            err = cudaMemcpy(ptr_, host_data, count * sizeof(T),
                            cudaMemcpyHostToDevice);
        }
        if (err != cudaSuccess) {
            throw FLANNException(
                std::string("CUDA upload failed: ") + cudaGetErrorString(err)
            );
        }
    }

    /**
     * @brief Download data from device to host
     * @param host_data Host memory pointer (must have space for 'count' elements)
     * @param count Number of elements to download
     * @param stream CUDA stream for async operation (nullptr = sync)
     */
    void download(T* host_data, size_t count, cudaStream_t stream = nullptr) const {
        if (count > count_) {
            throw FLANNException("Download count exceeds buffer size");
        }
        cudaError_t err;
        if (stream) {
            err = cudaMemcpyAsync(host_data, ptr_, count * sizeof(T),
                                  cudaMemcpyDeviceToHost, stream);
        } else {
            err = cudaMemcpy(host_data, ptr_, count * sizeof(T),
                            cudaMemcpyDeviceToHost);
        }
        if (err != cudaSuccess) {
            throw FLANNException(
                std::string("CUDA download failed: ") + cudaGetErrorString(err)
            );
        }
    }

    /**
     * @brief Fill buffer with constant value
     * @param value Value to fill (type T)
     * @param count Number of elements to fill
     */
    void fill(const T& value, size_t count) {
        if (count > count_) {
            throw FLANNException("Fill count exceeds buffer size");
        }
        // For byte-sized types, use cudaMemset
        if (sizeof(T) == 1) {
            CUDA_CHECK(cudaMemset(ptr_, *reinterpret_cast<const unsigned char*>(&value), count));
        } else {
            // For other types, upload repeated pattern
            std::vector<T> pattern(count, value);
            upload(pattern.data(), count);
        }
    }

    /**
     * @brief Resize buffer (may reallocate)
     * @param new_count New number of elements
     * @throws FLANNException if allocation fails
     * @warning Existing data is NOT preserved
     */
    void resize(size_t new_count) {
        if (new_count == count_) return;

        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = new_count;
        size_bytes_ = new_count * sizeof(T);

        if (new_count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, size_bytes_);
            if (err != cudaSuccess) {
                count_ = 0;
                size_bytes_ = 0;
                throw FLANNException(
                    std::string("CUDA reallocation failed: ") +
                    cudaGetErrorString(err)
                );
            }
        }
    }

    // Accessors
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t size_bytes() const { return size_bytes_; }
    bool empty() const { return count_ == 0; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_;
    size_t count_;
    size_t size_bytes_;
};

// Note: PinnedBuffer was originally removed (commit 2a48cd5) because A/B testing
// showed pinned memory provides no benefit for small transfers (<1MB).
// However, for GPU index loading, dataset transfers are 10-200MB and benefit
// significantly from DMA acceleration via pinned memory.

/**
 * @brief Align offset to 16-byte boundary for CUDA memory coalescing
 *
 * CUDA memory accesses perform best when aligned to 16 bytes.
 * Used when packing multiple arrays into a single GPU buffer.
 *
 * @param offset Current byte offset
 * @return Aligned offset (rounded up to next 16-byte boundary)
 */
inline size_t alignTo16(size_t offset) {
    return ((offset + 15) / 16) * 16;
}

/**
 * @brief RAII wrapper for pinned (page-locked) host memory
 *
 * Pinned memory enables faster DMA transfers to GPU by allowing
 * direct memory access without staging through pageable memory.
 *
 * Key characteristics:
 * - 1.5-2x faster cudaMemcpy for large transfers (>1MB)
 * - Automatic fallback to regular malloc if cudaMallocHost fails
 * - Move-only semantics (like CUDABuffer)
 * - Best used for staging buffers that flow directly to GPU
 *
 * @tparam T Element type
 *
 * @code
 * // Allocate pinned buffer for GPU upload
 * PinnedBuffer<float> staging(dataset_size);
 *
 * // Fill with data
 * std::memcpy(staging.get(), source_data, dataset_size * sizeof(float));
 *
 * // Upload to GPU (faster with pinned memory)
 * gpu_buffer.upload(staging.get(), dataset_size);
 *
 * // Check if pinned allocation succeeded (for debugging)
 * if (staging.is_pinned()) {
 *     std::cerr << "Using pinned memory for transfer" << std::endl;
 * }
 * @endcode
 */
template<typename T>
class PinnedBuffer {
public:
    /**
     * @brief Allocate pinned host memory for 'count' elements
     *
     * Falls back to regular malloc if cudaMallocHost fails.
     *
     * @param count Number of elements (not bytes)
     * @throws FLANNException if both pinned and regular allocation fail
     */
    explicit PinnedBuffer(size_t count = 0)
        : ptr_(nullptr), count_(count), is_pinned_(false)
    {
        if (count == 0) return;

        cudaError_t err = cudaMallocHost(&ptr_, count * sizeof(T));
        if (err == cudaSuccess) {
            is_pinned_ = true;
        } else {
            // Fallback to regular allocation (pageable memory)
            ptr_ = static_cast<T*>(malloc(count * sizeof(T)));
            if (!ptr_) {
                throw FLANNException(
                    std::string("Failed to allocate pinned/regular buffer for ") +
                    std::to_string(count * sizeof(T)) + " bytes"
                );
            }
            is_pinned_ = false;
        }
    }

    /**
     * @brief Destructor: free pinned or regular memory
     */
    ~PinnedBuffer() {
        if (ptr_) {
            if (is_pinned_) {
                cudaFreeHost(ptr_);  // Error ignored (destructor can't throw)
            } else {
                free(ptr_);
            }
        }
    }

    // Disable copy (prevent double-free)
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Enable move
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), is_pinned_(other.is_pinned_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.is_pinned_ = false;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                if (is_pinned_) cudaFreeHost(ptr_);
                else free(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            is_pinned_ = other.is_pinned_;
            other.ptr_ = nullptr;
            other.count_ = 0;
            other.is_pinned_ = false;
        }
        return *this;
    }

    // Accessors
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    T& operator[](size_t index) { return ptr_[index]; }
    const T& operator[](size_t index) const { return ptr_[index]; }
    size_t count() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }
    bool empty() const { return count_ == 0; }
    bool is_pinned() const { return is_pinned_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_;
    size_t count_;
    bool is_pinned_;
};

/**
 * @brief Query CUDA device capabilities for cooperative kernel
 *
 * Determines optimal LOC_SIZE (threads per block) for cooperative kernels.
 * This is a precision-vs-speed tradeoff parameter.
 *
 * DESIGN DECISION: LOC_SIZE=128 chosen for high precision (97.4%)
 * This differs from OpenCL (LOC_SIZE=32, 87.7% precision) because
 * CUDA implementation prioritizes accuracy over raw speed.
 *
 * LOC_SIZE benchmark results (SIFT100K, k=5):
 *   32:  87.7% precision,  5.91 µs/query (fastest, matches OpenCL)
 *   64:  93.4% precision,  7.74 µs/query
 *   128: 97.4% precision, 14.73 µs/query (DEFAULT - good balance)
 *   256: 99.3% precision, 16.57 µs/query
 *   512: 100%  precision, 30.78 µs/query (highest precision)
 *   1024: FAILED (exceeds GPU resource limits)
 *
 * @param device_id CUDA device ID (default: 0)
 * @return LOC_SIZE value (currently fixed at 128)
 */
inline int getCUDALocSize(int device_id = 0)
{
    (void)device_id;  // Reserved for future device-specific tuning
    // LOC_SIZE=128 provides 97.4% precision at acceptable latency.
    // For speed-critical applications needing OpenCL-like performance,
    // change to 32 (87.7% precision, ~2.5x faster).
    return 128;
}

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_CUDA_UTILS_H_
