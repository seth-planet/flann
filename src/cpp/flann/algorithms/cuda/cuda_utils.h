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
#include <cstdio>   // For fprintf/stderr
#include <cstring>  // For std::memcpy
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
            throw ::flann::FLANNException( \
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
            throw ::flann::FLANNException( \
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
     * @brief Allocate device memory using stream-ordered allocation (non-blocking)
     *
     * Uses cudaMallocAsync to avoid CPU-GPU synchronization stalls from cudaMalloc.
     * Memory is allocated from the device's memory pool and can be reused across
     * calls without OS allocation overhead.
     *
     * @param count Number of elements (not bytes)
     * @param stream CUDA stream for ordered allocation
     * @throws FLANNException if allocation fails
     *
     * @note Use freeAsync(stream) for explicit deallocation. The destructor
     *       falls back to sync cudaFree as a safety net (e.g., exception unwinding).
     */
    explicit CUDABuffer(size_t count, cudaStream_t stream)
        : ptr_(nullptr), count_(count), size_bytes_(count * sizeof(T))
    {
        if (count > 0) {
            cudaError_t err = cudaMallocAsync(&ptr_, size_bytes_, stream);
            if (err != cudaSuccess) {
                throw FLANNException(
                    std::string("CUDA async allocation failed for ") +
                    std::to_string(size_bytes_) + " bytes: " +
                    cudaGetErrorString(err));
            }
        }
    }

    /**
     * @brief Destructor: free device memory
     * Note: Logs errors to stderr since destructors cannot throw.
     */
    ~CUDABuffer() {
        if (ptr_) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                fprintf(stderr, "WARNING: cudaFree failed in CUDABuffer destructor: %s\n",
                        cudaGetErrorString(err));
            }
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
            if (ptr_) {
                cudaFree(ptr_);
            }
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
            // Use memcpy to avoid strict aliasing violation (UB via reinterpret_cast)
            unsigned char byte_val;
            std::memcpy(&byte_val, &value, 1);
            CUDA_CHECK(cudaMemset(ptr_, byte_val, count));
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

        if (ptr_) {
            cudaFree(ptr_);
        }
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

    /**
     * @brief Free device memory using stream-ordered deallocation (non-blocking)
     *
     * Queues the deallocation on the stream. Memory returns to the pool for reuse
     * by subsequent cudaMallocAsync calls. Call this BEFORE cudaStreamSynchronize
     * to follow the NVIDIA recommended pattern: alloc → use → freeAsync → sync.
     *
     * Nullifies the pointer to prevent double-free in the destructor.
     * If not called, the destructor falls back to sync cudaFree (safe for both
     * sync and async allocations per CUDA spec).
     *
     * @param stream CUDA stream for ordered deallocation
     */
    void freeAsync(cudaStream_t stream) {
        if (ptr_) {
            cudaError_t err = cudaFreeAsync(ptr_, stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "WARNING: cudaFreeAsync failed: %s\n",
                        cudaGetErrorString(err));
            }
            ptr_ = nullptr;
            count_ = 0;
            size_bytes_ = 0;
        }
    }

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
     * Note: Logs errors to stderr since destructors cannot throw
     */
    ~PinnedBuffer() {
        if (ptr_) {
            if (is_pinned_) {
                cudaError_t err = cudaFreeHost(ptr_);
                if (err != cudaSuccess) {
                    fprintf(stderr, "WARNING: cudaFreeHost failed in PinnedBuffer destructor: %s\n",
                            cudaGetErrorString(err));
                }
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

/**
 * @brief RAII wrapper for CUDA streams
 *
 * Manages lifecycle of CUDA streams with automatic cleanup.
 * Supports move semantics but prohibits copying.
 *
 * Intended for use with ThreadLocalStreamPool to provide
 * per-thread streams for concurrent GPU operations.
 *
 * @code
 * CUDAStream stream(cudaStreamNonBlocking);
 * kernel<<<grid, block, 0, stream.get()>>>(args);
 * stream.synchronize();
 * // Automatic cleanup on scope exit
 * @endcode
 */
class CUDAStream {
public:
    /**
     * @brief Create a new CUDA stream
     * @param flags Stream creation flags (default: cudaStreamNonBlocking)
     * @throws FLANNException if stream creation fails
     */
    explicit CUDAStream(unsigned int flags = cudaStreamNonBlocking)
        : stream_(nullptr)
    {
        cudaError_t err = cudaStreamCreateWithFlags(&stream_, flags);
        if (err != cudaSuccess) {
            throw FLANNException(
                std::string("Failed to create CUDA stream: ") +
                cudaGetErrorString(err)
            );
        }
    }

    /**
     * @brief Destructor: destroy the stream
     * Note: Logs errors to stderr since destructors cannot throw
     */
    ~CUDAStream() {
        if (stream_) {
            cudaError_t err = cudaStreamDestroy(stream_);
            if (err != cudaSuccess) {
                fprintf(stderr, "WARNING: cudaStreamDestroy failed in CUDAStream destructor: %s\n",
                        cudaGetErrorString(err));
            }
        }
    }

    // Disable copy (prevent double-destroy)
    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;

    // Enable move
    CUDAStream(CUDAStream&& other) noexcept
        : stream_(other.stream_)
    {
        other.stream_ = nullptr;
    }

    CUDAStream& operator=(CUDAStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the raw CUDA stream handle
     * @return cudaStream_t handle (may be nullptr if moved-from)
     */
    cudaStream_t get() const { return stream_; }

    /**
     * @brief Synchronize on this stream
     * @throws FLANNException if synchronization fails
     */
    void synchronize() const {
        if (stream_) {
            cudaError_t err = cudaStreamSynchronize(stream_);
            if (err != cudaSuccess) {
                throw FLANNException(
                    std::string("CUDA stream synchronization failed: ") +
                    cudaGetErrorString(err)
                );
            }
        }
    }

    /**
     * @brief Check if stream is valid
     */
    explicit operator bool() const { return stream_ != nullptr; }

private:
    cudaStream_t stream_;
};

/**
 * @brief Thread-local CUDA stream pool
 *
 * Provides per-thread CUDA streams for concurrent GPU operations.
 * Each thread gets its own stream, created on first access and
 * destroyed when the thread terminates.
 *
 * Benefits:
 * - No stream contention between threads
 * - Full GPU parallelism for concurrent searches
 * - Stream creation cost amortized (only on first call per thread)
 *
 * Usage:
 * @code
 * cudaStream_t stream = ThreadLocalStreamPool::getStream();
 * kernel<<<grid, block, 0, stream>>>(args);
 * cudaStreamSynchronize(stream);
 * @endcode
 */
class ThreadLocalStreamPool {
public:
    /**
     * @brief Get the CUDA stream for the current thread
     *
     * Creates a new stream on first call per thread.
     * Subsequent calls return the same stream.
     *
     * @return cudaStream_t handle for thread-local stream
     */
    static cudaStream_t getStream() {
        // Thread-local stream, created once per thread, destroyed on thread exit
        thread_local CUDAStream stream(cudaStreamNonBlocking);
        return stream.get();
    }
};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_CUDA_UTILS_H_
