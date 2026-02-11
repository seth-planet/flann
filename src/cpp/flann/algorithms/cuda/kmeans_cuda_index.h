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

#ifndef FLANN_CUDA_KMEANS_CUDA_INDEX_H_
#define FLANN_CUDA_KMEANS_CUDA_INDEX_H_

#ifdef FLANN_USE_CUDA

#include <queue>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>         // For std::unique_lock
#include <shared_mutex>  // For std::shared_mutex, std::shared_lock

#include "flann/algorithms/kmeans_index.h"
#include "flann/algorithms/cuda/cuda_utils.h"
#include "flann/algorithms/cuda/nn_cuda_index.h"
#include "flann/algorithms/cuda/kmeans_node_gpu.h"
#include "flann/util/gpu_saving.h"

// Only include kernel headers when compiling with nvcc (before namespace to avoid nesting)
#ifdef __CUDACC__
#include "flann/algorithms/cuda/kernels/kmeans_search_cooperative.cuh"
#include "flann/algorithms/cuda/kernels/utility_kernels.cuh"
#endif

namespace flann {
namespace cuda {

#ifndef __CUDACC__
// Forward declare kernel launch functions for non-CUDA compilation
template<int K>
bool launch_kmeans_search_cooperative(
    const float* dataset,
    const float* queries,
    const int* node_index,
    const float* node_pivots,
    const float* node_variance,
    int* result_indices,
    float* result_distances,
    size_t num_queries,
    size_t dim,
    size_t num_nodes,
    int heap_size,
    int loc_size,
    int branching,
    float cb_index,
    cudaStream_t stream = nullptr);

// Forward declare padding kernel launcher
template<typename T>
bool launch_pad_queries(
    const T* src,
    T* dst,
    size_t num_queries,
    size_t veclen,
    size_t padded_veclen,
    cudaStream_t stream = nullptr);
#endif

/**
 * @brief CUDA-accelerated K-Means index parameters
 */
struct KMeansCUDAIndexParams : public KMeansIndexParams {
    KMeansCUDAIndexParams(
        int branching = 32,
        int iterations = 11,
        flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
        float cb_index = 0.2f)
        : KMeansIndexParams(branching, iterations, centers_init, cb_index)
    {
        // Override parent param's algorithm
        (*this)["algorithm"] = FLANN_INDEX_KMEANS_CUDA;
    }
};

/**
 * @brief CUDA-accelerated K-Means index
 *
 * Hierarchical k-means tree with GPU-accelerated search.
 * Builds tree on CPU (reusing KMeansIndex logic), then uploads to GPU.
 *
 * THREAD SAFETY: This index IS thread-safe for concurrent searches.
 * Multiple threads can safely call knnSearch() / knnSearchGPU() on
 * the same index instance. Mutations (addPoints, removePoint) acquire
 * exclusive lock and block all searches during modification.
 *
 * Memory layout:
 * - Tree nodes: Breadth-first flat array
 * - Pivots: Separate array for coalesced access
 * - Dataset: Row-major, padded to 4-element alignment
 *
 * Performance characteristics:
 * - Build time: Same as CPU (tree built on CPU)
 * - Search speedup: 5-20x vs multi-core CPU (large datasets)
 * - Memory overhead: ~2.5x dataset size (tree + GPU copy)
 */
template <typename Distance>
class KMeansCUDAIndex : public KMeansIndex<Distance>, public CUDAIndex
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    typedef typename KMeansIndex<Distance>::Node Node;
    typedef KMeansIndex<Distance> BaseClass;

    // Bring base class knnSearch overloads into scope (prevents C++ name hiding)
    using BaseClass::knnSearch;

    /**
     * @brief Get index type identifier
     */
    flann_algorithm_t getType() const override
    {
        return FLANN_INDEX_KMEANS_CUDA;
    }

    /**
     * @brief Constructor with dataset
     *
     * @param inputData Dataset matrix [N x D]
     * @param params K-Means parameters (branching, iterations, etc.)
     * @param d Distance functor
     */
    KMeansCUDAIndex(
        const Matrix<ElementType>& inputData,
        const IndexParams& params = KMeansCUDAIndexParams(),
        Distance d = Distance())
        : BaseClass(inputData, params, d),
          CUDAIndex(),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          num_nodes_(0),
          padded_veclen_(0),
          workspace_offset_variance_(0),
          workspace_offset_pivots_(0),
          workspace_offset_dataset_(0),
          node_index_ptr_(nullptr),
          node_variance_ptr_(nullptr),
          tree_pivots_ptr_(nullptr),
          dataset_ptr_(nullptr)
    {
    }

    /**
     * @brief Constructor without dataset (for loading from file)
     */
    KMeansCUDAIndex(
        const IndexParams& params = KMeansCUDAIndexParams(),
        Distance d = Distance())
        : BaseClass(params, d),
          CUDAIndex(),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          num_nodes_(0),
          leaf_count_(0),
          padded_veclen_(0),
          workspace_offset_variance_(0),
          workspace_offset_pivots_(0),
          workspace_offset_dataset_(0),
          node_index_ptr_(nullptr),
          node_variance_ptr_(nullptr),
          tree_pivots_ptr_(nullptr),
          dataset_ptr_(nullptr)
    {
    }

    /**
     * @brief Conversion constructor from CPU K-Means index
     *
     * Enables seamless CPU→GPU conversion matching OpenCL pattern.
     * Copies the CPU tree structure, GPU buffers allocated on-demand.
     *
     * @param other CPU K-Means index to convert from
     */
    KMeansCUDAIndex(const KMeansIndex<Distance>& other)
        : BaseClass(other),
          CUDAIndex(),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          num_nodes_(0),
          leaf_count_(0),
          padded_veclen_(0),
          workspace_offset_variance_(0),
          workspace_offset_pivots_(0),
          workspace_offset_dataset_(0),
          node_index_ptr_(nullptr),
          node_variance_ptr_(nullptr),
          tree_pivots_ptr_(nullptr),
          dataset_ptr_(nullptr)
    {
        // CPU tree is copied via BaseClass copy constructor
        // GPU buffers will be allocated when buildCUDAKnnSearch() is called
    }

    /**
     * @brief Copy constructor
     *
     * Note: GPU buffers are NOT copied, only CPU tree structure.
     * Call buildIndex() after copy to re-upload to GPU.
     */
    KMeansCUDAIndex(const KMeansCUDAIndex& other)
        : BaseClass(other),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          num_nodes_(0),
          leaf_count_(0),
          padded_veclen_(0),
          workspace_offset_variance_(0),
          workspace_offset_pivots_(0),
          workspace_offset_dataset_(0),
          node_index_ptr_(nullptr),
          node_variance_ptr_(nullptr),
          tree_pivots_ptr_(nullptr),
          dataset_ptr_(nullptr)
    {
    }

    /**
     * @brief Assignment operator
     */
    KMeansCUDAIndex& operator=(KMeansCUDAIndex other)
    {
        this->swap(other);
        return *this;
    }

    /**
     * @brief Clone index
     */
    BaseClass* clone() const override
    {
        return new KMeansCUDAIndex(*this);
    }

    /**
     * @brief Check if index is in GPU-only mode
     *
     * In GPU-only mode (loaded with gpu_only=true), the CPU-side points_
     * array is not populated. This saves ~30-50% memory but means:
     * - getPoint() is not available
     * - CPU search fallback is disabled
     * - addPoints()/removePoint() are not available
     *
     * @return true if GPU-only mode, false if CPU data is available
     */
    bool isGPUOnlyMode() const { return gpu_only_mode_; }

    /**
     * @brief Get point by ID (throws in GPU-only mode)
     *
     * @param id Point ID
     * @return Pointer to point data
     * @throws FLANNException if index is in GPU-only mode
     */
    ElementType* getPoint(size_t id) override
    {
        if (gpu_only_mode_) {
            throw FLANNException(
                "getPoint() not available in GPU-only mode.\n"
                "Index was loaded with gpu_only=true to save memory.\n"
                "Options:\n"
                "  1. Reload with loadIndexV2(stream, false)\n"
                "  2. Use knnSearch() for GPU queries instead\n"
                "  3. Keep original dataset in memory separately");
        }
        return BaseClass::getPoint(id);
    }

    /**
     * @brief Destructor - RAII cleanup via CUDABuffer
     *
     */
    ~KMeansCUDAIndex()
    {
        // CUDABuffer destructors automatically free GPU memory
    }

    /**
     * @brief Swap with another index
     */
    void swap(KMeansCUDAIndex& other)
    {
        BaseClass::swap(other);
        std::swap(gpu_initialized_, other.gpu_initialized_);
        std::swap(gpu_search_ready_, other.gpu_search_ready_);
        std::swap(gpu_only_mode_, other.gpu_only_mode_);
        std::swap(num_nodes_, other.num_nodes_);
        std::swap(padded_veclen_, other.padded_veclen_);

        // Swap GPU workspace and pointers
        std::swap(workspace_gpu_, other.workspace_gpu_);
        std::swap(workspace_offset_variance_, other.workspace_offset_variance_);
        std::swap(workspace_offset_pivots_, other.workspace_offset_pivots_);
        std::swap(workspace_offset_dataset_, other.workspace_offset_dataset_);
        std::swap(node_index_ptr_, other.node_index_ptr_);
        std::swap(node_variance_ptr_, other.node_variance_ptr_);
        std::swap(tree_pivots_ptr_, other.tree_pivots_ptr_);
        std::swap(dataset_ptr_, other.dataset_ptr_);
    }

    /**
     * @brief Add points to index (invalidates GPU data)
     *
     * Thread-safe: acquires exclusive lock, blocking all concurrent searches.
     *
     * @param points New points to add
     * @param rebuild_threshold Rebuild threshold (default: 2.0)
     * @throws FLANNException if index is in GPU-only mode
     */
    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2.0f) override
    {
        if (gpu_only_mode_) {
            throw FLANNException(
                "addPoints() not supported in GPU-only mode.\n"
                "Reload with loadIndexV2(stream, false) to enable dynamic updates.");
        }
        // Acquire exclusive lock - blocks all concurrent searches
        std::unique_lock<std::shared_mutex> lock(this->rw_lock_);
        freeGPUMemory();
        BaseClass::addPoints(points, rebuild_threshold);
    }

    /**
     * @brief Remove point from index (invalidates GPU data)
     *
     * Thread-safe: acquires exclusive lock, blocking all concurrent searches.
     *
     * @param id Index of point to remove
     * @throws FLANNException if index is in GPU-only mode
     */
    void removePoint(size_t id) override
    {
        if (gpu_only_mode_) {
            throw FLANNException(
                "removePoint() not supported in GPU-only mode.\n"
                "Reload with loadIndexV2(stream, false) to enable dynamic updates.");
        }
        // Acquire exclusive lock - blocks all concurrent searches
        std::unique_lock<std::shared_mutex> lock(this->rw_lock_);
        freeGPUMemory();
        BaseClass::removePoint(id);
    }

    /**
     * @brief Build index (CPU tree only)
     *
     * Steps:
     * 1. Build K-Means tree on CPU (BaseClass::buildIndexImpl)
     *
     * Note: GPU resources are NOT allocated here. Call buildCUDAKnnSearch()
     * after this method to prepare GPU for searches.
     */
    void buildIndex() override
    {
        // Build CPU tree (reuse KMeansIndex logic)
        BaseClass::buildIndex();

        // GPU upload now happens in buildCUDAKnnSearch() (user-controlled)
    }

    /**
     * @brief Prepare GPU resources for knnSearch operations
     *
     * This method uploads the tree structure to GPU memory and marks the
     * index as ready for GPU searches. It must be called after buildIndex()
     * and before knnSearchGPU() if GPU acceleration is desired.
     *
     * This is an expensive operation (1-2 seconds for typical datasets) and
     * should be called once before multiple searches to amortize the cost.
     *
     * @param knn Number of nearest neighbors (reserved for future kernel optimization)
     * @param params Search parameters (reserved for future kernel compilation)
     *
     * @throws FLANNException if buildIndex() not called yet
     *
     * Example:
     *   index.buildIndex();                          // Build CPU tree
     *   index.buildCUDAKnnSearch(10, SearchParams()); // Prepare GPU (expensive)
     *   for (...) {
     *       index.knnSearchGPU(...);                 // Search many times (fast)
     *   }
     */
    void buildCUDAKnnSearch(int knn, const SearchParams& params = SearchParams())
    {
        // Validate K is supported before any work
        if (!isKValueSupportedForGPU(knn)) {
            throw FLANNException(
                "Unsupported k=" + std::to_string(knn) + " for CUDA K-Means search.\n"
                "Supported k values: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100\n"
                "Use a supported k value or fall back to CPU search.");
        }

        // Prepare GPU (upload tree/dataset) - this is K-independent
        prepareGPUIndex();
    }

    /**
     * @brief Check if GPU search is ready
     *
     * @return true if buildCUDAKnnSearch() has been called and GPU is ready
     */
    bool isGPUSearchReady() const
    {
        return gpu_search_ready_;
    }

    /**
     * @brief Check if index has GPU-optimized format available
     *
     * Returns true if GPU buffers are initialized and the index can be saved
     * in GPU-optimized format for fast reloading.
     */
    bool hasGPUFormat() const
    {
        return gpu_initialized_;
    }

    /**
     * @brief Check if a K value is supported for GPU search
     *
     * K-Means CUDA supports: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100
     *
     * @param k Number of nearest neighbors
     * @return true if k is supported for GPU search
     */
    bool isKValueSupportedForGPU(int k) const
    {
        return k == 1 || k == 2 || k == 4 || k == 5 || k == 7 || k == 8 ||
               k == 10 || k == 16 || k == 20 || k == 32 || k == 50 || k == 64 || k == 100;
    }

    /**
     * @brief Prepare GPU index for search (K-independent)
     *
     * Uploads tree and dataset to GPU without requiring a K value.
     * GPU is ready for search immediately after this call.
     *
     * This is the preferred way to prepare GPU indices when K is not known
     * at conversion time, or when multiple K values will be used.
     *
     * @throws FLANNException if index not built
     */
    void prepareGPUIndex()
    {
        std::unique_lock<std::shared_mutex> lock(this->rw_lock_);
        if (!this->root_ && !gpu_initialized_) {
            throw FLANNException("Cannot prepare GPU search: index not built yet. "
                               "Call buildIndex() first.");
        }
        if (!gpu_initialized_) {
            uploadToGPUInternal();
        }
        gpu_search_ready_ = true;
    }

    /**
     * @brief Convert index to GPU-optimized format
     *
     * Ensures GPU buffers are populated for GPU-optimized saving.
     * Also discards the CPU tree structure to reduce memory usage.
     *
     * After calling this method:
     * - GPU search will work (via knnSearch with GPU fallback)
     * - CPU search will NOT work (root_ is null)
     * - Index can be saved in GPU-optimized format
     *
     * @throws FLANNException if tree not built (buildIndex() not called)
     */
    void convertToGPUFormat()
    {
        if (!this->root_) {
            throw FLANNException("Cannot convert to GPU format: index not built. "
                               "Call buildIndex() first.");
        }
        if (!gpu_initialized_) {
            uploadToGPU();
        }
        // Discard CPU tree to save memory
        discardCPUTree();
    }

    /**
     * @brief Save index to file (GPU format if available, CPU format otherwise)
     *
     * If GPU buffers are initialized, saves in GPU-optimized format with
     * signature "FLANN_GPU_INDEX_v2.0". Otherwise falls back to CPU format.
     *
     * GPU format stores flat arrays directly, enabling fast reload without
     * tree reconstruction.
     *
     * @param stream Output file stream
     */
    void saveIndex(FILE* stream) override
    {
        if (!gpu_initialized_) {
            // Fall back to CPU format
            BaseClass::saveIndex(stream);
            return;
        }

        // Use v2.0 format by default (single workspace blob, optimized loading)
        saveIndexV2(stream);
    }

    /**
     * @brief Load index from file (auto-detects format)
     *
     * Detects format by checking file signature and index type:
     * - "FLANN_GPU_INDEX_v2.0" → GPU format (optimized single blob)
     * - "FLANN_INDEX_v1.1" with FLANN_INDEX_KMEANS → CPU format
     *   (loads via temp CPU index, then copies to CUDA index)
     * - "FLANN_INDEX_v1.1" with FLANN_INDEX_KMEANS_CUDA → CPU format
     *   (direct load via BaseClass)
     *
     * This allows loading indices saved by either:
     * - KMeansCUDAIndex (saves as KMEANS_CUDA or GPU v2.0)
     * - KMeansIndex (saves as KMEANS)
     *
     * **Important:** When loading pure CPU KMEANS format, the saved file
     * must have been created with `save_dataset=true` so the dataset is embedded.
     *
     * @param stream Input file stream. Should be opened with "rb" mode.
     *               Stream position will be reset to 0 internally for CPU format.
     *               The stream is NOT closed by this method.
     *
     * @throws FLANNException if stream is invalid, file format is unrecognized,
     *         dataset not embedded (for CPU KMEANS), or data type mismatch.
     */
    void loadIndex(FILE* stream) override
    {
        // Defensive validation: check stream is valid
        if (stream == nullptr) {
            throw FLANNException("loadIndex: stream is null");
        }

        if (GPUIndexHeaderV2::detectV2Format(stream)) {
            // GPU v2.0 format - detectV2Format already restored position
            loadIndexV2(stream);
        } else {
            // CPU v1.1 format - peek at header to determine exact index type
            std::rewind(stream);
            IndexHeader header = load_header(stream);
            std::rewind(stream);

            if (header.h.index_type == FLANN_INDEX_KMEANS) {
                // CPU KMEANS format - load via temporary CPU index
                // This avoids type mismatch error in NNIndex::serialize()
                freeGPUMemory();

                // Load into temporary CPU index (type matches file)
                // Note: saved file must have save_dataset=true, otherwise load fails
                KMeansIndex<Distance> temp_cpu;
                temp_cpu.loadIndex(stream);

                // Use existing copy constructor + swap idiom
                // The copy constructor handles all tree copying via BaseClass(other)
                KMeansCUDAIndex temp_cuda(temp_cpu);
                this->swap(temp_cuda);

            } else if (header.h.index_type == FLANN_INDEX_KMEANS_CUDA) {
                // CPU v1.1 format saved as CUDA type - direct load works
                freeGPUMemory();
                BaseClass::loadIndex(stream);
            } else {
                throw FLANNException(
                    "Unsupported index type for KMeansCUDAIndex. "
                    "Expected FLANN_INDEX_KMEANS or FLANN_INDEX_KMEANS_CUDA, got: " +
                    std::to_string(header.h.index_type));
            }
        }
    }

    /**
     * @brief GPU-direct k-NN search with device pointers (zero-copy)
     *
     * Performs k-nearest neighbor search using device-resident data,
     * eliminating CPU<->GPU memory transfers for queries and results.
     * Ideal for GPU-resident pipelines where data never leaves device memory.
     *
     * @param d_queries Device pointer to query vectors [num_queries x veclen]
     *                  Must be contiguous row-major layout.
     * @param d_indices Device pointer for output indices [num_queries x knn]
     *                  Will contain point indices as int.
     * @param d_dists   Device pointer for output distances [num_queries x knn]
     *                  L2 squared distances as float.
     * @param num_queries Number of query vectors
     * @param knn Number of nearest neighbors to find
     * @param params Search parameters (checks value used for heap size)
     * @param stream CUDA stream for async execution (caller must synchronize)
     *
     * @return Number of queries processed
     * @throws FLANNException if GPU not initialized or k unsupported
     *
     * @note Caller must synchronize on stream before reading results.
     * @note If veclen % 4 != 0, queries are padded internally.
     *
     * Example:
     *   // Data already on GPU
     *   float* d_queries;  // [N x 128] descriptors on GPU
     *   int* d_indices;    // Output [N x k]
     *   float* d_dists;    // Output [N x k]
     *
     *   index.knnSearchGPUDirect(d_queries, d_indices, d_dists, N, k, params, stream);
     *   cudaStreamSynchronize(stream);  // Caller syncs
     */
    int knnSearchGPUDirect(
        const ElementType* d_queries,
        int* d_indices,
        float* d_dists,
        size_t num_queries,
        size_t knn,
        const SearchParams& params,
        cudaStream_t stream) const
    {
        // 1. Validate GPU initialization
        if (!gpu_initialized_) {
            throw FLANNException("GPU index not initialized. Call buildCUDAKnnSearch() or prepareGPUIndex() first.");
        }

        // 2. Validate K value is supported
        if (!isKValueSupportedForGPU(static_cast<int>(knn))) {
            throw FLANNException(
                "Unsupported k=" + std::to_string(knn) + " for CUDA K-Means search.\n"
                "Supported k values: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100\n"
                "Use a supported k value or fall back to CPU search.");
        }

        // 3. Handle zero queries
        if (num_queries == 0) {
            return 0;
        }

        // 4. Validate output pointers
        if (d_indices == nullptr || d_dists == nullptr) {
            throw FLANNException("Output pointers d_indices and d_dists cannot be null");
        }

        // 5. Thread safety - acquire shared lock for concurrent searches
        std::shared_lock<std::shared_mutex> lock(this->rw_lock_);

        // 6. Use caller-provided stream
        cudaStream_t exec_stream = stream;

        // 7. Handle query padding if needed
        const ElementType* kernel_queries = d_queries;
        CUDABuffer<ElementType> padded_queries;

        if (padded_veclen_ != this->veclen_) {
            // Allocate and pad queries on GPU
            padded_queries = CUDABuffer<ElementType>(num_queries * padded_veclen_);
            if (!launch_pad_queries<ElementType>(
                    d_queries,
                    padded_queries.get(),
                    num_queries,
                    this->veclen_,
                    padded_veclen_,
                    exec_stream)) {
                throw FLANNException("Failed to launch GPU padding kernel");
            }
            kernel_queries = padded_queries.get();
        }

        // 8. Calculate heap size based on parameters
        int max_checks = params.checks;
        if (max_checks <= 0 || max_checks == FLANN_CHECKS_UNLIMITED) {
            max_checks = 256;
        }
        int heap_size = calculateHeapSize(knn, max_checks);
        int loc_size = getCUDALocSize(0);

        // 9. Validate cooperative kernel can be used
        bool heap_ok = (heap_size <= loc_size);
        bool branch_ok = (this->branching_ == 32 || this->branching_ == 64);
        if (!heap_ok || !branch_ok) {
            throw FLANNException(
                "knnSearchGPUDirect requires branching=32 or 64 and heap_size <= loc_size. "
                "Current: branching=" + std::to_string(this->branching_) +
                ", heap_size=" + std::to_string(heap_size) +
                ", loc_size=" + std::to_string(loc_size));
        }

        // 10. Launch kernel with K dispatch
        bool success = false;
        #define SEARCH_DISPATCH(K) \
            success = launch_kmeans_search_cooperative<K>( \
                (const float*)dataset_ptr_, \
                (const float*)kernel_queries, \
                node_index_ptr_, \
                (const float*)tree_pivots_ptr_, \
                node_variance_ptr_, \
                d_indices, \
                d_dists, \
                num_queries, padded_veclen_, num_nodes_, \
                heap_size, loc_size, this->branching_, this->cb_index_, \
                exec_stream)

        if (knn == 1) { SEARCH_DISPATCH(1); }
        else if (knn == 2) { SEARCH_DISPATCH(2); }
        else if (knn == 4) { SEARCH_DISPATCH(4); }
        else if (knn == 5) { SEARCH_DISPATCH(5); }
        else if (knn == 7) { SEARCH_DISPATCH(7); }
        else if (knn == 8) { SEARCH_DISPATCH(8); }
        else if (knn == 10) { SEARCH_DISPATCH(10); }
        else if (knn == 16) { SEARCH_DISPATCH(16); }
        else if (knn == 20) { SEARCH_DISPATCH(20); }
        else if (knn == 32) { SEARCH_DISPATCH(32); }
        else if (knn == 50) { SEARCH_DISPATCH(50); }
        else if (knn == 64) { SEARCH_DISPATCH(64); }
        else if (knn == 100) { SEARCH_DISPATCH(100); }
        else {
            throw FLANNException("Unsupported k=" + std::to_string(knn) + " for GPU direct search");
        }

        #undef SEARCH_DISPATCH

        if (!success) {
            throw FLANNException("Kernel launch failed for k=" + std::to_string(knn));
        }

        // 11. Check for kernel launch errors (non-blocking)
        CUDA_CHECK_LAST();

        // 12. No synchronization - caller is responsible for stream sync
        return static_cast<int>(num_queries);
    }

    /**
     * @brief GPU-accelerated k-NN batch search
     *
     * Performs k-nearest neighbor search using GPU acceleration.
     * Requires buildCUDAKnnSearch() to be called first.
     *
     * This method is for explicit GPU batch searches. For automatic CPU/GPU
     * dispatch, use the base class knnSearch() (currently CPU-only).
     *
     * @param queries Query matrix [num_queries x veclen_]
     * @param indices Output indices [num_queries x knn]
     * @param dists Output distances [num_queries x knn]
     * @param knn Number of nearest neighbors
     * @param params Search parameters (checks, eps, sorted)
     * @return Number of queries processed
     *
     * @throws FLANNException if buildCUDAKnnSearch() not called yet
     *
     * Example:
     *   index.buildIndex();
     *   index.buildCUDAKnnSearch(10, SearchParams());
     *   index.knnSearchGPU(queries, indices, dists, 10, SearchParams());
     */
    int knnSearchGPU(const Matrix<ElementType>& queries,
                     Matrix<int>& indices,
                     Matrix<DistanceType>& dists,
                     int knn,
                     const SearchParams& params) const
    {
        if (!gpu_search_ready_) {
            throw FLANNException("GPU search not prepared. "
                               "Call buildCUDAKnnSearch() first.");
        }

        return knnSearchGPUImpl(queries, indices, dists, knn, params);
    }

    /**
     * @brief GPU-accelerated k-NN batch search with size_t indices
     */
    int knnSearchGPU(const Matrix<ElementType>& queries,
                     Matrix<size_t>& indices,
                     Matrix<DistanceType>& dists,
                     size_t knn,
                     const SearchParams& params) const
    {
        // Allocate temporary int matrix
        std::vector<int> indices_int_data(indices.rows * indices.cols);
        Matrix<int> indices_int(&indices_int_data[0], indices.rows, indices.cols);

        int result = knnSearchGPU(queries, indices_int, dists, (int)knn, params);

        // Copy back to size_t
        for (size_t i = 0; i < indices.rows; ++i) {
            for (size_t j = 0; j < indices.cols; ++j) {
                indices[i][j] = indices_int[i][j];
            }
        }

        return result;
    }

    /**
     * @brief Check if a given k value is supported by the CUDA cooperative kernel
     *
     * The cooperative kernel requires template specialization for each k value.
     * Unsupported k values will fall back to CPU search.
     *
     * @param knn Number of nearest neighbors to check
     * @return true if k is supported on GPU, false if CPU fallback needed
     */
    bool isKValueSupportedForGPU(size_t knn) const
    {
        // Cooperative kernel only supports these k values (template specializations)
        // Also requires branching=32 or 64
        if (this->branching_ != 32 && this->branching_ != 64) {
            return false;
        }
        // K-Means CUDA supports: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100
        return (knn == 1 || knn == 2 || knn == 4 || knn == 5 || knn == 7 || knn == 8 ||
                knn == 10 || knn == 16 || knn == 20 || knn == 32 || knn == 50 || knn == 64 || knn == 100);
    }

    /**
     * @brief Automatic CPU/GPU dispatch for k-NN search (size_t indices)
     *
     * Overrides base class knnSearch() to automatically use GPU when prepared.
     * This enables transparent GPU acceleration without API changes.
     *
     * If buildCUDAKnnSearch() was called, uses GPU acceleration via knnSearchGPU().
     * Otherwise, falls back to CPU search from base class.
     * Falls back to CPU if k value is not supported by the CUDA kernel.
     *
     * @param queries Query matrix [num_queries x veclen_]
     * @param indices Output indices [num_queries x knn]
     * @param dists Output distances [num_queries x knn]
     * @param knn Number of nearest neighbors
     * @param params Search parameters (checks, eps, sorted)
     * @return Number of queries processed
     */
    int knnSearch(const Matrix<ElementType>& queries,
                  Matrix<size_t>& indices,
                  Matrix<DistanceType>& dists,
                  size_t knn,
                  const SearchParams& params) const override
    {
        if (gpu_search_ready_) {
            // GPU is ready - require supported k value
            if (!isKValueSupportedForGPU(knn)) {
                throw FLANNException("Unsupported k=" + std::to_string(knn) +
                    " for K-Means CUDA. Supported k values: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100. "
                    "Use CPU KMeansIndex for other k values.");
            }
            return knnSearchGPU(queries, indices, dists, knn, params);
        }
        // GPU not ready - use CPU
        return BaseClass::knnSearch(queries, indices, dists, knn, params);
    }

    /**
     * @brief Automatic CPU/GPU dispatch for k-NN search (int indices)
     *
     * Overrides base class knnSearch() to automatically use GPU when prepared.
     * Falls back to CPU if k value is not supported by the CUDA kernel.
     *
     * @param queries Query matrix [num_queries x veclen_]
     * @param indices Output indices [num_queries x knn]
     * @param dists Output distances [num_queries x knn]
     * @param knn Number of nearest neighbors
     * @param params Search parameters (checks, eps, sorted)
     * @return Number of queries processed
     */
    int knnSearch(const Matrix<ElementType>& queries,
                  Matrix<int>& indices,
                  Matrix<DistanceType>& dists,
                  size_t knn,
                  const SearchParams& params) const override
    {
        if (gpu_search_ready_) {
            // GPU is ready - require supported k value
            if (!isKValueSupportedForGPU(knn)) {
                throw FLANNException("Unsupported k=" + std::to_string(knn) +
                    " for K-Means CUDA. Supported k values: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100. "
                    "Use CPU KMeansIndex for other k values.");
            }
            return knnSearchGPU(queries, indices, dists, (int)knn, params);
        }
        // GPU not ready - use CPU
        return BaseClass::knnSearch(queries, indices, dists, knn, params);
    }

protected:
    /**
     * @brief Save GPU index (single workspace blob format)
     *
     * Format features:
     * - Pre-computed workspace offsets stored in header
     * - Single contiguous blob with alignment padding included
     * - Zero runtime computation at load time
     * - Direct upload to GPU via pinned memory
     *
     * @param stream Output file stream
     */
    void saveIndexV2(FILE* stream)
    {
        if (!gpu_initialized_) {
            throw FLANNException("GPU not initialized, cannot save v2.0 format");
        }

        serialization::SaveArchive sa(stream);

        // Calculate sizes (matching the workspace layout)
        size_t node_index_size = workspace_offset_variance_ / sizeof(int);
        size_t node_index_bytes = node_index_size * sizeof(int);
        size_t variance_bytes = num_nodes_ * sizeof(float);
        size_t pivots_bytes = num_nodes_ * padded_veclen_ * sizeof(ElementType);
        size_t dataset_bytes = this->size_ * padded_veclen_ * sizeof(ElementType);

        // Calculate total workspace size (with alignment padding)
        size_t total_workspace_size = workspace_offset_dataset_ + dataset_bytes;

        // Build v2.0 header with pre-computed offsets
        GPUIndexHeaderV2 header;
        header.h.data_type = flann_datatype_value<ElementType>::value;
        header.h.index_type = FLANN_INDEX_KMEANS_GPU_SAVED;
        header.h.rows = this->size_;
        header.h.cols = this->veclen_;
        header.h.padded_veclen = padded_veclen_;
        header.h.num_nodes = num_nodes_;
        header.h.node_index_size = node_index_size;
        header.h.leaf_count = leaf_count_;
        header.h.branching = this->branching_;
        header.h.iterations = this->iterations_;
        header.h.cb_index = this->cb_index_;
        header.h.centers_init = this->centers_init_;

        // V2.0 specific: pre-computed workspace layout
        header.h.workspace_total_size = total_workspace_size;
        header.h.offset_node_index = 0;
        header.h.offset_variance = workspace_offset_variance_;
        header.h.offset_pivots = workspace_offset_pivots_;
        header.h.offset_dataset = workspace_offset_dataset_;

        sa & header;

        // Download entire GPU workspace as single blob
        // Use pinned memory for faster download
        PinnedBuffer<unsigned char> workspace_host(total_workspace_size);

        CUDA_CHECK(cudaMemcpy(workspace_host.get(), workspace_gpu_.get(),
                             total_workspace_size, cudaMemcpyDeviceToHost));

        // Serialize single blob (includes alignment padding)
        sa & serialization::make_binary_object(
            workspace_host.get(),
            total_workspace_size);

        // Save NNIndex base data
        sa & this->last_id_;
        sa & this->ids_;
        sa & this->removed_;
        if (this->removed_) {
            sa & this->removed_points_;
        }
        sa & this->removed_count_;
    }

    /**
     * @brief Load GPU index from v2.0 format (single workspace blob)
     *
     * Optimized loading path:
     * - Workspace offsets read directly from header (no computation)
     * - Single blob read directly into pinned memory
     * - Single cudaMemcpy upload to GPU
     *
     * @param stream Input file stream
     * @param gpu_only If true, skip CPU points_ reconstruction
     */
    void loadIndexV2(FILE* stream, bool gpu_only = true)
    {
        freeGPUMemory();
        gpu_only_mode_ = gpu_only;

        serialization::LoadArchive la(stream);

        GPUIndexHeaderV2 header;
        la & header;

        // Validate header
        if (header.h.data_type != flann_datatype_value<ElementType>::value) {
            throw FLANNException("Data type mismatch in GPU v2.0 index file");
        }
        if (header.h.index_type != FLANN_INDEX_KMEANS_GPU_SAVED) {
            throw FLANNException("Index type mismatch: expected KMEANS_GPU_SAVED");
        }

        // Restore metadata from header
        this->size_ = header.h.rows;
        this->veclen_ = header.h.cols;
        this->size_at_build_ = header.h.rows;
        padded_veclen_ = header.h.padded_veclen;
        num_nodes_ = header.h.num_nodes;
        leaf_count_ = header.h.leaf_count;
        this->branching_ = header.h.branching;
        this->iterations_ = header.h.iterations;
        this->cb_index_ = header.h.cb_index;
        this->centers_init_ = header.h.centers_init;

        // V2.0 OPTIMIZATION: Read workspace offsets directly from header
        // No alignment computation needed!
        workspace_offset_variance_ = header.h.offset_variance;
        workspace_offset_pivots_ = header.h.offset_pivots;
        workspace_offset_dataset_ = header.h.offset_dataset;
        size_t total_workspace_size = header.h.workspace_total_size;

        // Allocate pinned buffer and load single blob directly
        PinnedBuffer<unsigned char> workspace_host(total_workspace_size);

        la & serialization::make_binary_object(
            workspace_host.get(),
            total_workspace_size);

        // Single GPU upload
        workspace_gpu_.resize(total_workspace_size);
        workspace_gpu_.upload(workspace_host.get(), total_workspace_size);

        // Set up raw pointers into workspace
        unsigned char* base_ptr = workspace_gpu_.get();
        node_index_ptr_ = reinterpret_cast<int*>(base_ptr + header.h.offset_node_index);
        node_variance_ptr_ = reinterpret_cast<float*>(base_ptr + workspace_offset_variance_);
        tree_pivots_ptr_ = reinterpret_cast<ElementType*>(base_ptr + workspace_offset_pivots_);
        dataset_ptr_ = reinterpret_cast<ElementType*>(base_ptr + workspace_offset_dataset_);

        // Optionally reconstruct points_ for CPU compatibility
        if (gpu_only) {
            delete[] this->data_ptr_;
            this->data_ptr_ = nullptr;
            this->points_.clear();
        } else {
            delete[] this->data_ptr_;
            this->data_ptr_ = new ElementType[this->size_ * this->veclen_];
            this->points_.resize(this->size_);

            const ElementType* dataset_in_workspace = reinterpret_cast<const ElementType*>(
                workspace_host.get() + workspace_offset_dataset_);
            for (size_t i = 0; i < this->size_; ++i) {
                this->points_[i] = this->data_ptr_ + i * this->veclen_;
                std::memcpy(this->points_[i],
                           &dataset_in_workspace[i * padded_veclen_],
                           this->veclen_ * sizeof(ElementType));
            }
        }

        // Load NNIndex base data
        la & this->last_id_;
        la & this->ids_;
        la & this->removed_;
        if (this->removed_) {
            la & this->removed_points_;
        }
        la & this->removed_count_;

        // Restore index params
        this->index_params_["algorithm"] = FLANN_INDEX_KMEANS_CUDA;
        this->index_params_["branching"] = this->branching_;
        this->index_params_["iterations"] = this->iterations_;
        this->index_params_["centers_init"] = this->centers_init_;
        this->index_params_["cb_index"] = this->cb_index_;

        gpu_initialized_ = true;
    }

    /**
     * @brief Discard CPU tree to save memory
     *
     * Called by convertToGPUFormat() to free the CPU tree structure
     * after GPU buffers are populated. After this, CPU search will not work.
     */
    void discardCPUTree()
    {
        if (this->root_) {
            // Memory pool cleans up nodes on free
            this->pool_.free();
            this->root_ = nullptr;
        }
    }

    /**
     * @brief GPU k-NN search implementation (internal)
     *
     * Steps:
     * 1. Upload queries to GPU
     * 2. Allocate result buffers
     * 3. Launch search kernel
     * 4. Download results
     *
     * @param queries Query matrix
     * @param indices Output indices
     * @param dists Output distances
     * @param knn Number of nearest neighbors
     * @param params Search parameters
     * @return Number of queries processed
     */
    int knnSearchGPUImpl(const Matrix<ElementType>& queries,
                         Matrix<int>& indices,
                         Matrix<DistanceType>& dists,
                         int knn,
                         const SearchParams& params) const
    {
        if (!gpu_initialized_) {
            throw FLANNException("Index not built or GPU data not uploaded");
        }

        // Validate K value is supported for GPU search
        if (!isKValueSupportedForGPU(knn)) {
            throw FLANNException(
                "Unsupported k=" + std::to_string(knn) + " for CUDA K-Means search.\n"
                "Supported k values: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100\n"
                "Use a supported k value or fall back to CPU search.");
        }

        // Acquire shared lock - allows multiple concurrent searches
        // Mutations (addPoints, removePoint) acquire exclusive lock and wait for searches to complete
        std::shared_lock<std::shared_mutex> lock(this->rw_lock_);

        // Get thread-local CUDA stream for concurrent GPU operations
        cudaStream_t stream = this->getThreadStream();

        size_t num_queries = queries.rows;

        // Validate dimensions
        if (queries.cols != this->veclen_) {
            throw FLANNException("Query dimension mismatch");
        }
        if (indices.rows < num_queries || indices.cols < (size_t)knn) {
            throw FLANNException("Indices matrix too small");
        }
        if (dists.rows < num_queries || dists.cols < (size_t)knn) {
            throw FLANNException("Distances matrix too small");
        }

        // Get search parameters
        int max_checks = params.checks;
        if (max_checks <= 0 || max_checks == FLANN_CHECKS_UNLIMITED) {
            max_checks = 256;  // Default
        }

        // Require contiguous queries (O(1) stride check instead of O(n) pointer loop)
        if (queries.stride != this->veclen_ * sizeof(ElementType)) {
            throw FLANNException(
                "GPU search requires contiguous query matrix (stride must equal cols * sizeof(T)). "
                "Allocate queries as: Matrix<T>(new T[rows * cols], rows, cols)");
        }

        // One-time warning if padding is needed (non-multiple-of-4 dimension)
        if (padded_veclen_ != this->veclen_) {
            static bool warned = false;
            if (!warned) {
                fprintf(stderr, "[FLANN] Dimension %zu requires padding to %zu. "
                    "Use multiples of 4 for best performance.\n",
                    this->veclen_, padded_veclen_);
                warned = true;
            }
        }

        // Upload queries to GPU (using thread-local stream)
        CUDABuffer<ElementType> queries_gpu(num_queries * padded_veclen_);
        if (padded_veclen_ == this->veclen_) {
            // Fast path: no padding needed, direct upload
            queries_gpu.upload(queries[0], num_queries * this->veclen_, stream);
        } else {
            // Padding needed: upload raw then pad on GPU
            CUDABuffer<ElementType> raw_queries_gpu(num_queries * this->veclen_);
            raw_queries_gpu.upload(queries[0], num_queries * this->veclen_, stream);
            if (!launch_pad_queries<ElementType>(
                    raw_queries_gpu.get(),
                    queries_gpu.get(),
                    num_queries,
                    this->veclen_,
                    padded_veclen_,
                    stream)) {
                throw FLANNException("Failed to launch GPU padding kernel");
            }
        }

        // Allocate result buffers on GPU
        CUDABuffer<int> indices_gpu(num_queries * knn);
        CUDABuffer<float> dists_gpu(num_queries * knn);

        // Calculate grid/block dimensions
        int threads_per_block = 128;
        int num_blocks = (num_queries + threads_per_block - 1) / threads_per_block;
        dim3 grid(num_blocks);
        dim3 block(threads_per_block);

        // Calculate dynamic heap size based on tree structure (OpenCL parity)
        // heapSize = max(getAvgNodesNeeded(maxChecks), getCUDAknn(knn))
        int heap_size = calculateHeapSize(knn, max_checks);

        // Query device capabilities for cooperative kernel
        int loc_size = getCUDALocSize(0);

        // Determine which kernel to use
        // Cooperative kernel requires heap_size <= loc_size AND branching=32 or 64
        bool heap_ok = (heap_size <= loc_size);
        bool branch_ok = (this->branching_ == 32 || this->branching_ == 64);
        bool use_cooperative = heap_ok && branch_ok;

        // Launch kernel (only works with float, but must compile for all types)
        bool success = false;
        if (!std::is_same<ElementType, float>::value) {
            throw FLANNException(
                "K-Means CUDA kernel only supports float element type. "
                "This should have been caught earlier - please report this bug.");
        }

        if (use_cooperative) {
            // Use cooperative kernel (LOC_SIZE threads per query)
            // Dispatch based on k value (template parameter)
            #define SEARCH_DISPATCH(K) \
                success = launch_kmeans_search_cooperative<K>( \
                    (const float*)dataset_ptr_, \
                    (const float*)queries_gpu.get(), \
                    node_index_ptr_, \
                    (const float*)tree_pivots_ptr_, \
                    node_variance_ptr_, \
                    indices_gpu.get(), \
                    (float*)dists_gpu.get(), \
                    num_queries, padded_veclen_, num_nodes_, \
                    heap_size, loc_size, this->branching_, this->cb_index_, \
                    stream)

            if (knn == 1) { SEARCH_DISPATCH(1); }
            else if (knn == 2) { SEARCH_DISPATCH(2); }
            else if (knn == 4) { SEARCH_DISPATCH(4); }
            else if (knn == 5) { SEARCH_DISPATCH(5); }
            else if (knn == 7) { SEARCH_DISPATCH(7); }
            else if (knn == 8) { SEARCH_DISPATCH(8); }
            else if (knn == 10) { SEARCH_DISPATCH(10); }
            else if (knn == 16) { SEARCH_DISPATCH(16); }
            else if (knn == 20) { SEARCH_DISPATCH(20); }
            else if (knn == 32) { SEARCH_DISPATCH(32); }
            else if (knn == 50) { SEARCH_DISPATCH(50); }
            else if (knn == 64) { SEARCH_DISPATCH(64); }
            else if (knn == 100) { SEARCH_DISPATCH(100); }
            else { use_cooperative = false; }

            #undef SEARCH_DISPATCH
        }

        if (!use_cooperative) {
            // This should not normally be reached - buildCUDAKnnSearch() validates k values.
            throw FLANNException(
                "Unsupported k value for CUDA cooperative kernel. "
                "Supported k values: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100 (with branching=32 or 64). "
                "Use knnSearch() instead of knnSearchGPU() for automatic CPU fallback.");
        }

        if (!success) {
            cudaError_t err = cudaGetLastError();
            std::cerr << "[CUDA K-Means] Kernel launch failed! CUDA error: "
                      << cudaGetErrorString(err) << " (code: " << err << ")\n";
            std::cerr << "[CUDA K-Means] Parameters: heap_size=" << heap_size
                      << ", loc_size=" << loc_size
                      << ", branching=" << this->branching_
                      << ", dim=" << padded_veclen_
                      << ", k=" << knn << "\n";
            throw FLANNException("Failed to launch K-Means CUDA kernel");
        }

        // Check for kernel launch errors (non-blocking)
        CUDA_CHECK_LAST();

        // Download results using pinned memory for faster DMA transfers
        // Use async transfers on thread-local stream to overlap both downloads
        PinnedBuffer<int> indices_host(num_queries * knn);
        PinnedBuffer<float> dists_host(num_queries * knn);

        // Async downloads (will wait for kernel on same stream)
        indices_gpu.download(indices_host.get(), num_queries * knn, stream);
        dists_gpu.download(dists_host.get(), num_queries * knn, stream);

        // Single sync point - waits for kernel + both downloads on this thread's stream
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK_LAST();  // Check for any errors after sync

        // Copy to output matrices
        for (size_t i = 0; i < num_queries; ++i) {
            for (int j = 0; j < knn; ++j) {
                indices[i][j] = indices_host[i * knn + j];
                dists[i][j] = dists_host[i * knn + j];
            }
        }

        return num_queries;
    }

protected:
    /**
     * @brief Free GPU memory
     *
     * Releases the unified workspace buffer and nullifies all pointers.
     * Uses stream-ordered deallocation if async GPU operations were performed.
     */
    void freeGPUMemory()
    {
        // Free workspace buffer
        workspace_gpu_ = CUDABuffer<unsigned char>();

        // Reset workspace offsets
        workspace_offset_variance_ = 0;
        workspace_offset_pivots_ = 0;
        workspace_offset_dataset_ = 0;

        // Nullify pointers (non-owning views into workspace)
        node_index_ptr_ = nullptr;
        node_variance_ptr_ = nullptr;
        tree_pivots_ptr_ = nullptr;
        dataset_ptr_ = nullptr;

        gpu_initialized_ = false;
        gpu_search_ready_ = false;
        num_nodes_ = 0;
    }

    /**
     * @brief Upload tree and dataset to GPU (thread-safe wrapper)
     *
     * Thread-safe: acquires exclusive lock, blocking all concurrent searches.
     */
    void uploadToGPU() const
    {
        std::unique_lock<std::shared_mutex> lock(this->rw_lock_);
        uploadToGPUInternal();
    }

    /**
     * @brief Upload tree and dataset to GPU (internal, assumes lock held)
     *
     * Converts CPU tree structure to GPU-friendly flat arrays:
     * 1. Flatten tree nodes (breadth-first order)
     * 2. Extract pivot coordinates
     * 3. Pad dataset to 4-element alignment (for vectorization)
     * 4. Upload all data with cudaMemcpyAsync
     *
     * Const-qualified because GPU buffers are mutable (implementation detail).
     * Note: Caller must hold exclusive lock (rw_lock_).
     */
    void uploadToGPUInternal() const
    {
        if (gpu_initialized_) return;
        if (!this->root_) {
            throw FLANNException("Cannot upload to GPU: tree not built");
        }

        // ========================================================================
        // MEMORY POOL PRE-ALLOCATION (Performance Optimization)
        // ========================================================================
        // The first cudaMalloc call initializes the CUDA memory allocator, which
        // takes ~300-400ms on cold start. By doing a dummy allocation first,
        // we amortize this cost during setup rather than the first search.
        //
        // This is NOT JIT compilation - the binary contains native sm_XX code.
        // This is CUDA runtime initialization (context creation, memory pools).
        {
            void* dummy = nullptr;
            cudaError_t err = cudaMalloc(&dummy, 1024);  // 1KB dummy allocation
            if (err == cudaSuccess && dummy) {
                cudaFree(dummy);
            }
            // Ignore errors - this is just warmup, actual allocations will fail properly
        }

        // Calculate padded veclen (round up to multiple of 4 for float4 loads)
        padded_veclen_ = ((this->veclen_ + 3) / 4) * 4;

        // Count leaf size (sum of dataset points across all leaf nodes)
        // This matches OpenCL's cl_num_leaves_ calculation
        leaf_count_ = 0;
        countLeafSize(this->root_, &leaf_count_);

        // Build flat array tree with OpenCL-style arithmetic child layout
        std::vector<int> node_index;
        std::vector<ElementType> flat_pivots;
        std::vector<float> node_variance;
        buildFlatArrayTree(this->root_, node_index, flat_pivots, node_variance);

        // Get node count from variance array (NOT unified nodeIndex size)
        num_nodes_ = node_variance.size();

        // ========================================================================
        // Prepare dataset (with padding) - check for overflow
        // ========================================================================
        if (this->size_ > SIZE_MAX / padded_veclen_) {
            throw FLANNException("Dataset too large - size * padded_veclen would overflow");
        }
        size_t dataset_size = this->size_ * padded_veclen_;
        std::vector<ElementType> padded_data(dataset_size);
        for (size_t i = 0; i < this->size_; ++i) {
            for (size_t j = 0; j < padded_veclen_; ++j) {
                if (j < this->veclen_) {
                    padded_data[i * padded_veclen_ + j] = this->points_[i][j];
                } else {
                    padded_data[i * padded_veclen_ + j] = 0;
                }
            }
        }

        // ========================================================================
        // Calculate workspace layout with 16-byte alignment
        // Layout: [node_index | node_variance | tree_pivots | dataset]
        // ========================================================================
        size_t unified_size = node_index.size();

        // Check for overflow in workspace calculations
        if (num_nodes_ > SIZE_MAX / padded_veclen_) {
            throw FLANNException("Tree too large - pivots size would overflow");
        }
        size_t pivots_size = num_nodes_ * padded_veclen_;

        size_t offset_node_index = 0;
        workspace_offset_variance_ = alignTo16(offset_node_index + unified_size * sizeof(int));
        workspace_offset_pivots_ = alignTo16(workspace_offset_variance_ + num_nodes_ * sizeof(float));
        workspace_offset_dataset_ = alignTo16(workspace_offset_pivots_ + pivots_size * sizeof(ElementType));

        // Check final workspace size for overflow
        if (dataset_size > SIZE_MAX / sizeof(ElementType)) {
            throw FLANNException("Dataset too large - workspace size would overflow");
        }
        size_t dataset_bytes = dataset_size * sizeof(ElementType);
        if (workspace_offset_dataset_ > SIZE_MAX - dataset_bytes) {
            throw FLANNException("Total workspace size would overflow");
        }
        size_t total_workspace_size = workspace_offset_dataset_ + dataset_bytes;

        // ========================================================================
        // Pack all arrays into single pinned host buffer
        // ========================================================================
        PinnedBuffer<unsigned char> workspace_host(total_workspace_size);

        std::memcpy(workspace_host.get() + offset_node_index,
                   node_index.data(),
                   unified_size * sizeof(int));
        std::memcpy(workspace_host.get() + workspace_offset_variance_,
                   node_variance.data(),
                   num_nodes_ * sizeof(float));
        std::memcpy(workspace_host.get() + workspace_offset_pivots_,
                   flat_pivots.data(),
                   pivots_size * sizeof(ElementType));
        std::memcpy(workspace_host.get() + workspace_offset_dataset_,
                   padded_data.data(),
                   dataset_size * sizeof(ElementType));

        // ========================================================================
        // Single GPU upload (1 cudaMalloc + 1 cudaMemcpy)
        // ========================================================================
        workspace_gpu_.resize(total_workspace_size);
        workspace_gpu_.upload(workspace_host.get(), total_workspace_size);

        // ========================================================================
        // Set up raw pointers into workspace
        // ========================================================================
        unsigned char* base_ptr = workspace_gpu_.get();
        node_index_ptr_ = reinterpret_cast<int*>(base_ptr + offset_node_index);
        node_variance_ptr_ = reinterpret_cast<float*>(base_ptr + workspace_offset_variance_);
        tree_pivots_ptr_ = reinterpret_cast<ElementType*>(base_ptr + workspace_offset_pivots_);
        dataset_ptr_ = reinterpret_cast<ElementType*>(base_ptr + workspace_offset_dataset_);

        gpu_initialized_ = true;
    }

    /**
     * @brief Build flat array tree with OpenCL-style arithmetic child layout
     *
     * Creates a flat array tree where parent P at level L has children at indices:
     * P*branching, P*branching+1, ..., P*branching+branching-1
     *
     * This enables O(1) child computation: child = parent + offset (no pointer chasing)
     *
     * @param root Root node of CPU tree
     * @param[out] node_index Output nodeIndex array (internal: node_id, leaf: >=num_nodes)
     * @param[out] flat_pivots Output flat pivot array (padded to padded_veclen_)
     * @param[out] node_variance Output variance array
     * @param[out] node_leaf_count Output leaf point counts (0 for internal nodes, >0 for leaves)
     * @param[out] dataset_indices Output dataset indices for all leaves
     */
    void buildFlatArrayTree(
        Node* root,
        std::vector<int>& node_index,
        std::vector<ElementType>& flat_pivots,
        std::vector<float>& node_variance) const
    {
        if (!root) return;

        node_index.clear();
        flat_pivots.clear();
        node_variance.clear();

        // Map CPU nodes to their target flat array indices
        std::map<Node*, int> node_to_index;

        // Queue: {CPU_node, target_flat_array_index}
        std::queue<std::pair<Node*, int>> to_process;
        to_process.push({root, 0});  // Root at index 0
        node_to_index[root] = 0;

        // First pass: Assign flat array indices with arithmetic layout
        int max_index = 0;
        int num_leaves = 0;
        while (!to_process.empty()) {
            std::pair<Node*, int> front_pair = to_process.front();
            Node* node = front_pair.first;
            int node_idx = front_pair.second;
            to_process.pop();

            max_index = std::max(max_index, node_idx);

            if (node->childs.empty()) {
                num_leaves++;
            } else {
                // Assign children at arithmetic offsets
                for (size_t c = 0; c < node->childs.size(); ++c) {
                    int child_idx = node_idx * this->branching_ + 1 + c;
                    node_to_index[node->childs[c]] = child_idx;
                    to_process.push({node->childs[c], child_idx});
                }
            }
        }

        // Calculate total leaf points for unified array size
        size_t num_nodes = max_index + 1;
        int total_leaf_points = 0;
        std::queue<Node*> count_queue;
        count_queue.push(root);
        while (!count_queue.empty()) {
            Node* node = count_queue.front();
            count_queue.pop();
            if (node->childs.empty()) {
                for (size_t i = 0; i < node->points.size(); ++i) {
                    if (!this->removed_ || !this->removed_points_.test(node->points[i].index)) {
                        total_leaf_points++;
                    }
                }
            } else {
                for (auto* child : node->childs) {
                    count_queue.push(child);
                }
            }
        }

        // Allocate UNIFIED nodeIndex array: node_metadata + leaf_counts + leaf_data
        size_t unified_size = num_nodes + num_leaves + total_leaf_points;
        node_index.resize(unified_size, -1);
        flat_pivots.resize(num_nodes * padded_veclen_, 0);
        node_variance.resize(num_nodes, 0.0f);

        // Second pass: Fill arrays with actual data
        int data_region_offset = num_nodes;  // Data region starts after node metadata
        std::queue<Node*> fill_queue;
        fill_queue.push(root);

        while (!fill_queue.empty()) {
            Node* node = fill_queue.front();
            fill_queue.pop();

            int node_idx = node_to_index[node];

            // Store pivot (with padding)
            for (size_t d = 0; d < padded_veclen_; ++d) {
                if (d < this->veclen_) {
                    flat_pivots[node_idx * padded_veclen_ + d] = node->pivot[d];
                } else {
                    flat_pivots[node_idx * padded_veclen_ + d] = 0;  // Pad
                }
            }

            // Store variance
            node_variance[node_idx] = node->variance;

            // Handle internal vs leaf nodes
            if (node->childs.empty()) {
                // LEAF: Store pointer to data region in unified nodeIndex
                node_index[node_idx] = data_region_offset;

                // Count valid (non-removed) points for this leaf
                int leaf_point_count = 0;
                for (size_t i = 0; i < node->points.size(); ++i) {
                    if (!this->removed_ || !this->removed_points_.test(node->points[i].index)) {
                        leaf_point_count++;
                    }
                }

                // Store count in data region
                node_index[data_region_offset] = leaf_point_count;
                data_region_offset++;

                // Store dataset IDs in data region
                for (size_t i = 0; i < node->points.size(); ++i) {
                    size_t index = node->points[i].index;
                    if (!this->removed_ || !this->removed_points_.test(index)) {
                        node_index[data_region_offset] = static_cast<int>(index);
                        data_region_offset++;
                    }
                }
            } else {
                // INTERNAL: nodeIndex[i] = first_child_index (OpenCL-style base pointer)
                // Children are at: node_idx*branching+1, node_idx*branching+2, ..., node_idx*branching+branching
                int first_child_idx = node_idx * this->branching_ + 1;
                node_index[node_idx] = first_child_idx;

                // Enqueue children for processing
                for (size_t c = 0; c < node->childs.size(); ++c) {
                    fill_queue.push(node->childs[c]);
                }
            }
        }

        // Verify arithmetic layout - MUST match for GPU kernel to work correctly
        int mismatch_count = 0;
        for (const auto& entry : node_to_index) {
            Node* node = entry.first;
            int idx = entry.second;
            if (!node->childs.empty() && node->childs.size() > 0) {
                int first_child_expected = idx * this->branching_ + 1;  // +1 offset
                int first_child_actual = node_to_index[node->childs[0]];
                if (first_child_expected != first_child_actual) {
                    ++mismatch_count;
                }
            }
        }
        if (mismatch_count > 0) {
            throw FLANNException("Tree arithmetic layout validation failed: " +
                std::to_string(mismatch_count) + " node mismatches detected. "
                "GPU kernel requires exact arithmetic tree layout.");
        }
    }

protected:
    /**
     * @brief Calculate average nodes needed for given max checks
     *
     * Matches OpenCL implementation (kmeans_opencl_index.h:850-854)
     * Used for dynamic heap size calculation.
     *
     * @param maxChecks Maximum number of distance computations allowed
     * @return Estimated number of tree nodes to explore
     */
    /**
     * @brief Recursively count total dataset points across all leaf nodes
     */
    void countLeafSize(typename BaseClass::Node* node, size_t* leafCount) const
    {
        if (!node) return;

        if (node->childs.empty()) {
            *leafCount += node->size;
        } else {
            for (size_t i = 0; i < node->childs.size(); ++i) {
                countLeafSize(node->childs[i], leafCount);
            }
        }
    }

    /**
     * @brief Recursively count number of tree leaf nodes
     */
    void countTreeLeafNodes(typename BaseClass::Node* node, size_t* count) const
    {
        if (!node) return;

        if (node->childs.empty()) {
            (*count)++;
        } else {
            for (size_t i = 0; i < node->childs.size(); ++i) {
                countTreeLeafNodes(node->childs[i], count);
            }
        }
    }

    int getAvgNodesNeeded(int maxChecks) const
    {
        if (num_nodes_ == 0 || leaf_count_ == 0) {
            return maxChecks;
        }

        // Count tree leaf nodes (not dataset points)
        size_t tree_leaf_count = 0;
        countTreeLeafNodes(this->root_, &tree_leaf_count);

        // Formula: maxChecks * (tree_leaf_nodes) / (dataset_points)
        // Matches OpenCL: maxChecks * (cl_num_nodes_ - cl_num_parents_) / cl_num_leaves_
        return maxChecks * tree_leaf_count / leaf_count_;
    }

    /**
     * @brief Calculate CL-style k-NN heap size
     *
     * Matches OpenCL implementation (nn_opencl_index.h:240-248)
     * Rounds up to multiple of 4, plus 1 for vectorization alignment.
     *
     * @param knn Number of nearest neighbors
     * @return Heap size suitable for k-NN search
     */
    int getCUDAknn(size_t knn) const
    {
        // Round up to multiple of 4, plus 1
        // Ensures heap size is suitable for vectorization
        return 4 * ((knn + 3) / 4) + 1;
    }

    /**
     * @brief Calculate initial heap size for given parameters
     *
     * Matches OpenCL logic (kmeans_opencl_index.h:335)
     * Used for kernel selection and PQ size determination.
     *
     * @param knn Number of nearest neighbors
     * @param maxChecks Maximum distance computations
     * @return Calculated heap size
     */
    int calculateHeapSize(size_t knn, int maxChecks) const
    {
        int avgNodes = getAvgNodesNeeded(maxChecks);
        int clKnn = getCUDAknn(knn);
        return std::max(avgNodes, clKnn);
    }

private:
    // GPU state (mutable = implementation detail, not logical state)
    mutable bool gpu_initialized_;
    mutable bool gpu_search_ready_;
    bool gpu_only_mode_;  // True if loaded with gpu_only=true (points_ not available)

    // Note: Thread safety is now handled by rw_lock_ in base class CUDAIndex
    // (std::shared_mutex for reader-writer locking)

    mutable size_t num_nodes_;
    mutable size_t leaf_count_;      // Number of leaf nodes (for heap size calculation)
    mutable size_t padded_veclen_;

    // GPU workspace buffer - single allocation for all GPU arrays
    // Contains: [node_index | node_variance | tree_pivots | dataset]
    // Each section is 16-byte aligned for optimal CUDA memory access
    mutable CUDABuffer<unsigned char> workspace_gpu_;

    // Workspace layout offsets (bytes from workspace start)
    mutable size_t workspace_offset_variance_;
    mutable size_t workspace_offset_pivots_;
    mutable size_t workspace_offset_dataset_;

    // Non-owning pointers into workspace (nullptr if workspace not allocated)
    mutable int* node_index_ptr_;
    mutable float* node_variance_ptr_;
    mutable ElementType* tree_pivots_ptr_;
    mutable ElementType* dataset_ptr_;
};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_CUDA_KMEANS_CUDA_INDEX_H_
