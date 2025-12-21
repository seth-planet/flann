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

#ifndef FLANN_HIERARCHICAL_CUDA_INDEX_H_
#define FLANN_HIERARCHICAL_CUDA_INDEX_H_

#ifdef FLANN_USE_CUDA

#include <queue>
#include <utility>  // For std::swap
#include <cstring>  // For memcpy
#include <cstdint>  // For SIZE_MAX
#include <memory>   // For std::unique_ptr
#include <cuda_runtime.h>
#include <mutex>         // For std::unique_lock
#include <shared_mutex>  // For std::shared_mutex, std::shared_lock

#include "flann/algorithms/hierarchical_clustering_index.h"
#include "flann/algorithms/cuda/cuda_utils.h"
#include "flann/algorithms/cuda/nn_cuda_index.h"
#include "flann/util/gpu_saving.h"

namespace flann {
namespace cuda {

// Forward declare kernel launch functions (defined in hierarchical_cuda_kernels.cu)
// These are always forward declared - definition is in the .cu compilation unit
bool launch_hierarchical_search_cooperative(
    const unsigned char* dataset,
    const unsigned char* queries,
    const int* device_node_index,  // Interleaved [pivot, child_ptr] pairs
    int* result_indices,
    int* result_distances,
    size_t num_queries,
    size_t padded_bytes,
    size_t actual_bytes,
    size_t num_nodes,
    int k,
    int num_trees,
    int branching,
    cudaStream_t stream = nullptr);

// Note: launch_pad_queries is declared in kmeans_cuda_index.h (included first)

/**
 * @brief Index parameters for Hierarchical CUDA index
 *
 * Extends base HierarchicalClusteringIndexParams with CUDA-specific settings.
 */
struct HierarchicalCUDAIndexParams : public HierarchicalClusteringIndexParams
{
    HierarchicalCUDAIndexParams(
        int branching = 32,
        flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
        int trees = 4,
        int leaf_max_size = 100)
        : HierarchicalClusteringIndexParams(branching, centers_init, trees, leaf_max_size)
    {
        (*this)["algorithm"] = FLANN_INDEX_HIERARCHICAL_CUDA;
    }
};

/**
 * @brief CUDA-accelerated Hierarchical Clustering Index
 *
 * GPU-accelerated version of hierarchical clustering for binary descriptors.
 * Uses Hamming distance for similarity computation.
 *
 * THREAD SAFETY: This index IS thread-safe for concurrent searches.
 * Multiple threads can safely call knnSearch() / knnSearchGPU() on
 * the same index instance. Mutations (addPoints, removePoint) acquire
 * exclusive lock and block all searches during modification.
 *
 * **Architecture:**
 * - Dual inheritance from HierarchicalClusteringIndex (CPU) and CUDAIndex (GPU marker)
 * - Multi-tree structure for improved recall
 * - Breadth-first tree flattening for GPU-friendly access patterns
 * - Thread-per-query design with best-first search
 *
 * **Usage Pattern:**
 * ```cpp
 * HierarchicalCUDAIndex<Hamming<unsigned char>> index(dataset, params);
 * index.buildIndex();                    // Build tree on CPU
 * index.buildCUDAKnnSearch(k, params);  // Upload to GPU (one-time cost)
 * index.knnSearch(queries, ...);         // Fast GPU searches
 * ```
 *
 * **Performance Characteristics:**
 * - Best for binary descriptors (BRIEF, ORB, BRISK, etc.)
 * - Expected speedup: 5-15x vs CPU for large datasets
 * - Memory overhead: ~2x dataset size for GPU buffers
 * - One-time upload cost: 0.3-1.0s for 100K points
 *
 * @tparam Distance Distance functor (typically Hamming<unsigned char>)
 */
template <typename Distance>
class HierarchicalCUDAIndex : public HierarchicalClusteringIndex<Distance>, public CUDAIndex
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    typedef HierarchicalClusteringIndex<Distance> BaseClass;
    typedef typename BaseClass::Node* NodePtr;

    // Bring base class knnSearch overloads into scope (prevents C++ name hiding)
    using BaseClass::knnSearch;

    // ========================================================================
    // Constructors and Lifecycle
    // ========================================================================

    /**
     * @brief Default constructor (no dataset)
     */
    HierarchicalCUDAIndex(const IndexParams& params = HierarchicalCUDAIndexParams(),
                          Distance d = Distance())
        : BaseClass(params, d),
          CUDAIndex(),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_padded_bytes_(0),
          workspace_gpu_(),
          workspace_offset_node_index_(0),
          dataset_ptr_(nullptr),
          node_index_ptr_(nullptr)
    {
    }

    /**
     * @brief Constructor with dataset
     *
     * @param inputData Dataset matrix (N x D) where D is descriptor length in bytes
     * @param params Index parameters (branching, trees, etc.)
     * @param d Distance functor (typically Hamming<unsigned char>)
     */
    HierarchicalCUDAIndex(const Matrix<ElementType>& inputData,
                          const IndexParams& params = HierarchicalCUDAIndexParams(),
                          Distance d = Distance())
        : BaseClass(inputData, params, d),
          CUDAIndex(),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_padded_bytes_(0),
          workspace_gpu_(),
          workspace_offset_node_index_(0),
          dataset_ptr_(nullptr),
          node_index_ptr_(nullptr)
    {
    }

    /**
     * @brief Copy constructor from CPU index
     *
     * Allows converting a CPU-only hierarchical index to GPU-accelerated version.
     * Note: GPU buffers are NOT copied - they start empty.
     * Call buildCUDAKnnSearch() after copy to upload to GPU.
     */
    HierarchicalCUDAIndex(const HierarchicalClusteringIndex<Distance>& other)
        : BaseClass(other),
          CUDAIndex(),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_padded_bytes_(0),
          workspace_gpu_(),
          workspace_offset_node_index_(0),
          dataset_ptr_(nullptr),
          node_index_ptr_(nullptr)
    {
        // GPU buffers start empty - user must call buildCUDAKnnSearch()
    }

    /**
     * @brief Copy constructor from same class
     *
     * Note: GPU buffers are NOT copied, only CPU tree structure.
     * Call buildCUDAKnnSearch() after copy to re-upload to GPU.
     */
    HierarchicalCUDAIndex(const HierarchicalCUDAIndex& other)
        : BaseClass(other),
          CUDAIndex(),
          gpu_initialized_(false),
          gpu_search_ready_(false),
          gpu_only_mode_(false),
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_padded_bytes_(0),
          workspace_gpu_(),
          workspace_offset_node_index_(0),
          dataset_ptr_(nullptr),
          node_index_ptr_(nullptr)
    {
        // GPU buffers start empty - user must call buildCUDAKnnSearch()
    }

    /**
     * @brief Destructor
     */
    ~HierarchicalCUDAIndex()
    {
    }

    /**
     * @brief Assignment operator (copy-and-swap idiom)
     */
    HierarchicalCUDAIndex& operator=(HierarchicalCUDAIndex other)
    {
        this->swap(other);
        return *this;
    }

    /**
     * @brief Clone this index (virtual copy constructor)
     */
    BaseClass* clone() const override
    {
        return new HierarchicalCUDAIndex(*this);
    }

    /**
     * @brief Return index type identifier
     */
    flann_algorithm_t getType() const override
    {
        return FLANN_INDEX_HIERARCHICAL_CUDA;
    }

    // ========================================================================
    // GPU Initialization and Search Preparation
    // ========================================================================

    /**
     * @brief Prepare index for GPU search by uploading to device memory
     *
     * **One-time setup** that flattens the hierarchical tree and uploads:
     * - Flattened node structure (breadth-first traversal)
     * - Pivot descriptors (binary vectors)
     * - Dataset descriptors
     * - Node connectivity information
     *
     * Must be called after buildIndex() but before GPU searches.
     * Amortize this cost over many searches for best performance.
     *
     * @param knn Number of nearest neighbors to prepare for
     * @param params Search parameters (checks, etc.)
     *
     * @throws FLANNException if index not built yet
     */
    void buildCUDAKnnSearch(int knn, const SearchParams& params = SearchParams())
    {
        (void)params;  // Unused for hierarchical (kept for API consistency)

        // Validate K is supported before any work
        if (!isKValueSupportedForGPU(knn)) {
            throw FLANNException(
                "Unsupported k=" + std::to_string(knn) + " for CUDA hierarchical search.\n"
                "Supported k values: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128\n"
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
     */
    bool hasGPUFormat() const
    {
        return gpu_initialized_;
    }

    /**
     * @brief Check if a K value is supported for GPU search
     *
     * Hierarchical CUDA supports: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128
     *
     * @param k Number of nearest neighbors
     * @return true if k is supported for GPU search
     */
    bool isKValueSupportedForGPU(int k) const
    {
        return k == 1 || k == 2 || k == 3 || k == 4 || k == 5 || k == 8 ||
               k == 10 || k == 12 || k == 16 || k == 20 || k == 24 || k == 32 ||
               k == 50 || k == 64 || k == 100 || k == 128;
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
        if (this->tree_roots_.empty() && !gpu_initialized_) {
            throw FLANNException("Cannot prepare GPU search: index not built yet. "
                               "Call buildIndex() first.");
        }

        // Check that trees are large enough (not degenerate) - skip if loaded from GPU format
        if (!this->tree_roots_.empty()) {
            for (size_t i = 0; i < this->tree_roots_.size(); ++i) {
                if (this->tree_roots_[i]->childs.empty()) {
                    throw FLANNException("Tree " + std::to_string(i) + " is degenerate "
                                       "(root is a leaf). Try reducing branching factor or "
                                       "increasing dataset size.");
                }
            }
        }

        if (!gpu_initialized_) {
            uploadToGPUInternal();
        }
        gpu_search_ready_ = true;
    }

    /**
     * @brief Check if index is in GPU-only mode
     *
     * In GPU-only mode (loaded with gpu_only=true), the CPU-side points_
     * array is not populated. This saves memory but means:
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
     * @brief Convert index to GPU-optimized format
     *
     * Ensures GPU buffers are populated for GPU-optimized saving.
     * Also discards the CPU tree structure to reduce memory usage.
     *
     * @throws FLANNException if tree not built
     */
    void convertToGPUFormat()
    {
        if (this->tree_roots_.empty()) {
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
     * @brief Save index to file (GPU v2.0 format if available, CPU format otherwise)
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
     * - "FLANN_INDEX_v1.1" with FLANN_INDEX_HIERARCHICAL → CPU format
     *   (loads via temp CPU index, then copies to CUDA index)
     * - "FLANN_INDEX_v1.1" with FLANN_INDEX_HIERARCHICAL_CUDA → CPU format
     *   (direct load via BaseClass)
     *
     * This allows loading indices saved by either:
     * - HierarchicalCUDAIndex (saves as HIERARCHICAL_CUDA or GPU v2.0)
     * - HierarchicalClusteringIndex (saves as HIERARCHICAL)
     *
     * **Important:** When loading pure CPU HIERARCHICAL format, the saved file
     * must have been created with `save_dataset=true` so the dataset is embedded.
     *
     * @param stream Input file stream. Should be opened with "rb" mode.
     *               Stream position will be reset to 0 internally for CPU format.
     *               The stream is NOT closed by this method.
     *
     * @throws FLANNException if stream is invalid, file format is unrecognized,
     *         dataset not embedded (for CPU HIERARCHICAL), or data type mismatch.
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

            if (header.h.index_type == FLANN_INDEX_HIERARCHICAL) {
                // CPU HIERARCHICAL format - load via temporary CPU index
                // This avoids type mismatch error in NNIndex::serialize()
                freeGPUMemory();

                // Load into temporary CPU index (type matches file)
                // Note: saved file must have save_dataset=true, otherwise load fails
                HierarchicalClusteringIndex<Distance> temp_cpu;
                temp_cpu.loadIndex(stream);

                // Use existing copy constructor + swap idiom
                // The copy constructor handles all tree copying via BaseClass(other)
                HierarchicalCUDAIndex temp_cuda(temp_cpu);
                this->swap(temp_cuda);

            } else if (header.h.index_type == FLANN_INDEX_HIERARCHICAL_CUDA) {
                // CPU v1.1 format saved as CUDA type - direct load works
                freeGPUMemory();
                BaseClass::loadIndex(stream);
            } else {
                throw FLANNException(
                    "Unsupported index type for HierarchicalCUDAIndex. "
                    "Expected FLANN_INDEX_HIERARCHICAL or FLANN_INDEX_HIERARCHICAL_CUDA, got: " +
                    std::to_string(header.h.index_type));
            }
        }
    }

    /**
     * @brief Invalidate GPU data structures after adding points
     *
     * Thread-safe: acquires exclusive lock, blocking all concurrent searches.
     * Adding points changes the tree structure, so GPU buffers must be rebuilt.
     *
     * @throws FLANNException if index is in GPU-only mode
     */
    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2) override
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
     * @brief Invalidate GPU data structures after removing a point
     *
     * Thread-safe: acquires exclusive lock, blocking all concurrent searches.
     *
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
     * @brief Swap this index with another (for copy-and-swap idiom)
     */
    void swap(HierarchicalCUDAIndex& other)
    {
        BaseClass::swap(other);

        // Swap GPU state
        std::swap(gpu_initialized_, other.gpu_initialized_);
        std::swap(gpu_search_ready_, other.gpu_search_ready_);
        std::swap(gpu_only_mode_, other.gpu_only_mode_);
        std::swap(gpu_num_trees_, other.gpu_num_trees_);
        std::swap(gpu_num_nodes_, other.gpu_num_nodes_);
        std::swap(gpu_padded_bytes_, other.gpu_padded_bytes_);

        // Swap GPU workspace and pointers
        std::swap(workspace_gpu_, other.workspace_gpu_);
        std::swap(workspace_offset_node_index_, other.workspace_offset_node_index_);
        std::swap(dataset_ptr_, other.dataset_ptr_);
        std::swap(node_index_ptr_, other.node_index_ptr_);
    }

    // ========================================================================
    // k-NN Search (Automatic CPU/GPU Dispatch)
    // ========================================================================

    /**
     * @brief Perform k-NN search (automatically uses GPU if prepared)
     *
     * **Automatic dispatch logic:**
     * - If buildCUDAKnnSearch() was called → GPU search
     * - Otherwise → fallback to CPU search
     *
     * GPU search is faster for:
     * - Large datasets (>50K points)
     * - Batch queries (>100 queries)
     * - When upload cost is amortized
     *
     * @param queries Query descriptors (Q x D)
     * @param indices Output indices (Q x knn)
     * @param dists Output distances (Q x knn)
     * @param knn Number of nearest neighbors
     * @param params Search parameters (checks, etc.)
     * @return Number of neighbors found per query
     */
    int knnSearch(const Matrix<ElementType>& queries,
                  Matrix<size_t>& indices,
                  Matrix<DistanceType>& dists,
                  size_t knn,
                  const SearchParams& params) const override
    {
        if (gpu_search_ready_) {
            return knnSearchGPU(queries, indices, dists, knn, params);
        } else {
            // Fallback to CPU search
            return BaseClass::knnSearch(queries, indices, dists, knn, params);
        }
    }

    /**
     * @brief Batch k-NN search (overload for std::vector queries)
     */
    int knnSearch(const Matrix<ElementType>& queries,
                  std::vector<std::vector<size_t>>& indices,
                  std::vector<std::vector<DistanceType>>& dists,
                  size_t knn,
                  const SearchParams& params) const override
    {
        if (gpu_search_ready_) {
            // Validate output vector sizes
            if (indices.size() < queries.rows || dists.size() < queries.rows) {
                throw FLANNException("Output vectors too small for query count");
            }

            // Use RAII vectors for exception safety (no manual delete needed)
            std::vector<size_t> indices_storage(queries.rows * knn);
            std::vector<DistanceType> dists_storage(queries.rows * knn);
            flann::Matrix<size_t> indices_mat(indices_storage.data(), queries.rows, knn);
            flann::Matrix<DistanceType> dists_mat(dists_storage.data(), queries.rows, knn);

            int result = knnSearchGPU(queries, indices_mat, dists_mat, knn, params);

            // Copy results back to vector format
            for (size_t i = 0; i < queries.rows; ++i) {
                indices[i].resize(knn);
                dists[i].resize(knn);
                for (size_t j = 0; j < knn; ++j) {
                    indices[i][j] = indices_mat[i][j];
                    dists[i][j] = dists_mat[i][j];
                }
            }

            return result;
        } else {
            return BaseClass::knnSearch(queries, indices, dists, knn, params);
        }
    }

    /**
     * @brief GPU-direct k-NN search with device pointers (zero-copy)
     *
     * Performs k-nearest neighbor search using device-resident data,
     * eliminating CPU<->GPU memory transfers for queries and results.
     * Ideal for GPU-resident pipelines where data never leaves device memory.
     *
     * @param d_queries Device pointer to query descriptors [num_queries x veclen]
     *                  Must be contiguous row-major layout.
     * @param d_indices Device pointer for output indices [num_queries x knn]
     *                  Will contain point indices as int.
     * @param d_dists   Device pointer for output distances [num_queries x knn]
     *                  Hamming distances as int (popcount).
     * @param num_queries Number of query vectors
     * @param knn Number of nearest neighbors to find
     * @param params Search parameters (checks value currently unused)
     * @param stream CUDA stream for async execution (nullptr = thread-local stream)
     *
     * @return Number of queries processed
     * @throws FLANNException if GPU not initialized or k unsupported
     *
     * @note Caller must synchronize on stream before reading results.
     * @note If veclen % 4 != 0, queries are padded internally.
     */
    int knnSearchGPUDirect(
        const ElementType* d_queries,
        int* d_indices,
        int* d_dists,
        size_t num_queries,
        size_t knn,
        const SearchParams& params = SearchParams(),
        cudaStream_t stream = nullptr) const
    {
        // 1. Validate GPU initialization
        if (!gpu_initialized_) {
            throw FLANNException("GPU index not initialized. Call buildCUDAKnnSearch() or prepareGPUIndex() first.");
        }

        // 2. Validate K value is supported
        if (!isKValueSupportedForGPU(static_cast<int>(knn))) {
            throw FLANNException(
                "Unsupported k=" + std::to_string(knn) + " for CUDA hierarchical search.\n"
                "Supported k values: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128\n"
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

        // 6. Stream selection
        cudaStream_t exec_stream = (stream != nullptr) ? stream : this->getThreadStream();

        // 7. Handle query padding if veclen % 4 != 0
        int padded_bytes = 4 * ((this->veclen_ + 3) / 4);
        const ElementType* kernel_queries = d_queries;
        CUDABuffer<ElementType> padded_queries;

        if ((size_t)padded_bytes != this->veclen_) {
            // Allocate and pad queries on GPU
            padded_queries = CUDABuffer<ElementType>(num_queries * padded_bytes);
            if (!launch_pad_queries<ElementType>(
                    d_queries,
                    padded_queries.get(),
                    num_queries,
                    this->veclen_,
                    padded_bytes,
                    exec_stream)) {
                throw FLANNException("Failed to launch GPU padding kernel");
            }
            kernel_queries = padded_queries.get();
        }

        // 8. Launch cooperative search kernel
        bool success = launch_hierarchical_search_cooperative(
            reinterpret_cast<const unsigned char*>(dataset_ptr_),
            reinterpret_cast<const unsigned char*>(kernel_queries),
            node_index_ptr_,
            d_indices,   // Direct device output
            d_dists,     // Direct device output
            num_queries,
            padded_bytes,
            this->veclen_,
            gpu_num_nodes_,
            knn,
            gpu_num_trees_,
            this->branching_,
            exec_stream
        );

        if (!success) {
            throw FLANNException("Kernel launch failed for k=" + std::to_string(knn));
        }

        // 9. Check for kernel launch errors (non-blocking)
        CUDA_CHECK_LAST();

        // 10. No synchronization - caller is responsible for stream sync
        return static_cast<int>(num_queries);
    }

protected:
    // ========================================================================
    // GPU Memory Management
    // ========================================================================

    /**
     * @brief Free all GPU memory buffers
     *
     * Called when tree structure changes (add/remove points) or on destruction.
     */
    void freeGPUMemory()
    {
        // Free workspace buffer (single deallocation)
        workspace_gpu_ = CUDABuffer<unsigned char>();

        // Reset workspace offset
        workspace_offset_node_index_ = 0;
        gpu_padded_bytes_ = 0;

        // Nullify pointers (non-owning views into workspace)
        dataset_ptr_ = nullptr;
        node_index_ptr_ = nullptr;

        gpu_initialized_ = false;
        gpu_search_ready_ = false;
        gpu_num_nodes_ = 0;
        gpu_num_trees_ = 0;
    }

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

        // Use stored padded_bytes
        size_t padded_bytes = gpu_padded_bytes_;
        size_t dataset_bytes = this->size_ * padded_bytes * sizeof(ElementType);

        // Calculate node_index size from workspace layout
        size_t total_workspace_bytes = workspace_gpu_.size_bytes();
        size_t node_index_bytes = total_workspace_bytes - workspace_offset_node_index_;
        size_t node_index_size = node_index_bytes / sizeof(int);

        // Build v2.0 header with pre-computed offsets
        GPUIndexHeaderV2 header;
        header.h.data_type = flann_datatype_value<ElementType>::value;
        header.h.index_type = FLANN_INDEX_HIERARCHICAL_GPU_SAVED;
        header.h.rows = this->size_;
        header.h.cols = this->veclen_;
        header.h.padded_veclen = padded_bytes;
        header.h.num_nodes = gpu_num_nodes_;
        header.h.node_index_size = node_index_size;
        header.h.leaf_count = 0;  // Not used for hierarchical
        header.h.branching = this->branching_;
        header.h.trees = gpu_num_trees_;
        header.h.leaf_max_size = this->leaf_max_size_;
        header.h.centers_init = this->centers_init_;

        // V2.0 specific: pre-computed workspace layout
        // Hierarchical layout: [dataset | node_index]
        header.h.workspace_total_size = total_workspace_bytes;
        header.h.offset_node_index = workspace_offset_node_index_;
        header.h.offset_variance = 0;  // Not used for hierarchical
        header.h.offset_pivots = 0;    // Not used for hierarchical (pivots in dataset)
        header.h.offset_dataset = 0;   // Dataset at start

        sa & header;

        // Download entire GPU workspace as single blob
        PinnedBuffer<unsigned char> workspace_host(total_workspace_bytes);

        CUDA_CHECK(cudaMemcpy(workspace_host.get(), workspace_gpu_.get(),
                             total_workspace_bytes, cudaMemcpyDeviceToHost));

        // Serialize single blob (includes alignment padding)
        sa & serialization::make_binary_object(
            workspace_host.get(),
            total_workspace_bytes);

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
        if (header.h.index_type != FLANN_INDEX_HIERARCHICAL_GPU_SAVED) {
            throw FLANNException("Index type mismatch: expected HIERARCHICAL_GPU_SAVED");
        }

        // Restore metadata from header
        this->size_ = header.h.rows;
        this->veclen_ = header.h.cols;
        this->size_at_build_ = header.h.rows;
        int padded_bytes = header.h.padded_veclen;
        gpu_padded_bytes_ = padded_bytes;
        gpu_num_nodes_ = header.h.num_nodes;
        gpu_num_trees_ = header.h.trees;
        this->branching_ = header.h.branching;
        this->leaf_max_size_ = header.h.leaf_max_size;
        this->centers_init_ = header.h.centers_init;

        // V2.0 OPTIMIZATION: Read workspace offsets directly from header
        workspace_offset_node_index_ = header.h.offset_node_index;
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
        dataset_ptr_ = reinterpret_cast<ElementType*>(base_ptr + header.h.offset_dataset);
        node_index_ptr_ = reinterpret_cast<int*>(base_ptr + workspace_offset_node_index_);

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
                workspace_host.get() + header.h.offset_dataset);
            for (size_t i = 0; i < this->size_; ++i) {
                this->points_[i] = this->data_ptr_ + i * this->veclen_;
                std::memcpy(this->points_[i],
                           &dataset_in_workspace[i * padded_bytes],
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
        this->index_params_["algorithm"] = FLANN_INDEX_HIERARCHICAL_CUDA;
        this->index_params_["branching"] = this->branching_;
        this->index_params_["trees"] = gpu_num_trees_;
        this->index_params_["leaf_max_size"] = this->leaf_max_size_;
        this->index_params_["centers_init"] = this->centers_init_;

        gpu_initialized_ = true;
    }

    /**
     * @brief Discard CPU tree to save memory
     */
    void discardCPUTree()
    {
        // Clear tree roots (frees memory pool)
        for (size_t i = 0; i < this->tree_roots_.size(); ++i) {
            // Nodes are pool-allocated, pool handles cleanup
        }
        this->tree_roots_.clear();
        this->pool_.free();
    }

    // ========================================================================
    // Tree Upload to GPU
    // ========================================================================

    /**
     * @brief Upload tree structure to GPU memory
     *
     * **Flattening strategy (breadth-first):**
     * 1. Count nodes in all trees
     * 2. Allocate flat arrays for nodes, dataset
     * 3. Traverse trees breadth-first, copying data
     * 4. Upload to GPU with cudaMemcpy
     *
     * **Memory layout (interleaved for cache locality):**
     * - node_index: [pivot_0, child_ptr_0, pivot_1, child_ptr_1, ..., leaf_data...]
     *   - Even indices (i*2): pivot_index for node i
     *   - Odd indices (i*2+1): child_ptr for node i (child_start or leaf_data_ptr)
     * - Dataset: Packed binary descriptors (padded to multiples of 4 bytes)
     *
     * @throws FLANNException on CUDA errors
     */
    void uploadToGPU()
    {
        std::unique_lock<std::shared_mutex> lock(this->rw_lock_);
        uploadToGPUInternal();
    }

    /**
     * @brief Internal GPU upload (assumes lock held)
     *
     * Note: Caller must hold exclusive lock (rw_lock_).
     */
    void uploadToGPUInternal()
    {
        if (gpu_initialized_) {
            return;  // Already uploaded
        }

        // Count nodes in all trees (breadth-first traversal)
        // CRITICAL FIX: Count root CHILDREN only (roots are implicit, not stored)
        int num_nodes = 0;
        int num_parents = 0;
        int num_leaves = 0;

        gpu_num_trees_ = this->tree_roots_.size();  // Store number of trees for kernel
        for (size_t i = 0; i < this->tree_roots_.size(); ++i) {
            NodePtr root = this->tree_roots_[i];
            // Count children of root, NOT root itself
            for (size_t j = 0; j < root->childs.size(); ++j) {
                countNodes(root->childs[j], &num_nodes, &num_parents, &num_leaves);
            }
        }

        // Sanity checks
        if (num_nodes == 0) {
            throw FLANNException("Cannot upload empty tree to GPU");
        }

        // Store for later use (kernel needs this)
        gpu_num_nodes_ = num_nodes;

        // CRITICAL FIX: Pad to multiple of 4 to match OpenCL exactly
        // OpenCL uses: n_veclen = 4*((veclen+3)/4)
        // Must use identical formula for comparison
        int padded_veclen = 4 * ((this->veclen_ + 3) / 4);

        // Allocate host memory for flattening
        // Use temporary struct to collect node info before building interleaved array
        struct NodeInfo {
            int pivot_index;
            int child_start;  // -1 for leaves (will be replaced with leaf_data_ptr)
            int child_count;  // For leaves
        };
        std::vector<NodeInfo> nodes_info(num_nodes);

        // NOTE: pivots NOT stored separately - looked up from dataset via pivot_index
        std::vector<ElementType> dataset_host(this->size_ * padded_veclen, 0);
        std::vector<int> dataset_indices;  // Flattened leaf indices

        // Copy dataset with padding
        for (size_t i = 0; i < this->size_; ++i) {
            std::memcpy(&dataset_host[i * padded_veclen],
                       this->points_[i],
                       this->veclen_ * sizeof(ElementType));
            // Padding bytes remain zero from initialization
        }

        // Flatten trees breadth-first with IMPLICIT root storage (matching OpenCL)
        // CRITICAL FIX: Match OpenCL's hybrid indexing scheme EXACTLY
        // OpenCL pre-reserves indices 0-127 for first-level children (trees * branching)
        // then stores deeper nodes starting at index 128

        std::queue<NodePtr> node_queue;
        std::queue<int> node_id_queue;  // Track which slot each node goes to

        int num_trees = this->tree_roots_.size();
        int first_level_slots = num_trees * this->branching_;  // 4 * 32 = 128
        int next_available_slot = first_level_slots;  // Start at 128 for deeper nodes

        // Step 1: Process roots and assign their children to slots 0-127
        for (int tree_id = 0; tree_id < num_trees; ++tree_id) {
            NodePtr root = this->tree_roots_[tree_id];

            // Sanity check
            if (root->childs.empty()) {
                throw FLANNException("Tree " + std::to_string(tree_id) + " root is leaf");
            }

            // Assign this tree's children to their reserved slots
            int child_slot_start = tree_id * this->branching_;  // 0, 32, 64, 96

            for (size_t i = 0; i < root->childs.size(); ++i) {
                int child_slot = child_slot_start + i;
                node_queue.push(root->childs[i]);
                node_id_queue.push(child_slot);  // Child goes to its assigned slot
            }
        }

        // Step 2: Process all nodes in BFS order
        while (!node_queue.empty()) {
            NodePtr node = node_queue.front();
            node_queue.pop();

            int node_slot = node_id_queue.front();
            node_id_queue.pop();

            NodeInfo& info = nodes_info[node_slot];

            if (node->childs.empty()) {
                // Leaf node
                int leaf_offset = dataset_indices.size();
                info.child_start = -1;  // Mark as leaf, will set actual ptr later
                info.pivot_index = (node->pivot_index != SIZE_MAX)
                                 ? static_cast<int>(node->pivot_index) : -1;

                for (size_t i = 0; i < node->points.size(); ++i) {
                    size_t index = node->points[i].index;
                    if (!this->removed_ || !this->removed_points_.test(index)) {
                        dataset_indices.push_back(static_cast<int>(index));
                    }
                }

                info.child_count = dataset_indices.size() - leaf_offset;
            } else {
                // Parent node
                info.pivot_index = (node->pivot_index != SIZE_MAX) ?
                                  static_cast<int>(node->pivot_index) : -1;
                info.child_start = next_available_slot;  // Children start here
                info.child_count = node->childs.size();

                // Enqueue children at next available slots
                for (size_t i = 0; i < node->childs.size(); ++i) {
                    node_queue.push(node->childs[i]);
                    node_id_queue.push(next_available_slot++);
                }
            }
        }

        // ========================================================================
        // Interleaved Hybrid Array Construction
        // ========================================================================
        // Build interleaved array: [pivot_0, child_ptr_0, pivot_1, child_ptr_1, ..., leaf_data...]
        // This provides cache locality: pivot and child_ptr for same node are adjacent

        // Calculate hybrid array size (interleaved section + leaf data)
        int num_leaf_indices = dataset_indices.size();
        int interleaved_section_size = num_nodes * 2;  // 2 entries per node (pivot + child_ptr)
        int hybrid_size = interleaved_section_size + num_leaf_indices + num_leaves;  // + counts

        std::vector<int> hybrid_node_index(hybrid_size);
        int next_data_ptr = interleaved_section_size;  // Leaf data starts after interleaved section

        // Build interleaved array
        int dataset_idx_offset = 0;  // Track position in dataset_indices

        for (int i = 0; i < num_nodes; ++i) {
            const NodeInfo& info = nodes_info[i];

            // Always store pivot at even index
            hybrid_node_index[i * 2] = info.pivot_index;

            if (info.child_start < 0) {
                // LEAF: Store pointer to leaf data region at odd index
                hybrid_node_index[i * 2 + 1] = next_data_ptr;

                // Write leaf data: [count, idx1, idx2, ...]
                int leaf_count = info.child_count;
                hybrid_node_index[next_data_ptr++] = leaf_count;  // Count first

                // Copy leaf indices from dataset_indices
                for (int j = 0; j < leaf_count; ++j) {
                    hybrid_node_index[next_data_ptr++] = dataset_indices[dataset_idx_offset++];
                }
            } else {
                // PARENT: Store child_start pointer at odd index
                hybrid_node_index[i * 2 + 1] = info.child_start;
            }
        }

        // ========================================================================
        // Store padded bytes for later use (kernel/save needs this)
        // ========================================================================
        gpu_padded_bytes_ = padded_veclen;

        // ========================================================================
        // Calculate workspace layout with 16-byte alignment
        // Layout: [dataset | node_index]
        // ========================================================================

        // Check for overflow in dataset size calculation
        if (this->size_ > SIZE_MAX / padded_veclen) {
            throw FLANNException("Dataset too large - size * padded_veclen would overflow");
        }
        size_t dataset_size = this->size_ * padded_veclen;

        // Check for overflow in workspace calculations
        if (dataset_size > SIZE_MAX / sizeof(ElementType)) {
            throw FLANNException("Dataset too large - dataset bytes would overflow");
        }
        size_t dataset_bytes = dataset_size * sizeof(ElementType);

        size_t offset_dataset = 0;
        workspace_offset_node_index_ = alignTo16(offset_dataset + dataset_bytes);

        // Check for final workspace size overflow
        if (hybrid_size > SIZE_MAX / sizeof(int)) {
            throw FLANNException("Node index too large - would overflow");
        }
        size_t node_index_bytes = hybrid_size * sizeof(int);
        if (workspace_offset_node_index_ > SIZE_MAX - node_index_bytes) {
            throw FLANNException("Total workspace size would overflow");
        }
        size_t total_workspace_size = workspace_offset_node_index_ + node_index_bytes;

        // ========================================================================
        // Pack all arrays into single PINNED host buffer for fast DMA transfer
        // ========================================================================
        PinnedBuffer<unsigned char> workspace_host(total_workspace_size);

        std::memcpy(workspace_host.get() + offset_dataset,
                   dataset_host.data(),
                   dataset_bytes);
        std::memcpy(workspace_host.get() + workspace_offset_node_index_,
                   hybrid_node_index.data(),
                   node_index_bytes);

        // ========================================================================
        // SINGLE GPU upload (1 cudaMalloc + 1 cudaMemcpy) - THE KEY OPTIMIZATION
        // ========================================================================
        workspace_gpu_.resize(total_workspace_size);
        workspace_gpu_.upload(workspace_host.get(), total_workspace_size);

        // ========================================================================
        // Set up raw pointers into workspace
        // ========================================================================
        unsigned char* base_ptr = workspace_gpu_.get();
        dataset_ptr_ = reinterpret_cast<ElementType*>(base_ptr + offset_dataset);
        node_index_ptr_ = reinterpret_cast<int*>(base_ptr + workspace_offset_node_index_);

        gpu_initialized_ = true;
    }

    /**
     * @brief Count nodes in a tree (recursive helper)
     *
     * @param node Current node being traversed
     * @param num_nodes Total node count
     * @param num_parents Parent node count
     * @param num_leaves Leaf node count
     */
    void countNodes(NodePtr node, int* num_nodes, int* num_parents, int* num_leaves) const
    {
        (*num_nodes)++;

        if (node->childs.empty()) {
            (*num_leaves)++;
        } else {
            (*num_parents)++;
            for (size_t i = 0; i < node->childs.size(); ++i) {
                countNodes(node->childs[i], num_nodes, num_parents, num_leaves);
            }
        }
    }

    // ========================================================================
    // GPU Search Implementation
    // ========================================================================

    /**
     * @brief Perform k-NN search on GPU (implementation)
     *
     * Launches CUDA kernel for hierarchical search with:
     * - Thread-per-query design
     * - Best-first tree traversal with Hamming distance
     * - Priority queue for unexplored nodes
     * - Max-heap for k-NN results
     *
     * @param queries Query descriptors
     * @param indices Output indices
     * @param dists Output distances
     * @param knn Number of nearest neighbors
     * @param params Search parameters
     * @return Number of neighbors found
     */
    int knnSearchGPU(const Matrix<ElementType>& queries,
                     Matrix<size_t>& indices,
                     Matrix<DistanceType>& dists,
                     size_t knn,
                     const SearchParams& params) const
    {
        if (!gpu_initialized_) {
            throw FLANNException("Index not built or GPU data not uploaded");
        }

        // Validate K value is supported for GPU search
        if (!isKValueSupportedForGPU(static_cast<int>(knn))) {
            throw FLANNException(
                "Unsupported k=" + std::to_string(knn) + " for CUDA hierarchical search.\n"
                "Supported k values: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128\n"
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
        if (indices.rows < num_queries || indices.cols < knn) {
            throw FLANNException("Indices matrix too small");
        }
        if (dists.rows < num_queries || dists.cols < knn) {
            throw FLANNException("Distances matrix too small");
        }

        // Get search parameters
        int max_checks = params.checks;
        if (max_checks <= 0 || max_checks == FLANN_CHECKS_UNLIMITED) {
            max_checks = 256;  // Default
        }
        (void)max_checks;  // Currently unused, reserved for future optimization

        // CRITICAL FIX: Pad to multiple of 4 to match OpenCL exactly
        // OpenCL uses: n_veclen = 4*((veclen+3)/4)
        // Must use identical formula for comparison
        int padded_bytes = 4 * ((this->veclen_ + 3) / 4);

        // Allocate per-search GPU buffers (freed automatically when function returns)
        // Trade-off: ~5-8ms allocation overhead per search, but saves ~660MB persistent memory
        CUDABuffer<ElementType> gpu_queries(num_queries * padded_bytes);
        CUDABuffer<int> gpu_indices(num_queries * knn);
        CUDABuffer<int> gpu_dists(num_queries * knn);

        // Require contiguous queries (O(1) stride check instead of O(n) pointer loop)
        if (queries.stride != this->veclen_ * sizeof(ElementType)) {
            throw FLANNException(
                "GPU search requires contiguous query matrix (stride must equal cols * sizeof(T)). "
                "Allocate queries as: Matrix<T>(new T[rows * cols], rows, cols)");
        }

        // One-time warning if padding is needed (non-multiple-of-4 dimension)
        if ((size_t)padded_bytes != this->veclen_) {
            static bool warned = false;
            if (!warned) {
                fprintf(stderr, "[FLANN] Dimension %zu requires padding to %d. "
                    "Use multiples of 4 for best performance.\n",
                    this->veclen_, padded_bytes);
                warned = true;
            }
        }

        // Upload queries to GPU (using thread-local stream)
        if ((size_t)padded_bytes == this->veclen_) {
            // Fast path: no padding needed, direct upload
            gpu_queries.upload(queries[0], num_queries * this->veclen_, stream);
        } else {
            // Padding needed: upload raw then pad on GPU
            CUDABuffer<ElementType> raw_queries_gpu(num_queries * this->veclen_);
            raw_queries_gpu.upload(queries[0], num_queries * this->veclen_, stream);
            if (!launch_pad_queries<ElementType>(
                    raw_queries_gpu.get(),
                    gpu_queries.get(),
                    num_queries,
                    this->veclen_,
                    padded_bytes,
                    stream)) {
                throw FLANNException("Failed to launch GPU padding kernel");
            }
        }

        // Run cooperative kernel (on thread-local stream)
        bool success = launch_hierarchical_search_cooperative(
            reinterpret_cast<const unsigned char*>(dataset_ptr_),
            reinterpret_cast<const unsigned char*>(gpu_queries.get()),
            node_index_ptr_,  // Interleaved [pivot, child_ptr] + leaf data
            gpu_indices.get(),
            gpu_dists.get(),
            num_queries,
            padded_bytes,      // For array indexing (data stored with padding)
            this->veclen_,     // For Hamming distance (actual descriptor length)
            gpu_num_nodes_,    // Number of tree nodes
            knn,               // Number of nearest neighbors
            gpu_num_trees_,    // Number of trees (roots at indices 0..num_trees-1)
            this->branching_,  // Branching factor (tree N's children start at N*branching)
            stream             // CUDA stream for concurrent execution
        );

        if (!success) {
            throw FLANNException("Unsupported k value for GPU search");
        }

        // Check for kernel launch errors (non-blocking)
        CUDA_CHECK_LAST();

        // Download results using pinned memory for faster DMA transfers
        // Use async transfers on thread-local stream to overlap both downloads
        PinnedBuffer<int> result_indices(num_queries * knn);
        PinnedBuffer<int> result_dists(num_queries * knn);

        // Async downloads (will wait for kernel on same stream)
        gpu_indices.download(result_indices.get(), num_queries * knn, stream);
        gpu_dists.download(result_dists.get(), num_queries * knn, stream);

        // Single sync point - waits for kernel + both downloads on this thread's stream
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK_LAST();  // Check for any errors after sync

        // Copy to output matrices (convert int -> size_t for indices, int -> DistanceType for dists)
        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < knn; ++j) {
                indices[i][j] = static_cast<size_t>(result_indices[i * knn + j]);
                dists[i][j] = static_cast<DistanceType>(result_dists[i * knn + j]);
            }
        }

        return num_queries;
        // gpu_queries, gpu_indices, gpu_dists freed automatically here
    }

private:
    // ========================================================================
    // GPU State
    // ========================================================================

    bool gpu_initialized_;      ///< True if GPU buffers allocated and uploaded
    bool gpu_search_ready_;     ///< True if buildCUDAKnnSearch() was called
    bool gpu_only_mode_;        ///< True if loaded with gpu_only=true (points_ not available)

    // Note: Thread safety is now handled by rw_lock_ in base class CUDAIndex
    // (std::shared_mutex for reader-writer locking)

    int gpu_num_trees_;         ///< Number of trees (tree roots are nodes 0..num_trees-1)
    int gpu_num_nodes_;         ///< Number of tree nodes (needed for kernel)
    mutable size_t gpu_padded_bytes_;  ///< Padded descriptor size (for aligned access)

    // GPU workspace buffer - single allocation for all GPU arrays
    // Contains: [dataset | node_index]
    // Each section is 16-byte aligned for optimal CUDA memory access
    mutable CUDABuffer<unsigned char> workspace_gpu_;

    // Workspace layout offset (bytes from workspace start)
    mutable size_t workspace_offset_node_index_;

    // Non-owning pointers into workspace (nullptr if workspace not allocated)
    mutable ElementType* dataset_ptr_;
    mutable int* node_index_ptr_;

};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_HIERARCHICAL_CUDA_INDEX_H_
