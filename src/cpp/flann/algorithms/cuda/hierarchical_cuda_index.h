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
#include <map>  // For std::map
#include <utility>  // For std::swap
#include <cstring>  // For strlen, memcpy
#include <memory>   // For std::unique_ptr
#include <unistd.h>  // For write
#include <cuda_runtime.h>
#ifndef NDEBUG
#include <atomic>
#endif

#include "flann/algorithms/hierarchical_clustering_index.h"
#include "flann/algorithms/cuda/cuda_utils.h"
#include "flann/algorithms/cuda/nn_cuda_index.h"
// Note: KMeansNodeGPU no longer needed - pivot_index stored in interleaved node_index
#include "flann/util/gpu_saving.h"

namespace flann {
namespace cuda {

// Only include kernel headers when compiling with nvcc
#ifdef __CUDACC__
#include "flann/algorithms/cuda/kernels/hierarchical_search_kernel.cuh"
#include "flann/algorithms/cuda/kernels/hierarchical_search_cooperative.cuh"
#else
// Forward declare kernel launch functions for non-CUDA compilation
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
    int branching);
#endif

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
 * THREAD SAFETY: This index is NOT thread-safe for concurrent searches.
 * A single index instance should not be used from multiple threads
 * simultaneously. Each thread should have its own index instance,
 * or external synchronization must be used.
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
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_dataset_(),
          gpu_node_index_()
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
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_dataset_(),
          gpu_node_index_()
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
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_dataset_(),
          gpu_node_index_()
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
          gpu_num_trees_(0),
          gpu_num_nodes_(0),
          gpu_dataset_(),
          gpu_node_index_()
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
        // Allow case where GPU is already initialized from loading GPU format file
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
            uploadToGPU();
        }

        // Warmup kernel launch to trigger JIT compilation and CUDA memory pool initialization
        // This moves ~340-450ms one-time cost from first search to buildCUDAKnnSearch()
        // Uses local allocation (freed after warmup) - no persistent buffers
        {
            int padded_bytes = 4 * ((this->veclen_ + 3) / 4);

            // Allocate minimal warmup buffers (1 query)
            CUDABuffer<ElementType> warmup_queries(padded_bytes);
            CUDABuffer<int> warmup_indices(knn);
            CUDABuffer<int> warmup_dists(knn);

            // Upload dummy query (zeros)
            std::vector<ElementType> dummy_query(padded_bytes, 0);
            warmup_queries.upload(dummy_query.data(), padded_bytes);

            // Launch kernel with 1 query to trigger JIT compilation
            bool success = launch_hierarchical_search_cooperative(
                reinterpret_cast<const unsigned char*>(gpu_dataset_.get()),
                reinterpret_cast<const unsigned char*>(warmup_queries.get()),
                gpu_node_index_.get(),  // Interleaved [pivot, child_ptr] array
                warmup_indices.get(),
                warmup_dists.get(),
                1,                     // 1 query for warmup
                padded_bytes,
                this->veclen_,
                gpu_num_nodes_,
                knn,
                gpu_num_trees_,
                this->branching_
            );
            if (!success) {
                throw FLANNException(
                    "Unsupported k=" + std::to_string(knn) + " for CUDA hierarchical search.\n"
                    "Supported k values: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128\n"
                    "Use a supported k value or fall back to CPU search.");
            }
            // Wait for JIT compilation to complete
            cudaDeviceSynchronize();
            // warmup buffers freed automatically when scope exits
        }

        gpu_search_ready_ = true;
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
     * @brief Save index to file (GPU format if available, CPU format otherwise)
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

        serialization::SaveArchive sa(stream);

        // Get padded size for calculation
        int padded_bytes = 4 * ((this->veclen_ + 3) / 4);

        // Build GPU header with all metadata
        GPUIndexHeader header;
        header.h.data_type = flann_datatype_value<ElementType>::value;
        header.h.index_type = FLANN_INDEX_HIERARCHICAL_GPU_SAVED;
        header.h.rows = this->size_;
        header.h.cols = this->veclen_;
        header.h.padded_veclen = padded_bytes;
        header.h.num_nodes = gpu_num_nodes_;
        header.h.node_index_size = gpu_node_index_.count();
        header.h.leaf_count = 0;  // Not used for hierarchical
        header.h.branching = this->branching_;
        header.h.trees = gpu_num_trees_;
        header.h.leaf_max_size = this->leaf_max_size_;
        header.h.centers_init = this->centers_init_;

        sa & header;

        // Download and save GPU arrays
        // NOTE: No nodes array - pivot_index is stored interleaved in node_index
        // NOTE: pivots NOT saved - they're looked up from dataset via pivot_index
        std::vector<int> node_index_host(header.h.node_index_size);
        std::vector<ElementType> dataset_host(this->size_ * padded_bytes);

        gpu_node_index_.download(node_index_host.data(), header.h.node_index_size);
        gpu_dataset_.download(dataset_host.data(), this->size_ * padded_bytes);

        // Serialize arrays
        // node_index contains interleaved [pivot, child_ptr] pairs + leaf data
        sa & serialization::make_binary_object(
            node_index_host.data(),
            header.h.node_index_size * sizeof(int));
        sa & serialization::make_binary_object(
            dataset_host.data(),
            this->size_ * padded_bytes * sizeof(ElementType));

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
     * @brief Load index from file (auto-detects GPU vs CPU format)
     *
     * @param stream Input file stream
     */
    void loadIndex(FILE* stream) override
    {
        if (GPUIndexHeader::detectGPUFormat(stream)) {
            loadIndexGPU(stream);
        } else {
            // Fall back to CPU format
            freeGPUMemory();
            BaseClass::loadIndex(stream);
        }
    }

    /**
     * @brief Invalidate GPU data structures after adding points
     *
     * Adding points changes the tree structure, so GPU buffers must be rebuilt.
     */
    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2) override
    {
        freeGPUMemory();
        BaseClass::addPoints(points, rebuild_threshold);
    }

    /**
     * @brief Invalidate GPU data structures after removing a point
     */
    void removePoint(size_t id) override
    {
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
        std::swap(gpu_num_trees_, other.gpu_num_trees_);
        std::swap(gpu_num_nodes_, other.gpu_num_nodes_);

        // Swap GPU buffers using std::swap (works with move semantics)
        std::swap(gpu_dataset_, other.gpu_dataset_);
        std::swap(gpu_node_index_, other.gpu_node_index_);
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
        // Move-assign empty buffers to trigger RAII cleanup
        gpu_dataset_ = CUDABuffer<ElementType>();
        gpu_node_index_ = CUDABuffer<int>();

        gpu_initialized_ = false;
        gpu_search_ready_ = false;
    }

    /**
     * @brief Load GPU-optimized format from file
     *
     * @param stream Input file stream
     */
    void loadIndexGPU(FILE* stream)
    {
        freeGPUMemory();

        serialization::LoadArchive la(stream);

        GPUIndexHeader header;
        la & header;

        // Validate header
        if (header.h.data_type != flann_datatype_value<ElementType>::value) {
            throw FLANNException("Data type mismatch in GPU index file");
        }
        if (header.h.index_type != FLANN_INDEX_HIERARCHICAL_GPU_SAVED) {
            throw FLANNException("Index type mismatch: expected HIERARCHICAL_GPU_SAVED");
        }

        // Restore metadata
        this->size_ = header.h.rows;
        this->veclen_ = header.h.cols;
        this->size_at_build_ = header.h.rows;
        int padded_bytes = header.h.padded_veclen;
        gpu_num_nodes_ = header.h.num_nodes;
        gpu_num_trees_ = header.h.trees;
        this->branching_ = header.h.branching;
        this->leaf_max_size_ = header.h.leaf_max_size;
        this->centers_init_ = header.h.centers_init;

        // Allocate and load node_index from file (use unique_ptr to avoid zero-init)
        // NOTE: No nodes array - pivot_index is stored interleaved in node_index
        // NOTE: pivots NOT loaded - they're looked up from dataset via pivot_index
        std::unique_ptr<int[]> node_index_host(new int[header.h.node_index_size]);

        la & serialization::make_binary_object(
            node_index_host.get(),
            header.h.node_index_size * sizeof(int));

        // Upload node_index to GPU
        gpu_node_index_.resize(header.h.node_index_size);
        gpu_node_index_.upload(node_index_host.get(), header.h.node_index_size);

        // Reconstruct points_ and upload dataset to GPU
        // Use fast path when no padding is needed (veclen already aligned to 4)
        delete[] this->data_ptr_;
        this->data_ptr_ = new ElementType[this->size_ * this->veclen_];
        this->points_.resize(this->size_);

        if (static_cast<size_t>(padded_bytes) == this->veclen_) {
            // FAST PATH: No padding - read directly into data_ptr_, no temporary buffer
            la & serialization::make_binary_object(
                this->data_ptr_,
                this->size_ * this->veclen_ * sizeof(ElementType));

            gpu_dataset_.resize(this->size_ * this->veclen_);
            gpu_dataset_.upload(this->data_ptr_, this->size_ * this->veclen_);

            // Set up points_ pointers (no copy needed)
            for (size_t i = 0; i < this->size_; ++i) {
                this->points_[i] = this->data_ptr_ + i * this->veclen_;
            }
        } else {
            // PADDED PATH: Use temporary buffer with memcpy per row
            std::unique_ptr<ElementType[]> dataset_host(
                new ElementType[this->size_ * padded_bytes]);

            la & serialization::make_binary_object(
                dataset_host.get(),
                this->size_ * padded_bytes * sizeof(ElementType));

            gpu_dataset_.resize(this->size_ * padded_bytes);
            gpu_dataset_.upload(dataset_host.get(), this->size_ * padded_bytes);

            // Copy with memcpy per row (faster than element-by-element)
            for (size_t i = 0; i < this->size_; ++i) {
                this->points_[i] = this->data_ptr_ + i * this->veclen_;
                std::memcpy(this->points_[i],
                           &dataset_host[i * padded_bytes],
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
        // Note: tree_roots_ NOT reconstructed - GPU format is GPU-only
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

        // Allocate and upload GPU buffers
        // NOTE: No gpu_nodes_ - pivot_index is now in interleaved node_index
        gpu_dataset_.resize(this->size_ * padded_veclen);
        gpu_node_index_.resize(hybrid_size);  // Interleaved [pivot, child_ptr] + leaf data

        gpu_dataset_.upload(dataset_host.data(), this->size_ * padded_veclen);
        gpu_node_index_.upload(hybrid_node_index.data(), hybrid_size);

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

#ifndef NDEBUG
        // Thread safety check (debug builds only)
        bool expected = false;
        if (!search_in_progress_.compare_exchange_strong(expected, true)) {
            throw FLANNException("Concurrent search detected - HierarchicalCUDAIndex is NOT thread-safe. "
                "Each thread should have its own index instance.");
        }
        // RAII guard to reset flag on exit
        struct SearchGuard {
            std::atomic<bool>& flag;
            ~SearchGuard() { flag.store(false, std::memory_order_release); }
        } guard{search_in_progress_};
#endif

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

        // CRITICAL FIX: Pad to multiple of 4 to match OpenCL exactly
        // OpenCL uses: n_veclen = 4*((veclen+3)/4)
        // Must use identical formula for comparison
        int padded_bytes = 4 * ((this->veclen_ + 3) / 4);

        // Allocate per-search GPU buffers (freed automatically when function returns)
        // Trade-off: ~5-8ms allocation overhead per search, but saves ~660MB persistent memory
        CUDABuffer<ElementType> gpu_queries(num_queries * padded_bytes);
        CUDABuffer<int> gpu_indices(num_queries * knn);
        CUDABuffer<int> gpu_dists(num_queries * knn);

        // Prepare padded queries on host and upload to GPU
        std::vector<ElementType> padded_queries(num_queries * padded_bytes, 0);
        for (size_t i = 0; i < num_queries; ++i) {
            std::memcpy(&padded_queries[i * padded_bytes],
                       queries[i],
                       this->veclen_ * sizeof(ElementType));
            // Padding bytes already zero-initialized
        }
        gpu_queries.upload(padded_queries.data(), num_queries * padded_bytes);

        // Run cooperative kernel
        bool success = launch_hierarchical_search_cooperative(
            reinterpret_cast<const unsigned char*>(gpu_dataset_.get()),
            reinterpret_cast<const unsigned char*>(gpu_queries.get()),
            gpu_node_index_.get(),  // Interleaved [pivot, child_ptr] + leaf data
            gpu_indices.get(),
            gpu_dists.get(),
            num_queries,
            padded_bytes,      // For array indexing (data stored with padding)
            this->veclen_,     // For Hamming distance (actual descriptor length)
            gpu_num_nodes_,    // Number of tree nodes
            knn,               // Number of nearest neighbors
            gpu_num_trees_,    // Number of trees (roots at indices 0..num_trees-1)
            this->branching_   // Branching factor (tree N's children start at N*branching)
        );

        if (!success) {
            throw FLANNException("Unsupported k value for GPU search");
        }

        // Check for kernel errors and wait for completion
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download results from GPU
        std::vector<int> result_indices(num_queries * knn);
        std::vector<int> result_dists(num_queries * knn);
        gpu_indices.download(result_indices.data(), num_queries * knn);
        gpu_dists.download(result_dists.data(), num_queries * knn);

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

#ifndef NDEBUG
    // Thread safety detection (debug builds only)
    mutable std::atomic<bool> search_in_progress_{false};
#endif
    int gpu_num_trees_;         ///< Number of trees (tree roots are nodes 0..num_trees-1)
    int gpu_num_nodes_;         ///< Number of tree nodes (needed for kernel)

    // GPU buffers (RAII wrappers for automatic cleanup)
    // NOTE: No gpu_pivots_ - pivots accessed via pivot_index in dataset (eliminates duplication)
    // NOTE: No gpu_nodes_ - pivot_index and child_ptr are stored interleaved in gpu_node_index_
    mutable CUDABuffer<ElementType> gpu_dataset_;      ///< Dataset descriptors (padded)
    mutable CUDABuffer<int> gpu_node_index_;           ///< Interleaved [pivot, child_ptr] pairs + leaf data

};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_HIERARCHICAL_CUDA_INDEX_H_
