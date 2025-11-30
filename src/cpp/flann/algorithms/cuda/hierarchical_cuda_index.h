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
#include <cstring>  // For strlen
#include <unistd.h>  // For write
#include <cuda_runtime.h>

#include "flann/algorithms/hierarchical_clustering_index.h"
#include "flann/algorithms/cuda/cuda_utils.h"
#include "flann/algorithms/cuda/nn_cuda_index.h"
#include "flann/algorithms/cuda/kmeans_node_gpu.h"

namespace flann {
namespace cuda {

// Only include kernel headers when compiling with nvcc
#ifdef __CUDACC__
#include "flann/algorithms/cuda/kernels/hierarchical_search_kernel.cuh"
#include "flann/algorithms/cuda/kernels/hierarchical_search_cooperative.cuh"
#else
// Forward declare kernel launch functions for non-CUDA compilation
bool launch_hierarchical_search(
    const unsigned char* dataset,
    const unsigned char* queries,
    const KMeansNodeGPU* tree_nodes,
    const unsigned char* tree_pivots,
    const int* dataset_indices,
    const int* device_node_index,  // CRITICAL FIX: nodeIndex indirection array
    int* result_indices,
    int* result_distances,
    size_t num_queries,
    size_t padded_bytes,
    size_t actual_bytes,
    size_t num_nodes,
    int knn,
    int max_checks,
    int num_trees,
    int branching,
    dim3 grid,
    dim3 block);

bool launch_hierarchical_search_cooperative(
    const unsigned char* dataset,
    const unsigned char* queries,
    const KMeansNodeGPU* tree_nodes,
    const unsigned char* tree_pivots,
    const int* device_node_index,
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
          gpu_nodes_(),
          gpu_pivots_(),
          gpu_dataset_(),
          gpu_dataset_indices_(),
          gpu_node_index_()          // CRITICAL FIX: Initialize nodeIndex buffer
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
          gpu_nodes_(),
          gpu_pivots_(),
          gpu_dataset_(),
          gpu_dataset_indices_(),
          gpu_node_index_()          // CRITICAL FIX: Initialize nodeIndex buffer
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
          gpu_nodes_(),              // Explicit default construction
          gpu_pivots_(),             // Prevents copy attempt (copy ctor deleted)
          gpu_dataset_(),            // CUDABuffer starts empty (ptr_ = nullptr)
          gpu_dataset_indices_(),    // User must call buildCUDAKnnSearch()
          gpu_node_index_()          // CRITICAL FIX: Initialize nodeIndex buffer
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
          gpu_nodes_(),              // Explicit default construction
          gpu_pivots_(),             // Prevents copy attempt (copy ctor deleted)
          gpu_dataset_(),            // CUDABuffer starts empty (ptr_ = nullptr)
          gpu_dataset_indices_(),    // User must call buildCUDAKnnSearch()
          gpu_node_index_()          // CRITICAL FIX: Initialize nodeIndex buffer
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
        if (this->tree_roots_.empty()) {
            throw FLANNException("Cannot prepare GPU search: index not built yet. "
                               "Call buildIndex() first.");
        }

        // Check that trees are large enough (not degenerate)
        for (size_t i = 0; i < this->tree_roots_.size(); ++i) {
            if (this->tree_roots_[i]->childs.empty()) {
                throw FLANNException("Tree " + std::to_string(i) + " is degenerate "
                                   "(root is a leaf). Try reducing branching factor or "
                                   "increasing dataset size.");
            }
        }

        if (!gpu_initialized_) {
            uploadToGPU();
        }

        // Pre-allocate persistent query/result buffers to eliminate per-search cudaMalloc overhead
        // Use 20K as default max batch size (21MB GPU memory, trivial vs typical GPU)
        const size_t DEFAULT_MAX_QUERIES = 20000;
        const size_t DEFAULT_MAX_KNN = 128;

        gpu_buffer_max_queries_ = DEFAULT_MAX_QUERIES;
        gpu_buffer_max_knn_ = std::max(static_cast<size_t>(knn), DEFAULT_MAX_KNN);

        // Pad to multiple of 4 to match OpenCL exactly
        int padded_bytes = 4 * ((this->veclen_ + 3) / 4);

        // Pre-allocate GPU buffers
        gpu_query_buffer_.resize(gpu_buffer_max_queries_ * padded_bytes);
        gpu_indices_buffer_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);
        gpu_dists_buffer_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);

        // Pre-allocate pinned host staging buffers (2-3x faster DMA transfers)
        pinned_query_staging_.resize(gpu_buffer_max_queries_ * padded_bytes);
        pinned_indices_staging_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);
        pinned_dists_staging_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);

        // Warmup kernel launch to trigger JIT compilation during setup
        // This moves ~37ms one-time cost from first search to buildCUDAKnnSearch()
        {
            // Create dummy query (zeros)
            std::vector<ElementType> dummy_query(padded_bytes, 0);
            gpu_query_buffer_.upload(dummy_query.data(), padded_bytes);

            // Launch kernel with 1 query to trigger JIT compilation
            int num_nodes = gpu_nodes_.count();
            bool success = launch_hierarchical_search_cooperative(
                reinterpret_cast<const unsigned char*>(gpu_dataset_.get()),
                reinterpret_cast<const unsigned char*>(gpu_query_buffer_.get()),
                gpu_nodes_.get(),
                reinterpret_cast<const unsigned char*>(gpu_pivots_.get()),
                gpu_node_index_.get(),
                gpu_indices_buffer_.get(),
                gpu_dists_buffer_.get(),
                1,                     // 1 query for warmup
                padded_bytes,
                this->veclen_,
                num_nodes,
                knn,
                gpu_num_trees_,
                this->branching_
            );
            if (success) {
                // Wait for JIT compilation to complete
                cudaDeviceSynchronize();
            }
        }

        gpu_search_ready_ = true;
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

        // Swap GPU buffers using std::swap (works with move semantics)
        std::swap(gpu_nodes_, other.gpu_nodes_);
        std::swap(gpu_pivots_, other.gpu_pivots_);
        std::swap(gpu_dataset_, other.gpu_dataset_);
        std::swap(gpu_dataset_indices_, other.gpu_dataset_indices_);
        std::swap(gpu_node_index_, other.gpu_node_index_);

        // Swap persistent query/result buffers
        std::swap(gpu_query_buffer_, other.gpu_query_buffer_);
        std::swap(gpu_indices_buffer_, other.gpu_indices_buffer_);
        std::swap(gpu_dists_buffer_, other.gpu_dists_buffer_);
        std::swap(gpu_buffer_max_queries_, other.gpu_buffer_max_queries_);
        std::swap(gpu_buffer_max_knn_, other.gpu_buffer_max_knn_);

        // Swap pinned host staging buffers
        std::swap(pinned_query_staging_, other.pinned_query_staging_);
        std::swap(pinned_indices_staging_, other.pinned_indices_staging_);
        std::swap(pinned_dists_staging_, other.pinned_dists_staging_);
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
            // Convert to matrix format for GPU search
            assert(indices.size() >= queries.rows);
            assert(dists.size() >= queries.rows);

            flann::Matrix<size_t> indices_mat(new size_t[queries.rows * knn], queries.rows, knn);
            flann::Matrix<DistanceType> dists_mat(new DistanceType[queries.rows * knn], queries.rows, knn);

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

            delete[] indices_mat.ptr();
            delete[] dists_mat.ptr();

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
        gpu_nodes_ = CUDABuffer<KMeansNodeGPU>();
        gpu_pivots_ = CUDABuffer<ElementType>();
        gpu_dataset_ = CUDABuffer<ElementType>();
        gpu_dataset_indices_ = CUDABuffer<int>();
        gpu_node_index_ = CUDABuffer<int>();

        // Free persistent query/result buffers
        gpu_query_buffer_ = CUDABuffer<ElementType>();
        gpu_indices_buffer_ = CUDABuffer<int>();
        gpu_dists_buffer_ = CUDABuffer<int>();
        gpu_buffer_max_queries_ = 0;
        gpu_buffer_max_knn_ = 0;

        // Free pinned host staging buffers
        pinned_query_staging_.resize(0);
        pinned_indices_staging_.resize(0);
        pinned_dists_staging_.resize(0);

        gpu_initialized_ = false;
        gpu_search_ready_ = false;
    }

    // ========================================================================
    // Tree Upload to GPU
    // ========================================================================

    /**
     * @brief Upload tree structure to GPU memory
     *
     * **Flattening strategy (breadth-first):**
     * 1. Count nodes in all trees
     * 2. Allocate flat arrays for nodes, pivots, dataset
     * 3. Traverse trees breadth-first, copying data
     * 4. Upload to GPU with cudaMemcpy
     *
     * **Memory layout:**
     * - Nodes: KMeansNodeGPU structs (pivot_index, child_start, child_count, etc.)
     * - Pivots: Packed binary descriptors (padded to multiples of 16 bytes)
     * - Dataset: Packed binary descriptors (padded to multiples of 16 bytes)
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

        // CRITICAL FIX: Pad to multiple of 4 to match OpenCL exactly
        // OpenCL uses: n_veclen = 4*((veclen+3)/4)
        // Must use identical formula for comparison
        int padded_veclen = 4 * ((this->veclen_ + 3) / 4);

        // Allocate host memory for flattening
        std::vector<KMeansNodeGPU> nodes_host(num_nodes);
        std::vector<ElementType> pivots_host(num_nodes * padded_veclen, 0);  // Zero-initialize padding
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

            KMeansNodeGPU& gpu_node = nodes_host[node_slot];

            // CRITICAL FIX: Copy pivot for ALL nodes (leaves AND parents)
            // OpenCL does this at hierarchical_opencl_index.h:721-727
            // The search kernel needs pivots for all children when exploring branches
            if (node->pivot != nullptr) {
                std::memcpy(&pivots_host[node_slot * padded_veclen],
                           node->pivot,
                           this->veclen_ * sizeof(ElementType));
            }

            if (node->childs.empty()) {
                // Leaf node
                int leaf_offset = dataset_indices.size();
                gpu_node.child_start = -(leaf_offset + 1);
                gpu_node.pivot_index = -1;
                gpu_node.radius = 0.0f;
                gpu_node.variance = 0.0f;

                for (size_t i = 0; i < node->points.size(); ++i) {
                    size_t index = node->points[i].index;
                    if (!this->removed_ || !this->removed_points_.test(index)) {
                        dataset_indices.push_back(static_cast<int>(index));
                    }
                }

                gpu_node.child_count = static_cast<uint16_t>(
                    dataset_indices.size() - leaf_offset
                );
            } else {
                // Parent node
                gpu_node.pivot_index = (node->pivot_index != SIZE_MAX) ?
                                      static_cast<int>(node->pivot_index) : -1;
                gpu_node.child_start = next_available_slot;  // Children start here
                gpu_node.child_count = static_cast<uint16_t>(node->childs.size());
                gpu_node.radius = 0.0f;
                gpu_node.variance = 0.0f;

                // Enqueue children at next available slots
                for (size_t i = 0; i < node->childs.size(); ++i) {
                    node_queue.push(node->childs[i]);
                    node_id_queue.push(next_available_slot++);
                }
            }
        }

        // ========================================================================
        // Hybrid Array Construction - Matching OpenCL Exactly
        // ========================================================================
        // Build hybrid flat array containing both pointers AND embedded leaf data
        // Structure: [node_pointers...][leaf_region_1: count, idx...][leaf_region_2...]

        // Calculate hybrid array size
        int num_leaf_indices = dataset_indices.size();
        int hybrid_size = num_nodes + num_leaf_indices + num_leaves;  // pointers + data + counts

        std::vector<int> hybrid_node_index(hybrid_size);
        int next_data_ptr = num_nodes;  // Data section starts after pointer section

        // Build hybrid array using already-constructed nodes_host and dataset_indices
        int dataset_idx_offset = 0;  // Track position in dataset_indices

        for (int i = 0; i < num_nodes; ++i) {
            const KMeansNodeGPU& node = nodes_host[i];

            if (node.child_start < 0) {
                // LEAF: Store pointer to leaf data region in hybrid array
                hybrid_node_index[i] = next_data_ptr;

                // Write leaf data: [count, idx1, idx2, ...]
                int leaf_count = node.child_count;
                hybrid_node_index[next_data_ptr++] = leaf_count;  // Count first

                // Copy leaf indices from dataset_indices
                for (int j = 0; j < leaf_count; ++j) {
                    hybrid_node_index[next_data_ptr++] = dataset_indices[dataset_idx_offset++];
                }
            } else {
                // PARENT: Store child_start pointer (same as CUDA tree_nodes)
                hybrid_node_index[i] = node.child_start;
            }
        }

        // Allocate and upload GPU buffers using .resize()
        gpu_nodes_.resize(num_nodes);
        gpu_pivots_.resize(num_nodes * padded_veclen);
        gpu_dataset_.resize(this->size_ * padded_veclen);
        gpu_dataset_indices_.resize(dataset_indices.size());
        gpu_node_index_.resize(hybrid_size);  // CRITICAL: Hybrid array with embedded leaf data

        gpu_nodes_.upload(nodes_host.data(), num_nodes);
        gpu_pivots_.upload(pivots_host.data(), num_nodes * padded_veclen);
        gpu_dataset_.upload(dataset_host.data(), this->size_ * padded_veclen);
        gpu_dataset_indices_.upload(dataset_indices.data(), dataset_indices.size());
        gpu_node_index_.upload(hybrid_node_index.data(), hybrid_size);  // CRITICAL: Upload hybrid array

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

        // Resize persistent buffers if needed (rare path - only if exceeding pre-allocated capacity)
        if (num_queries > gpu_buffer_max_queries_ || knn > gpu_buffer_max_knn_) {
            gpu_buffer_max_queries_ = std::max(num_queries, gpu_buffer_max_queries_);
            gpu_buffer_max_knn_ = std::max(knn, gpu_buffer_max_knn_);
            gpu_query_buffer_.resize(gpu_buffer_max_queries_ * padded_bytes);
            gpu_indices_buffer_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);
            gpu_dists_buffer_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);
            pinned_query_staging_.resize(gpu_buffer_max_queries_ * padded_bytes);
            pinned_indices_staging_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);
            pinned_dists_staging_.resize(gpu_buffer_max_queries_ * gpu_buffer_max_knn_);
        }

        // Pad queries into pinned host buffer (fast DMA transfer)
        ElementType* query_ptr = pinned_query_staging_.get();
        for (size_t i = 0; i < num_queries; ++i) {
            std::memcpy(query_ptr + i * padded_bytes,
                       queries[i],
                       this->veclen_ * sizeof(ElementType));
            // Clear padding bytes (needed since buffer is reused)
            std::memset(query_ptr + i * padded_bytes + this->veclen_,
                       0,
                       (padded_bytes - this->veclen_) * sizeof(ElementType));
        }

        // Upload from pinned memory (2-3x faster DMA transfer)
        gpu_query_buffer_.upload(query_ptr, num_queries * padded_bytes);

        // Run cooperative kernel using persistent GPU buffers
        int num_nodes = gpu_nodes_.count();

        bool success = launch_hierarchical_search_cooperative(
            reinterpret_cast<const unsigned char*>(gpu_dataset_.get()),
            reinterpret_cast<const unsigned char*>(gpu_query_buffer_.get()),  // Persistent buffer
            gpu_nodes_.get(),
            reinterpret_cast<const unsigned char*>(gpu_pivots_.get()),
            gpu_node_index_.get(),  // CRITICAL: Pass nodeIndex indirection array
            gpu_indices_buffer_.get(),  // Persistent buffer
            gpu_dists_buffer_.get(),    // Persistent buffer
            num_queries,
            padded_bytes,      // For array indexing (data stored with padding)
            this->veclen_,     // For Hamming distance (actual descriptor length)
            num_nodes,
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

        // Download results to pinned host buffers (2-3x faster DMA transfer)
        int* indices_ptr = pinned_indices_staging_.get();
        int* dists_ptr = pinned_dists_staging_.get();
        gpu_indices_buffer_.download(indices_ptr, num_queries * knn);
        gpu_dists_buffer_.download(dists_ptr, num_queries * knn);

        // Copy to output matrices (convert int -> size_t for indices, int -> DistanceType for dists)
        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < knn; ++j) {
                indices[i][j] = static_cast<size_t>(indices_ptr[i * knn + j]);
                dists[i][j] = static_cast<DistanceType>(dists_ptr[i * knn + j]);
            }
        }

        return num_queries;
    }

private:
    // ========================================================================
    // GPU State
    // ========================================================================

    bool gpu_initialized_;      ///< True if GPU buffers allocated and uploaded
    bool gpu_search_ready_;     ///< True if buildCUDAKnnSearch() was called
    int gpu_num_trees_;         ///< Number of trees (tree roots are nodes 0..num_trees-1)

    // GPU buffers (RAII wrappers for automatic cleanup)
    mutable CUDABuffer<KMeansNodeGPU> gpu_nodes_;      ///< Flattened node array
    mutable CUDABuffer<ElementType> gpu_pivots_;       ///< Pivot descriptors (padded)
    mutable CUDABuffer<ElementType> gpu_dataset_;      ///< Dataset descriptors (padded)
    mutable CUDABuffer<int> gpu_dataset_indices_;      ///< Leaf node dataset indices
    mutable CUDABuffer<int> gpu_node_index_;           ///< Indirection array matching OpenCL (pointers not indices)

    // Persistent query/result buffers for search (reused across searches to eliminate cudaMalloc overhead)
    mutable CUDABuffer<ElementType> gpu_query_buffer_;  ///< Reusable query upload buffer
    mutable CUDABuffer<int> gpu_indices_buffer_;        ///< Reusable results buffer
    mutable CUDABuffer<int> gpu_dists_buffer_;          ///< Reusable distances buffer
    mutable size_t gpu_buffer_max_queries_ = 0;         ///< Current query buffer capacity
    mutable size_t gpu_buffer_max_knn_ = 0;             ///< Current k capacity

    // Pinned host-side staging buffers for fast DMA transfers (2-3x faster than pageable)
    mutable PinnedBuffer<ElementType> pinned_query_staging_;  ///< Pinned query staging buffer
    mutable PinnedBuffer<int> pinned_indices_staging_;        ///< Pinned results staging buffer
    mutable PinnedBuffer<int> pinned_dists_staging_;          ///< Pinned distances staging buffer
};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_HIERARCHICAL_CUDA_INDEX_H_
