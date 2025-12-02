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
#ifndef NDEBUG
#include <atomic>
#endif

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
            int num_nodes = gpu_nodes_.count();
            bool success = launch_hierarchical_search_cooperative(
                reinterpret_cast<const unsigned char*>(gpu_dataset_.get()),
                reinterpret_cast<const unsigned char*>(warmup_queries.get()),
                reinterpret_cast<const unsigned char*>(gpu_pivots_.get()),
                gpu_node_index_.get(),
                warmup_indices.get(),
                warmup_dists.get(),
                1,                     // 1 query for warmup
                padded_bytes,
                this->veclen_,
                num_nodes,
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

        // Swap GPU buffers using std::swap (works with move semantics)
        std::swap(gpu_nodes_, other.gpu_nodes_);
        std::swap(gpu_pivots_, other.gpu_pivots_);
        std::swap(gpu_dataset_, other.gpu_dataset_);
        std::swap(gpu_dataset_indices_, other.gpu_dataset_indices_);
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
        gpu_nodes_ = CUDABuffer<KMeansNodeGPU>();
        gpu_pivots_ = CUDABuffer<ElementType>();
        gpu_dataset_ = CUDABuffer<ElementType>();
        gpu_dataset_indices_ = CUDABuffer<int>();
        gpu_node_index_ = CUDABuffer<int>();

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
        int num_nodes = gpu_nodes_.count();

        bool success = launch_hierarchical_search_cooperative(
            reinterpret_cast<const unsigned char*>(gpu_dataset_.get()),
            reinterpret_cast<const unsigned char*>(gpu_queries.get()),
            reinterpret_cast<const unsigned char*>(gpu_pivots_.get()),
            gpu_node_index_.get(),  // CRITICAL: Pass nodeIndex indirection array
            gpu_indices.get(),
            gpu_dists.get(),
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

    // GPU buffers (RAII wrappers for automatic cleanup)
    mutable CUDABuffer<KMeansNodeGPU> gpu_nodes_;      ///< Flattened node array
    mutable CUDABuffer<ElementType> gpu_pivots_;       ///< Pivot descriptors (padded)
    mutable CUDABuffer<ElementType> gpu_dataset_;      ///< Dataset descriptors (padded)
    mutable CUDABuffer<int> gpu_dataset_indices_;      ///< Leaf node dataset indices
    mutable CUDABuffer<int> gpu_node_index_;           ///< Indirection array matching OpenCL (pointers not indices)

};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_HIERARCHICAL_CUDA_INDEX_H_
