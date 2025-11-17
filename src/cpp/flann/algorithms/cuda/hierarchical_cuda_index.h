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
#else
// Forward declare kernel launch function for non-CUDA compilation
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
        fprintf(stderr, "[DEBUG] HierarchicalCUDAIndex copy constructor ENTRY\n");
        // GPU buffers start empty - user must call buildCUDAKnnSearch()
        fprintf(stderr, "[DEBUG] HierarchicalCUDAIndex copy constructor EXIT\n");
    }

    /**
     * @brief Destructor
     */
    ~HierarchicalCUDAIndex()
    {
        fprintf(stderr, "[DEBUG] ~HierarchicalCUDAIndex() destructor called\n");
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
        const char* msg = "[DEBUG] buildCUDAKnnSearch() ENTRY\n";
        write(2, msg, strlen(msg));
        if (this->tree_roots_.empty()) {
            throw FLANNException("Cannot prepare GPU search: index not built yet. "
                               "Call buildIndex() first.");
        }
        msg = "[DEBUG] tree_roots_ not empty\n";
        write(2, msg, strlen(msg));

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
        printf("[DEBUG] uploadToGPU() ENTRY\n"); fflush(stdout);
        if (gpu_initialized_) {
            printf("[DEBUG] Already uploaded, returning\n"); fflush(stdout);
            return;  // Already uploaded
        }
        printf("[DEBUG] Starting upload process...\n"); fflush(stdout);

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
        printf("[DEBUG] After counting: num_nodes=%d, num_parents=%d, num_leaves=%d\n", num_nodes, num_parents, num_leaves); fflush(stdout);
        if (num_nodes == 0) {
            throw FLANNException("Cannot upload empty tree to GPU");
        }
        printf("[DEBUG] Sanity check passed\n"); fflush(stdout);

        // CRITICAL FIX: Pad to multiple of 4 to match OpenCL exactly
        // OpenCL uses: n_veclen = 4*((veclen+3)/4)
        // Must use identical formula for comparison
        int padded_veclen = 4 * ((this->veclen_ + 3) / 4);

        printf("[UPLOAD DEBUG] veclen_=%zu, padded_veclen=%d (multiple of 4, matching OpenCL), sizeof(ElementType)=%zu\n",
               this->veclen_, padded_veclen, sizeof(ElementType));

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
        // Roots are virtual - only children are stored
        std::queue<NodePtr> node_queue;
        int node_index = 0;

        // Initialize queue with ROOT CHILDREN directly (skip roots themselves)
        // Tree N's children will start at index N * branching
        int total_first_level_children = 0;
        for (size_t tree_id = 0; tree_id < this->tree_roots_.size(); ++tree_id) {
            NodePtr root = this->tree_roots_[tree_id];

            // Sanity check: hierarchical roots must have children
            if (root->childs.empty()) {
                throw FLANNException("Hierarchical tree " + std::to_string(tree_id) +
                                   " has degenerate root with no children");
            }

            // Enqueue all children of this root
            for (size_t i = 0; i < root->childs.size(); ++i) {
                node_queue.push(root->childs[i]);
                total_first_level_children++;
            }
        }

        // Track next available index for children
        // CRITICAL FIX: Start AFTER all first-level children we just enqueued!
        // Otherwise first-level nodes would point to themselves
        int next_child_node_idx = total_first_level_children;

        // Breadth-first traversal to flatten tree structure
        while (!node_queue.empty()) {
            NodePtr node = node_queue.front();
            node_queue.pop();

            KMeansNodeGPU& gpu_node = nodes_host[node_index];

            if (node->childs.empty()) {
                // Leaf node: encode offset into dataset_indices as -(offset + 1)
                int leaf_offset = dataset_indices.size();
                gpu_node.child_start = -(leaf_offset + 1);  // Negative marks leaf
                gpu_node.pivot_index = -1;  // Leaves don't have pivots
                gpu_node.radius = 0.0f;
                gpu_node.padding = 0;

                // Add all dataset indices for this leaf
                for (size_t i = 0; i < node->points.size(); ++i) {
                    size_t index = node->points[i].index;
                    // Skip removed points if using removed points tracking
                    if (!this->removed_ || !this->removed_points_.test(index)) {
                        dataset_indices.push_back(static_cast<int>(index));
                    }
                }

                // Store actual count of indices added (after filtering removed)
                gpu_node.child_count = static_cast<uint16_t>(
                    dataset_indices.size() - leaf_offset
                );
            } else {
                // Parent node: store pivot and children info
                gpu_node.pivot_index = (node->pivot_index != SIZE_MAX) ?
                                      static_cast<int>(node->pivot_index) : -1;

                // CRITICAL FIX: Use global counter instead of queue-based prediction
                // This ensures correct absolute positioning when all roots are enqueued first
                gpu_node.child_start = next_child_node_idx;
                gpu_node.child_count = static_cast<uint16_t>(node->childs.size());

                // Advance counter by number of children this node will add
                next_child_node_idx += node->childs.size();

                gpu_node.radius = 0.0f;  // Hierarchical doesn't use radius
                gpu_node.padding = 0;

                // Copy pivot descriptor with padding
                if (node->pivot != nullptr) {
                    std::memcpy(&pivots_host[node_index * padded_veclen],
                               node->pivot,
                               this->veclen_ * sizeof(ElementType));
                }

                // Enqueue children for processing
                for (size_t i = 0; i < node->childs.size(); ++i) {
                    node_queue.push(node->childs[i]);
                }
            }

            node_index++;
        }

        // ========================================================================
        // ========================================================================
        // PHASE 3: nodeIndex Construction - Comparing CUDA vs OpenCL Algorithms
        // ========================================================================

        printf("\n");
        printf("╔════════════════════════════════════════════════════════════════════╗\n");
        printf("║  PHASE 3: nodeIndex Algorithm Investigation                       ║\n");
        printf("╚════════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
        printf("[PHASE 3] Tree Structure:\n");
        printf("  Total nodes: %d\n", num_nodes);
        printf("  Trees: %zu, Branching: %d\n", this->tree_roots_.size(), this->branching_);
        printf("  Parents: %d, Leaves: %d\n", num_parents, num_leaves);
        printf("\n");

        // ========================================================================
        // PHASE 4: Hybrid Array Construction - Matching OpenCL Exactly
        // ========================================================================
        // Build hybrid flat array containing both pointers AND embedded leaf data
        // Structure: [node_pointers...][leaf_region_1: count, idx...][leaf_region_2...]

        printf("[PHASE 4] Building hybrid nodeIndex array (OpenCL-style)\n");

        // Calculate hybrid array size
        int num_leaf_indices = dataset_indices.size();
        int hybrid_size = num_nodes + num_leaf_indices + num_leaves;  // pointers + data + counts

        printf("  Hybrid array size: %d (nodes: %d + leaf_data: %d + counts: %d)\n",
               hybrid_size, num_nodes, num_leaf_indices, num_leaves);

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

        printf("  Built hybrid array with %d total elements\n", next_data_ptr);
        printf("  Pointer section: [0..%d], Data section: [%d..%d]\n",
               num_nodes - 1, num_nodes, next_data_ptr - 1);

        // Validation: Print first 20 values
        printf("  First 20 hybrid array values:\n    ");
        for (int i = 0; i < 20 && i < hybrid_size; ++i) {
            printf("%d ", hybrid_node_index[i]);
        }
        printf("\n");

        // Show example leaf data region
        if (num_nodes > 0) {
            int first_leaf_idx = -1;
            for (int i = 0; i < num_nodes; ++i) {
                if (nodes_host[i].child_start < 0) {
                    first_leaf_idx = i;
                    break;
                }
            }
            if (first_leaf_idx >= 0) {
                int leaf_ptr = hybrid_node_index[first_leaf_idx];
                int leaf_count = hybrid_node_index[leaf_ptr];
                printf("  Example leaf node %d: ptr=%d, count=%d, indices=[",
                       first_leaf_idx, leaf_ptr, leaf_count);
                for (int j = 0; j < std::min(5, leaf_count); ++j) {
                    printf("%d ", hybrid_node_index[leaf_ptr + 1 + j]);
                }
                if (leaf_count > 5) printf("...");
                printf("]\n");
            }
        }
        printf("\n");

        // ========================================================================
        // End of hybrid array construction
        // ========================================================================

        // Allocate and upload GPU buffers using .resize()
        fprintf(stderr, "[DEBUG] Before gpu_nodes_.resize(%d)\n", num_nodes);
        gpu_nodes_.resize(num_nodes);
        fprintf(stderr, "[DEBUG] Before gpu_pivots_.resize(%d)\n", num_nodes * padded_veclen);
        gpu_pivots_.resize(num_nodes * padded_veclen);
        fprintf(stderr, "[DEBUG] Before gpu_dataset_.resize(%d)\n", this->size_ * padded_veclen);
        gpu_dataset_.resize(this->size_ * padded_veclen);
        fprintf(stderr, "[DEBUG] Before gpu_dataset_indices_.resize(%d)\n", (int)dataset_indices.size());
        gpu_dataset_indices_.resize(dataset_indices.size());
        fprintf(stderr, "[DEBUG] Before gpu_node_index_.resize(%d) - HYBRID ARRAY\n", hybrid_size);
        gpu_node_index_.resize(hybrid_size);  // CRITICAL: Hybrid array with embedded leaf data
        fprintf(stderr, "[DEBUG] After all resizes\n");

        fprintf(stderr, "[DEBUG] Before gpu_nodes_.upload\n");
        gpu_nodes_.upload(nodes_host.data(), num_nodes);
        fprintf(stderr, "[DEBUG] Before gpu_pivots_.upload\n");
        gpu_pivots_.upload(pivots_host.data(), num_nodes * padded_veclen);
        fprintf(stderr, "[DEBUG] Before gpu_dataset_.upload\n");
        gpu_dataset_.upload(dataset_host.data(), this->size_ * padded_veclen);
        fprintf(stderr, "[DEBUG] Before gpu_dataset_indices_.upload\n");
        gpu_dataset_indices_.upload(dataset_indices.data(), dataset_indices.size());
        fprintf(stderr, "[DEBUG] Before gpu_node_index_.upload - HYBRID ARRAY\n");
        gpu_node_index_.upload(hybrid_node_index.data(), hybrid_size);  // CRITICAL: Upload hybrid array
        fprintf(stderr, "[DEBUG] After all uploads\n");

        // ========================================================================
        // PHASE 1 VALIDATION: Save CUDA structure to file for comparison with OpenCL
        // ========================================================================
        {
            FILE* cuda_log = fopen("/tmp/cuda_structure.log", "w");
            if (cuda_log) {
                fprintf(cuda_log, "=== CUDA HIERARCHICAL INDEX STRUCTURE ===\n\n");

                // Log tree structure summary
                fprintf(cuda_log, "TREE SUMMARY:\n");
                fprintf(cuda_log, "  Total nodes: %d\n", num_nodes);
                fprintf(cuda_log, "  Parents: %d, Leaves: %d\n", num_parents, num_leaves);
                fprintf(cuda_log, "  Trees: %zu, Branching: %d\n", this->tree_roots_.size(), this->branching_);
                fprintf(cuda_log, "  Hybrid array size: %d\n\n", hybrid_size);

                // Log first 100 node structures
                fprintf(cuda_log, "FIRST 100 NODES (child_start values):\n");
                for (int i = 0; i < 100 && i < num_nodes; ++i) {
                    const KMeansNodeGPU& node = nodes_host[i];
                    fprintf(cuda_log, "  Node[%3d]: child_start=%6d, count=%3d, %s",
                           i, node.child_start, node.child_count,
                           node.child_start < 0 ? "LEAF\n" : "PARENT\n");
                }
                fprintf(cuda_log, "\n");

                // Log first 100 hybrid array values
                fprintf(cuda_log, "FIRST 100 HYBRID ARRAY VALUES:\n");
                for (int i = 0; i < 100 && i < hybrid_size; ++i) {
                    fprintf(cuda_log, "  hybrid[%3d] = %6d\n", i, hybrid_node_index[i]);
                }
                fprintf(cuda_log, "\n");

                // Log first 100 pivot descriptors (first 8 bytes each)
                fprintf(cuda_log, "FIRST 100 PIVOTS (first 8 bytes each):\n");
                for (int i = 0; i < 100 && i < num_nodes; ++i) {
                    const ElementType* pivot = &pivots_host[i * padded_veclen];
                    fprintf(cuda_log, "  Pivot[%3d]: ", i);
                    for (int j = 0; j < 8 && j < padded_veclen; ++j) {
                        fprintf(cuda_log, "%02x ", (unsigned char)pivot[j]);
                    }
                    fprintf(cuda_log, "\n");
                }
                fprintf(cuda_log, "\n");

                fclose(cuda_log);
                printf("[PHASE 1] CUDA structure saved to /tmp/cuda_structure.log\n");
            } else {
                fprintf(stderr, "[ERROR] Could not open /tmp/cuda_structure.log for writing\n");
            }
        }
        // ========================================================================

        // Debug: Print first 4 nodes (tree roots)
        printf("[DEBUG] Tree root nodes:\n");
        for (int i = 0; i < 4 && i < num_nodes; ++i) {
            printf("  Node %d: child_start=%d, child_count=%d, level=%d\n",
                   i, nodes_host[i].child_start, nodes_host[i].child_count, nodes_host[i].level);
        }

        // Debug: Show pivot storage pattern
        printf("[DEBUG] Pivot array layout (first 8 pivots, showing first 4 bytes each):\n");
        for (int i = 0; i < 8 && i < num_nodes; ++i) {
            const ElementType* pivot_ptr = &pivots_host[i * padded_veclen];
            printf("  Pivot[%d]: %02x %02x %02x %02x", i,
                   (unsigned char)pivot_ptr[0], (unsigned char)pivot_ptr[1],
                   (unsigned char)pivot_ptr[2], (unsigned char)pivot_ptr[3]);
            if (i < 4) {
                printf(" (root %d)\n", i);
            } else {
                printf(" (tree 0 child %d)\n", i - 4);
            }
        }

        // ========================================================================
        // PHASE 1: Data Structure Verification (Debug Instrumentation)
        // ========================================================================
        printf("\n[PHASE 1] Verifying uploaded data structures...\n");

        // Verify tree nodes (download first 32)
        int verify_node_count = std::min(32, num_nodes);
        std::vector<KMeansNodeGPU> nodes_verify(verify_node_count);
        gpu_nodes_.download(nodes_verify.data(), verify_node_count);

        printf("[CUDA NODES] First %d nodes:\n", verify_node_count);
        for (int i = 0; i < verify_node_count && i < 8; ++i) {
            printf("  Node %d: child_start=%d, child_count=%d, pivot_idx=%d, level=%d\n",
                   i, nodes_verify[i].child_start, nodes_verify[i].child_count,
                   nodes_verify[i].pivot_index, nodes_verify[i].level);
        }

        // Verify pivots (download first 8 full descriptors)
        int verify_pivot_count = std::min(8, num_nodes);
        std::vector<ElementType> pivots_verify(verify_pivot_count * padded_veclen);
        gpu_pivots_.download(pivots_verify.data(), verify_pivot_count * padded_veclen);

        printf("[CUDA PIVOTS] First %d pivots (showing first 16 bytes each):\n", verify_pivot_count);
        for (int i = 0; i < verify_pivot_count; ++i) {
            printf("  Pivot %d: ", i);
            for (int j = 0; j < padded_veclen && j < 16; ++j) {
                printf("%02x ", (unsigned char)pivots_verify[i * padded_veclen + j]);
            }
            if (padded_veclen > 16) printf("... ");
            printf("(padded_len=%d)\n", padded_veclen);
        }

        // Verify dataset (download first 4 descriptors)
        int verify_dataset_count = std::min(4, (int)this->size_);
        std::vector<ElementType> dataset_verify(verify_dataset_count * padded_veclen);
        gpu_dataset_.download(dataset_verify.data(), verify_dataset_count * padded_veclen);

        printf("[CUDA DATASET] First %d dataset descriptors (showing first 16 bytes each):\n", verify_dataset_count);
        for (int i = 0; i < verify_dataset_count; ++i) {
            printf("  Dataset %d: ", i);
            for (int j = 0; j < padded_veclen && j < 16; ++j) {
                printf("%02x ", (unsigned char)dataset_verify[i * padded_veclen + j]);
            }
            if (padded_veclen > 16) printf("... ");
            printf("\n");
        }

        printf("[PHASE 1] Data structure verification complete.\n\n");
        // ========================================================================

        // ========================================================================
        // VALIDATION: Compare tree_nodes.child_start with OpenCL nodeIndex values
        // ========================================================================
        printf("\n[VALIDATION] CUDA tree_nodes.child_start values (compare with OpenCL nodeIndex):\n");
        printf("For OpenCL comparison: nodeIndex[i] should equal tree_nodes[i].child_start\n");
        printf("(if structures are equivalent)\n\n");

        int validation_count = std::min(10, num_nodes);
        std::vector<KMeansNodeGPU> validation_nodes(validation_count);
        gpu_nodes_.download(validation_nodes.data(), validation_count);

        printf("First %d CUDA tree_nodes[i].child_start values:\n", validation_count);
        for (int i = 0; i < validation_count; ++i) {
            int child_start = validation_nodes[i].child_start;
            int child_count = validation_nodes[i].child_count;
            int level = validation_nodes[i].level;

            if (child_start >= 0) {
                // Parent node - child_start is index to first child
                printf("  tree_nodes[%d].child_start = %d  (parent: %d children at indices %d-%d, level %d)\n",
                       i, child_start, child_count, child_start, child_start + child_count - 1, level);
            } else {
                // Leaf node - child_start is negative offset
                int leaf_offset = -(child_start + 1);
                printf("  tree_nodes[%d].child_start = %d  (leaf: offset=%d, %d points, level %d)\n",
                       i, child_start, leaf_offset, child_count, level);
            }
        }

        printf("\n[VALIDATION] Expected OpenCL nodeIndex[i] values should match above child_start values.\n");
        printf("If OpenCL shows DIFFERENT values, CUDA needs nodeIndex-equivalent array.\n\n");
        // ========================================================================

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

        printf("[SEARCH DEBUG] veclen_=%zu, padded_bytes=%d (multiple of 4, matching OpenCL), sizeof(ElementType)=%zu\n",
               this->veclen_, padded_bytes, sizeof(ElementType));

        // Upload queries to GPU
        CUDABuffer<ElementType> queries_gpu(num_queries * padded_bytes);
        std::vector<ElementType> padded_queries(num_queries * padded_bytes, 0);

        // Pad queries to padded_bytes
        for (size_t i = 0; i < num_queries; ++i) {
            std::memcpy(&padded_queries[i * padded_bytes],
                       queries[i],
                       this->veclen_ * sizeof(ElementType));
            // Padding bytes remain zero from initialization
        }
        queries_gpu.upload(padded_queries.data(), num_queries * padded_bytes);

        // Allocate result buffers on GPU
        CUDABuffer<int> indices_gpu(num_queries * knn);
        CUDABuffer<int> dists_gpu(num_queries * knn);

        // Calculate grid/block dimensions
        int threads_per_block = 128;
        int num_blocks = (num_queries + threads_per_block - 1) / threads_per_block;
        dim3 grid(num_blocks);
        dim3 block(threads_per_block);

        // Launch kernel (links to kernel compiled with nvcc)
        int num_nodes = gpu_nodes_.count();
        bool success = launch_hierarchical_search(
            gpu_dataset_.get(),
            queries_gpu.get(),
            gpu_nodes_.get(),
            gpu_pivots_.get(),
            gpu_dataset_indices_.get(),
            gpu_node_index_.get(),  // CRITICAL FIX: Pass nodeIndex indirection array
            indices_gpu.get(),
            dists_gpu.get(),
            num_queries,
            padded_bytes,      // For array indexing (data stored with padding)
            this->veclen_,     // For Hamming distance (actual descriptor length)
            num_nodes,
            knn,
            max_checks,
            gpu_num_trees_,    // Number of trees (roots at indices 0..num_trees-1)
            this->branching_,  // Branching factor (tree N's children start at N*branching)
            grid,
            block
        );

        if (!success) {
            throw FLANNException("Unsupported k value for GPU search");
        }

        // Check for kernel errors
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download results
        std::vector<int> indices_host(num_queries * knn);
        std::vector<int> dists_host(num_queries * knn);
        indices_gpu.download(indices_host.data(), num_queries * knn);
        dists_gpu.download(dists_host.data(), num_queries * knn);

        // Copy to output matrices (convert int -> size_t for indices, int -> DistanceType for dists)
        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < knn; ++j) {
                indices[i][j] = static_cast<size_t>(indices_host[i * knn + j]);
                dists[i][j] = static_cast<DistanceType>(dists_host[i * knn + j]);
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
};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_HIERARCHICAL_CUDA_INDEX_H_
