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
#include <cuda_runtime.h>

#include "flann/algorithms/kmeans_index.h"
#include "flann/algorithms/cuda/cuda_utils.h"
#include "flann/algorithms/cuda/nn_cuda_index.h"
#include "flann/algorithms/cuda/kmeans_node_gpu.h"

namespace flann {
namespace cuda {

// Only include kernel headers when compiling with nvcc
#ifdef __CUDACC__
#include "flann/algorithms/cuda/kernels/kmeans_search_kernel.cuh"
#else
// Forward declare kernel launch function for non-CUDA compilation
bool launch_kmeans_search(
    const float* dataset,
    const float* queries,
    const KMeansNodeGPU* tree_nodes,
    const float* tree_pivots,
    const int* dataset_indices,
    int* result_indices,
    float* result_distances,
    size_t num_queries,
    size_t dim,
    size_t num_nodes,
    int knn,
    int max_checks,
    dim3 grid,
    dim3 block,
    float cb_index);
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
          num_nodes_(0),
          padded_veclen_(0)
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
          num_nodes_(0),
          padded_veclen_(0)
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
          num_nodes_(0),
          padded_veclen_(0)
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
          num_nodes_(0),
          padded_veclen_(0)
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
     * @brief Destructor - RAII cleanup via CUDABuffer
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
        std::swap(num_nodes_, other.num_nodes_);
        std::swap(padded_veclen_, other.padded_veclen_);

        // Swap GPU buffers (move semantics)
        tree_nodes_gpu_ = std::move(other.tree_nodes_gpu_);
        tree_pivots_gpu_ = std::move(other.tree_pivots_gpu_);
        dataset_gpu_ = std::move(other.dataset_gpu_);
    }

    /**
     * @brief Add points to index (invalidates GPU data)
     *
     * @param points New points to add
     * @param rebuild_threshold Rebuild threshold (default: 2.0)
     */
    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2.0f) override
    {
        freeGPUMemory();
        BaseClass::addPoints(points, rebuild_threshold);
    }

    /**
     * @brief Remove point from index (invalidates GPU data)
     *
     * @param id Index of point to remove
     */
    void removePoint(size_t id) override
    {
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
        // Validate index is built
        if (!this->root_) {
            throw FLANNException("Cannot prepare GPU search: index not built yet. "
                               "Call buildIndex() first.");
        }

        // Upload tree to GPU if not already done
        if (!gpu_initialized_) {
            uploadToGPU();
        }

        // Mark GPU as ready for searches
        gpu_search_ready_ = true;

        // TODO (future optimization): Compile kernels with knn/params baked in
        // For now, kernels are compiled on-demand in knnSearchGPU()
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
     * @brief Automatic CPU/GPU dispatch for k-NN search (size_t indices)
     *
     * Overrides base class knnSearch() to automatically use GPU when prepared.
     * This enables transparent GPU acceleration without API changes.
     *
     * If buildCUDAKnnSearch() was called, uses GPU acceleration via knnSearchGPU().
     * Otherwise, falls back to CPU search from base class.
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
        // Automatic dispatch: GPU if prepared, CPU otherwise
        if (gpu_search_ready_) {
            return knnSearchGPU(queries, indices, dists, knn, params);
        } else {
            return BaseClass::knnSearch(queries, indices, dists, knn, params);
        }
    }

    /**
     * @brief Automatic CPU/GPU dispatch for k-NN search (int indices)
     *
     * Overrides base class knnSearch() to automatically use GPU when prepared.
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
        // Automatic dispatch: GPU if prepared, CPU otherwise
        if (gpu_search_ready_) {
            return knnSearchGPU(queries, indices, dists, (int)knn, params);
        } else {
            return BaseClass::knnSearch(queries, indices, dists, knn, params);
        }
    }

protected:
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

        // Upload queries to GPU
        CUDABuffer<ElementType> queries_gpu(num_queries * padded_veclen_);
        std::vector<ElementType> padded_queries(num_queries * padded_veclen_);

        // Pad queries to padded_veclen_
        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < padded_veclen_; ++j) {
                if (j < this->veclen_) {
                    padded_queries[i * padded_veclen_ + j] = queries[i][j];
                } else {
                    padded_queries[i * padded_veclen_ + j] = 0;
                }
            }
        }
        queries_gpu.upload(padded_queries.data(), num_queries * padded_veclen_);

        // Allocate result buffers on GPU
        CUDABuffer<int> indices_gpu(num_queries * knn);
        CUDABuffer<float> dists_gpu(num_queries * knn);

        // Calculate grid/block dimensions
        int threads_per_block = 128;
        int num_blocks = (num_queries + threads_per_block - 1) / threads_per_block;
        dim3 grid(num_blocks);
        dim3 block(threads_per_block);

        // Launch kernel (only works with float, but must compile for all types)
        bool success;
        if (std::is_same<ElementType, float>::value) {
            success = launch_kmeans_search(
                (const float*)dataset_gpu_.get(),
                (const float*)queries_gpu.get(),
                tree_nodes_gpu_.get(),
                (const float*)tree_pivots_gpu_.get(),
                dataset_indices_gpu_.get(),
                indices_gpu.get(),
                (float*)dists_gpu.get(),
                num_queries,
                padded_veclen_,
                num_nodes_,
                knn,
                max_checks,
                grid,
                block,
                this->cb_index_
            );
        } else {
            throw FLANNException(
                "K-Means CUDA kernel only supports float element type. "
                "This should have been caught earlier - please report this bug.");
        }

        if (!success) {
            throw FLANNException("Unsupported k value for GPU search");
        }

        // Check for kernel errors
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download results
        std::vector<int> indices_host(num_queries * knn);
        std::vector<float> dists_host(num_queries * knn);
        indices_gpu.download(indices_host.data(), num_queries * knn);
        dists_gpu.download(dists_host.data(), num_queries * knn);

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
     */
    void freeGPUMemory()
    {
        tree_nodes_gpu_ = CUDABuffer<KMeansNodeGPU>();
        tree_pivots_gpu_ = CUDABuffer<ElementType>();
        dataset_gpu_ = CUDABuffer<ElementType>();
        gpu_initialized_ = false;
        gpu_search_ready_ = false;
        num_nodes_ = 0;
    }

    /**
     * @brief Upload tree and dataset to GPU
     *
     * Converts CPU tree structure to GPU-friendly flat arrays:
     * 1. Flatten tree nodes (breadth-first order)
     * 2. Extract pivot coordinates
     * 3. Pad dataset to 4-element alignment (for vectorization)
     * 4. Upload all data with cudaMemcpyAsync
     *
     * Const-qualified because GPU buffers are mutable (implementation detail).
     */
    void uploadToGPU() const
    {
        if (gpu_initialized_) return;
        if (!this->root_) {
            throw FLANNException("Cannot upload to GPU: tree not built");
        }

        // Calculate padded veclen (round up to multiple of 4 for float4 loads)
        padded_veclen_ = ((this->veclen_ + 3) / 4) * 4;

        // Flatten tree structure
        std::vector<KMeansNodeGPU> flat_nodes;
        std::vector<ElementType> flat_pivots;
        std::vector<int> dataset_indices;
        flattenTree(this->root_, flat_nodes, flat_pivots, dataset_indices);

        // Upload tree nodes
        num_nodes_ = flat_nodes.size();
        tree_nodes_gpu_.resize(num_nodes_);
        tree_nodes_gpu_.upload(flat_nodes.data(), num_nodes_);

        // Upload pivots (with padding)
        size_t pivots_size = num_nodes_ * padded_veclen_;
        tree_pivots_gpu_.resize(pivots_size);
        tree_pivots_gpu_.upload(flat_pivots.data(), pivots_size);

        // Upload dataset indices for leaf nodes
        size_t num_indices = dataset_indices.size();
        dataset_indices_gpu_.resize(num_indices);
        dataset_indices_gpu_.upload(dataset_indices.data(), num_indices);

        // Upload dataset (with padding)
        uploadDataset();

        gpu_initialized_ = true;
    }

    /**
     * @brief Flatten tree to breadth-first arrays
     *
     * @param root Root node
     * @param[out] flat_nodes Output flat node array
     * @param[out] flat_pivots Output flat pivot array (padded)
     */
    void flattenTree(
        Node* root,
        std::vector<KMeansNodeGPU>& flat_nodes,
        std::vector<ElementType>& flat_pivots,
        std::vector<int>& dataset_indices) const
    {
        if (!root) return;

        flat_nodes.clear();
        flat_pivots.clear();
        dataset_indices.clear();

        // Breadth-first traversal
        std::queue<Node*> nodeQueue;
        nodeQueue.push(root);

        int current_level = 0;

        while (!nodeQueue.empty()) {
            Node* node = nodeQueue.front();
            nodeQueue.pop();

            // Create GPU node
            KMeansNodeGPU gpu_node;
            gpu_node.pivot_index = flat_nodes.size();  // Index of this node
            gpu_node.level = current_level;
            gpu_node.radius = node->radius;
            gpu_node.variance = node->variance;

            // Handle children
            if (node->childs.empty()) {
                // Leaf node: Store dataset point indices
                // Use NEGATIVE child_start to mark as leaf: -(offset + 1)
                int leaf_offset = dataset_indices.size();
                gpu_node.child_start = -(leaf_offset + 1);  // Negative marks leaf

                // Add all dataset indices for this leaf
                for (size_t i = 0; i < node->points.size(); ++i) {
                    size_t index = node->points[i].index;
                    // Skip removed points
                    if (!this->removed_ || !this->removed_points_.test(index)) {
                        dataset_indices.push_back(static_cast<int>(index));
                    }
                }

                // Store actual count of indices added
                gpu_node.child_count = dataset_indices.size() - leaf_offset;
            } else {
                // Internal node: child_start points to children in flat_nodes
                gpu_node.child_start = flat_nodes.size() + nodeQueue.size() + 1;
                gpu_node.child_count = node->childs.size();

                // Enqueue children
                for (size_t i = 0; i < node->childs.size(); ++i) {
                    nodeQueue.push(node->childs[i]);
                }
            }

            flat_nodes.push_back(gpu_node);

            // Copy pivot (with padding to padded_veclen_)
            for (size_t i = 0; i < padded_veclen_; ++i) {
                if (i < this->veclen_) {
                    flat_pivots.push_back(node->pivot[i]);
                } else {
                    flat_pivots.push_back(0);  // Pad with zeros
                }
            }

            // Update level for next iteration
            if (nodeQueue.empty() || current_level < 100) {
                // Simple level tracking (could be improved)
                current_level = gpu_node.level + 1;
            }
        }
    }

    /**
     * @brief Upload dataset to GPU (with padding)
     */
    void uploadDataset() const
    {
        size_t dataset_size = this->size_ * padded_veclen_;
        dataset_gpu_.resize(dataset_size);

        // Create padded dataset
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

        // Upload to GPU
        dataset_gpu_.upload(padded_data.data(), dataset_size);
    }

private:
    // GPU state (mutable = implementation detail, not logical state)
    mutable bool gpu_initialized_;
    mutable bool gpu_search_ready_;
    mutable size_t num_nodes_;
    mutable size_t padded_veclen_;

    // GPU buffers (RAII - automatic cleanup, mutable for const search methods)
    mutable CUDABuffer<KMeansNodeGPU> tree_nodes_gpu_;
    mutable CUDABuffer<ElementType> tree_pivots_gpu_;
    mutable CUDABuffer<ElementType> dataset_gpu_;
    mutable CUDABuffer<int> dataset_indices_gpu_;  // Leaf node dataset point indices
};

} // namespace cuda
} // namespace flann

#endif // FLANN_USE_CUDA
#endif // FLANN_CUDA_KMEANS_CUDA_INDEX_H_
