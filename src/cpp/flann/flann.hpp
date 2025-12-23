/***********************************************************************
 * Software License Agreement (BSD License)
 *
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

#ifndef FLANN_HPP_
#define FLANN_HPP_


#include <vector>
#include <string>
#include <cassert>
#include <cstdio>

#include "flann/general.h"
#include "flann/util/matrix.h"
#include "flann/util/params.h"
#include "flann/util/saving.h"
#ifdef FLANN_USE_CUDA
#include "flann/util/gpu_saving.h"
#endif

#include "flann/algorithms/all_indices.h"

namespace flann
{

/**
 * Sets the log level used for all flann functions
 * @param level Verbosity level
 */
inline void log_verbosity(int level)
{
    if (level >= 0) {
        Logger::setLevel(level);
    }
}

/**
 * (Deprecated) Index parameters for creating a saved index.
 */
struct SavedIndexParams : public IndexParams
{
    SavedIndexParams(std::string filename)
    {
        (*this)["algorithm"] = FLANN_INDEX_SAVED;
        (*this)["filename"] = filename;
    }
};



template<typename Distance>
class Index
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;
    typedef NNIndex<Distance> IndexType;

    Index(const IndexParams& params, Distance distance = Distance() )
        : index_params_(params)
    {
        flann_algorithm_t index_type = get_param<flann_algorithm_t>(params,"algorithm");
        loaded_ = false;

        Matrix<ElementType> features;
        if (index_type == FLANN_INDEX_SAVED) {
            nnIndex_ = load_saved_index(features, get_param<std::string>(params,"filename"), distance);
            loaded_ = true;
        }
        else {
        	flann_algorithm_t index_type = get_param<flann_algorithm_t>(params, "algorithm");
            nnIndex_ = create_index_by_type<Distance>(index_type, features, params, distance);
        }
    }


    Index(const Matrix<ElementType>& features, const IndexParams& params, Distance distance = Distance() )
        : index_params_(params)
    {
        flann_algorithm_t index_type = get_param<flann_algorithm_t>(params,"algorithm");
        loaded_ = false;

        if (index_type == FLANN_INDEX_SAVED) {
            nnIndex_ = load_saved_index(features, get_param<std::string>(params,"filename"), distance);
            loaded_ = true;
        }
        else {
        	flann_algorithm_t index_type = get_param<flann_algorithm_t>(params, "algorithm");
            nnIndex_ = create_index_by_type<Distance>(index_type, features, params, distance);
        }
    }


    Index(const Index& other) : loaded_(other.loaded_), index_params_(other.index_params_)
    {
    	nnIndex_ = other.nnIndex_->clone();
    }

    Index& operator=(Index other)
    {
    	this->swap(other);
    	return *this;
    }

    virtual ~Index()
    {
        delete nnIndex_;
    }

    /**
     * Builds the index.
     */
    void buildIndex()
    {
        if (!loaded_) {
            nnIndex_->buildIndex();
        }
    }

    void buildIndex(const Matrix<ElementType>& points)
    {
    	nnIndex_->buildIndex(points);
    }

    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        nnIndex_->addPoints(points, rebuild_threshold);
    }

    /**
     * Remove point from the index
     * @param index Index of point to be removed
     */
    void removePoint(size_t point_id)
    {
    	nnIndex_->removePoint(point_id);
    }

    /**
     * Returns pointer to a data point with the specified id.
     * @param point_id the id of point to retrieve
     * @return
     */
    ElementType* getPoint(size_t point_id)
    {
    	return nnIndex_->getPoint(point_id);
    }

    /**
     * Save index to file
     * @param filename
     */
    void save(std::string filename)
    {
        FILE* fout = fopen(filename.c_str(), "wb");
        if (fout == NULL) {
            throw FLANNException("Cannot open file");
        }
        nnIndex_->saveIndex(fout);
        fclose(fout);
    }

    /**
     * \returns number of features in this index.
     */
    size_t veclen() const
    {
        return nnIndex_->veclen();
    }

    /**
     * \returns The dimensionality of the features in this index.
     */
    size_t size() const
    {
        return nnIndex_->size();
    }

    /**
     * \returns The index type (kdtree, kmeans,...)
     */
    flann_algorithm_t getType() const
    {
        return nnIndex_->getType();
    }

    /**
     * \returns The amount of memory (in bytes) used by the index.
     */
    int usedMemory() const
    {
        return nnIndex_->usedMemory();
    }


    /**
     * \returns The index parameters
     */
    IndexParams getParameters() const
    {
        return nnIndex_->getParameters();
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 Matrix<size_t>& indices,
                                 Matrix<DistanceType>& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param knn
     * @param params
     * @return
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 Matrix<int>& indices,
                                 Matrix<DistanceType>& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 std::vector< std::vector<size_t> >& indices,
                                 std::vector<std::vector<DistanceType> >& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param knn
     * @param params
     * @return
     */
    int knnSearch(const Matrix<ElementType>& queries,
                                 std::vector< std::vector<int> >& indices,
                                 std::vector<std::vector<DistanceType> >& dists,
                                 size_t knn,
                           const SearchParams& params) const
    {
    	return nnIndex_->knnSearch(queries, indices, dists, knn, params);
    }

    /**
     * \brief Perform radius search
     * \param[in] queries The query points
     * \param[out] indices The indices of the neighbors found within the given radius
     * \param[out] dists The distances to the nearest neighbors found
     * \param[in] radius The radius used for search
     * \param[in] params Search parameters
     * \returns Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    Matrix<size_t>& indices,
                                    Matrix<DistanceType>& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param radius
     * @param params
     * @return
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    Matrix<int>& indices,
                                    Matrix<DistanceType>& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     * \brief Perform radius search
     * \param[in] queries The query points
     * \param[out] indices The indices of the neighbors found within the given radius
     * \param[out] dists The distances to the nearest neighbors found
     * \param[in] radius The radius used for search
     * \param[in] params Search parameters
     * \returns Number of neighbors found
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    std::vector< std::vector<size_t> >& indices,
                                    std::vector<std::vector<DistanceType> >& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     *
     * @param queries
     * @param indices
     * @param dists
     * @param radius
     * @param params
     * @return
     */
    int radiusSearch(const Matrix<ElementType>& queries,
                                    std::vector< std::vector<int> >& indices,
                                    std::vector<std::vector<DistanceType> >& dists,
                                    float radius,
                              const SearchParams& params) const
    {
    	return nnIndex_->radiusSearch(queries, indices, dists, radius, params);
    }

    /**
     *
     */
#ifdef FLANN_USE_OPENCL
    void buildCLKnnSearch(size_t knn,
                          const SearchParams& params,
                          cl_command_queue cq = NULL)
    {
        IndexType* nnIndexTmp = nnIndex_;
        switch(nnIndex_->getType()){
            case FLANN_INDEX_HIERARCHICAL:
                nnIndex_ = new HierarchicalClusteringOpenCLIndex<Distance>(*(HierarchicalClusteringIndex<Distance>*)nnIndex_);

                // Delete the non-OpenCL version
                delete nnIndexTmp;
            case FLANN_INDEX_HIERARCHICAL_OPENCL:
                return nnIndex_->buildCLKnnSearch(knn, params, cq);
            case FLANN_INDEX_KMEANS:
                nnIndex_ = new KMeansOpenCLIndex<Distance>(*(KMeansIndex<Distance>*)nnIndex_);

                // Delete the non-OpenCL version
                delete nnIndexTmp;
            case FLANN_INDEX_KMEANS_OPENCL:
                return nnIndex_->buildCLKnnSearch(knn, params, cq);
            default:
                // OpenCL acceleration not supported for other index types
                break;
        }
    }
#endif /* FLANN_USE_OPENCL */

#ifdef FLANN_USE_CUDA
    /**
     * Prepare CUDA-accelerated k-NN search (matches OpenCL pattern)
     *
     * Converts CPU index to CUDA index if needed, then prepares GPU resources.
     * Call once after buildIndex() before search operations.
     *
     * @param knn Number of nearest neighbors
     * @param params Search parameters
     */
    void buildCUDAKnnSearch(size_t knn,
                            const SearchParams& params = SearchParams())
    {
        // Dispatch based on index type (avoids template instantiation issues with switch/static_cast)
        flann_algorithm_t index_type = nnIndex_->getType();

        // Handle K-Means indices
        if (index_type == FLANN_INDEX_KMEANS || index_type == FLANN_INDEX_KMEANS_CUDA) {
            // CUDA K-Means kernel only supports float element type
            if (!std::is_same<ElementType, float>::value) {
                throw FLANNException(
                    "CUDA K-Means index requires float element type. "
                    "For binary descriptors with Hamming distance, use HierarchicalCUDAIndex.");
            }

            // Convert CPU→GPU if needed
            if (index_type == FLANN_INDEX_KMEANS) {
                IndexType* nnIndexTmp = nnIndex_;
                nnIndex_ = new cuda::KMeansCUDAIndex<Distance>(
                    *(KMeansIndex<Distance>*)nnIndex_);
                delete nnIndexTmp;
            }

            // Call GPU preparation (safe cast - type checked above)
            return static_cast<cuda::KMeansCUDAIndex<Distance>*>(
                nnIndex_)->buildCUDAKnnSearch(knn, params);
        }

        // Handle Hierarchical indices
        if (index_type == FLANN_INDEX_HIERARCHICAL || index_type == FLANN_INDEX_HIERARCHICAL_CUDA) {
            // CUDA Hierarchical kernel only supports unsigned char element type
            if (!std::is_same<ElementType, unsigned char>::value) {
                throw FLANNException(
                    "CUDA Hierarchical index requires unsigned char element type (binary descriptors). "
                    "For float descriptors with L2/L1 distance, use KMeansCUDAIndex.");
            }

            // Convert CPU→GPU if needed
            if (index_type == FLANN_INDEX_HIERARCHICAL) {
                IndexType* nnIndexTmp = nnIndex_;
                nnIndex_ = new cuda::HierarchicalCUDAIndex<Distance>(
                    *(HierarchicalClusteringIndex<Distance>*)nnIndex_);
                delete nnIndexTmp;
            }

            // Call GPU preparation (safe cast - type checked above)
            return static_cast<cuda::HierarchicalCUDAIndex<Distance>*>(
                nnIndex_)->buildCUDAKnnSearch(knn, params);
        }

        throw FLANNException(
            "buildCUDAKnnSearch() only supports K-Means and Hierarchical index types. "
            "Supported: FLANN_INDEX_KMEANS, FLANN_INDEX_KMEANS_CUDA, "
            "FLANN_INDEX_HIERARCHICAL, FLANN_INDEX_HIERARCHICAL_CUDA.");
    }

    /**
     * @brief Prepare GPU index for search (K-independent)
     *
     * Uploads tree and dataset to GPU without requiring a K value.
     *
     * @throws FLANNException if index not built or unsupported index type
     */
    void prepareGPUIndex()
    {
        flann_algorithm_t index_type = nnIndex_->getType();

        // Handle K-Means indices
        if (index_type == FLANN_INDEX_KMEANS || index_type == FLANN_INDEX_KMEANS_CUDA) {
            if (!std::is_same<ElementType, float>::value) {
                throw FLANNException(
                    "CUDA K-Means index requires float element type.");
            }
            if (index_type == FLANN_INDEX_KMEANS) {
                IndexType* nnIndexTmp = nnIndex_;
                nnIndex_ = new cuda::KMeansCUDAIndex<Distance>(
                    *(KMeansIndex<Distance>*)nnIndex_);
                delete nnIndexTmp;
            }
            return static_cast<cuda::KMeansCUDAIndex<Distance>*>(
                nnIndex_)->prepareGPUIndex();
        }

        // Handle Hierarchical indices
        if (index_type == FLANN_INDEX_HIERARCHICAL || index_type == FLANN_INDEX_HIERARCHICAL_CUDA) {
            if (!std::is_same<ElementType, unsigned char>::value) {
                throw FLANNException(
                    "CUDA Hierarchical index requires unsigned char element type.");
            }
            if (index_type == FLANN_INDEX_HIERARCHICAL) {
                IndexType* nnIndexTmp = nnIndex_;
                nnIndex_ = new cuda::HierarchicalCUDAIndex<Distance>(
                    *(HierarchicalClusteringIndex<Distance>*)nnIndex_);
                delete nnIndexTmp;
            }
            return static_cast<cuda::HierarchicalCUDAIndex<Distance>*>(
                nnIndex_)->prepareGPUIndex();
        }

        throw FLANNException(
            "prepareGPUIndex() only supports K-Means and Hierarchical index types.");
    }

    /**
     * Check if GPU search is ready
     * @return true if buildCUDAKnnSearch() has been called and GPU is ready
     */
    bool isGPUSearchReady() const
    {
        flann_algorithm_t index_type = nnIndex_->getType();

        if (index_type == FLANN_INDEX_KMEANS_CUDA) {
            return static_cast<cuda::KMeansCUDAIndex<Distance>*>(
                nnIndex_)->isGPUSearchReady();
        }

        if (index_type == FLANN_INDEX_HIERARCHICAL_CUDA) {
            return static_cast<cuda::HierarchicalCUDAIndex<Distance>*>(
                nnIndex_)->isGPUSearchReady();
        }

        // Non-CUDA index types are never GPU-ready
        return false;
    }

    /**
     * Check if index has GPU-optimized format available for saving
     * @return true if GPU buffers are initialized and index can be saved in GPU format
     */
    bool hasGPUFormat() const
    {
        flann_algorithm_t index_type = nnIndex_->getType();

        if (index_type == FLANN_INDEX_KMEANS_CUDA) {
            return static_cast<cuda::KMeansCUDAIndex<Distance>*>(
                nnIndex_)->hasGPUFormat();
        }

        if (index_type == FLANN_INDEX_HIERARCHICAL_CUDA) {
            return static_cast<cuda::HierarchicalCUDAIndex<Distance>*>(
                nnIndex_)->hasGPUFormat();
        }

        return false;
    }

    /**
     * Convert index to GPU-optimized format for faster save/load
     *
     * After calling this method:
     * - GPU search will work
     * - CPU search will NOT work (CPU tree discarded)
     * - Index can be saved in GPU-optimized format
     *
     * @throws FLANNException if index not built or not a CUDA index type
     */
    void convertToGPUFormat()
    {
        flann_algorithm_t index_type = nnIndex_->getType();

        if (index_type == FLANN_INDEX_KMEANS_CUDA) {
            static_cast<cuda::KMeansCUDAIndex<Distance>*>(
                nnIndex_)->convertToGPUFormat();
            return;
        }

        if (index_type == FLANN_INDEX_HIERARCHICAL_CUDA) {
            static_cast<cuda::HierarchicalCUDAIndex<Distance>*>(
                nnIndex_)->convertToGPUFormat();
            return;
        }

        throw FLANNException("convertToGPUFormat() only supported for CUDA index types");
    }

    /**
     * @brief GPU-direct k-NN search with device pointers (zero-copy)
     *
     * Performs k-nearest neighbor search using device-resident data,
     * eliminating CPU<->GPU memory transfers for queries and results.
     * Ideal for GPU-resident pipelines where data never leaves device memory.
     *
     * For HierarchicalCUDAIndex (unsigned char/Hamming):
     *   d_dists is int* (Hamming distance as popcount)
     *
     * For KMeansCUDAIndex (float/L2):
     *   d_dists is float* (L2 squared distance)
     *
     * @param d_queries Device pointer to query vectors [num_queries x veclen]
     * @param d_indices Device pointer for output indices [num_queries x knn]
     * @param d_dists   Device pointer for output distances [num_queries x knn]
     * @param num_queries Number of query vectors
     * @param knn Number of nearest neighbors to find
     * @param params Search parameters
     * @param stream CUDA stream for async execution (caller must synchronize)
     *
     * @return Number of queries processed
     * @throws FLANNException if not a CUDA index or GPU not initialized
     *
     * @note Caller must synchronize on stream before reading results.
     */
    template<typename DistT>
    int knnSearchGPUDirect(
        const ElementType* d_queries,
        int* d_indices,
        DistT* d_dists,
        size_t num_queries,
        size_t knn,
        const SearchParams& params,
        cudaStream_t stream) const
    {
        flann_algorithm_t index_type = nnIndex_->getType();

        if (index_type == FLANN_INDEX_KMEANS_CUDA) {
            static_assert(std::is_same<DistT, float>::value || std::is_same<DistT, int>::value,
                "KMeansCUDAIndex requires float* for d_dists");
            return static_cast<cuda::KMeansCUDAIndex<Distance>*>(nnIndex_)
                ->knnSearchGPUDirect(d_queries, d_indices,
                    reinterpret_cast<float*>(d_dists),
                    num_queries, knn, params, stream);
        }

        if (index_type == FLANN_INDEX_HIERARCHICAL_CUDA) {
            static_assert(std::is_same<DistT, int>::value || std::is_same<DistT, float>::value,
                "HierarchicalCUDAIndex requires int* for d_dists");
            return static_cast<cuda::HierarchicalCUDAIndex<Distance>*>(nnIndex_)
                ->knnSearchGPUDirect(d_queries, d_indices,
                    reinterpret_cast<int*>(d_dists),
                    num_queries, knn, params, stream);
        }

        throw FLANNException("knnSearchGPUDirect() only supported for CUDA index types "
            "(FLANN_INDEX_KMEANS_CUDA, FLANN_INDEX_HIERARCHICAL_CUDA)");
    }
#endif /* FLANN_USE_CUDA */

private:
    IndexType* load_saved_index(const Matrix<ElementType>& dataset, const std::string& filename, Distance distance)
    {
        FILE* fin = fopen(filename.c_str(), "rb");
        if (fin == NULL) {
            return NULL;
        }

#ifdef FLANN_USE_CUDA
        // Check for GPU v2.0 format (has different signature)
        if (GPUIndexHeaderV2::detectV2Format(fin)) {
            // Read GPU header directly (don't use LoadArchive - it expects full file read)
            rewind(fin);
            GPUIndexHeaderV2Struct header;
            if (fread(&header, sizeof(header), 1, fin) != 1) {
                fclose(fin);
                throw FLANNException("Failed to read GPU index header");
            }
            if (header.data_type != flann_datatype_value<ElementType>::value) {
                fclose(fin);
                throw FLANNException("Datatype of saved GPU index is different than of the one to be loaded.");
            }
            IndexParams params;
            params["algorithm"] = header.index_type;
            IndexType* nnIndex = create_index_by_type<Distance>(header.index_type, dataset, params, distance);
            rewind(fin);
            nnIndex->loadIndex(fin);
            fclose(fin);
            return nnIndex;
        }
#else
        // Check for GPU signature without CUDA support - detect by peeking
        {
            char sig[24];
            if (fread(sig, sizeof(sig), 1, fin) == 1) {
                if (strncmp(sig, "FLANN_GPU_INDEX", strlen("FLANN_GPU_INDEX")) == 0) {
                    fclose(fin);
                    throw FLANNException("Cannot load GPU-saved index: FLANN compiled without CUDA support. "
                        "Rebuild FLANN with -DBUILD_CUDA_LIB=ON to load GPU-format indices.");
                }
            }
            rewind(fin);
        }
#endif

        // Standard FLANN format
        IndexHeader header = load_header(fin);
        if (header.h.data_type != flann_datatype_value<ElementType>::value) {
            fclose(fin);
            throw FLANNException("Datatype of saved index is different than of the one to be loaded.");
        }

        IndexParams params;
        params["algorithm"] = header.h.index_type;
        IndexType* nnIndex = create_index_by_type<Distance>(header.h.index_type, dataset, params, distance);
        rewind(fin);
        nnIndex->loadIndex(fin);
        fclose(fin);

        return nnIndex;
    }

    void swap( Index& other)
    {
    	std::swap(nnIndex_, other.nnIndex_);
    	std::swap(loaded_, other.loaded_);
    	std::swap(index_params_, other.index_params_);
    }

private:
    /** Pointer to actual index class */
    IndexType* nnIndex_;
    /** Indices if the index was loaded from a file */
    bool loaded_;
    /** Parameters passed to the index */
    IndexParams index_params_;
};





/**
 * Performs a hierarchical clustering of the points passed as argument and then takes a cut in the
 * the clustering tree to return a flat clustering.
 * @param[in] points Points to be clustered
 * @param centers The computed cluster centres. Matrix should be preallocated and centers.rows is the
 *  number of clusters requested.
 * @param params Clustering parameters (The same as for flann::KMeansIndex)
 * @param d Distance to be used for clustering (eg: flann::L2)
 * @return number of clusters computed (can be different than clusters.rows and is the highest number
 * of the form (branching-1)*K+1 smaller than clusters.rows).
 */
template <typename Distance>
int hierarchicalClustering(const Matrix<typename Distance::ElementType>& points, Matrix<typename Distance::ResultType>& centers,
                           const KMeansIndexParams& params, Distance d = Distance())
{
    KMeansIndex<Distance> kmeans(points, params, d);
    kmeans.buildIndex();

    int clusterNum = kmeans.getClusterCenters(centers);
    return clusterNum;
}

}
#endif /* FLANN_HPP_ */
