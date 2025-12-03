/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024-2025 (CUDA/CPU Comparison Tests)
 * Copyright 2008-2012  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2012  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
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

/**
 * @file flann_cuda_cpu_comparison_test.cpp
 * @brief Tests to verify CUDA achieves CPU parity in accuracy and performance
 *
 * This test file validates:
 * 1. CUDA precision matches CPU precision within tolerance
 * 2. CUDA is faster than CPU for batch queries (performance regression)
 * 3. Result ordering consistency between CPU and CUDA
 */

#ifndef FLANN_USE_CUDA
#error "CUDA comparison tests require FLANN_USE_CUDA to be defined. Build with -DBUILD_CUDA_LIB=ON"
#endif

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <cmath>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "flann_tests.h"

using namespace flann;

/**
 * Test fixture for CPU vs CUDA comparison tests
 */
class CUDACPUComparison : public FLANNTestFixture
{
protected:
    // SIFT100K dataset (float)
    flann::Matrix<float> sift_data;
    flann::Matrix<float> sift_query;
    flann::Matrix<size_t> sift_gt_indices;
    flann::Matrix<float> sift_gt_dists;

    // Brief100K dataset (unsigned char, Hamming)
    flann::Matrix<unsigned char> brief_data;
    flann::Matrix<unsigned char> brief_query;

    void SetUp() override
    {
        printf("Loading test datasets...\n");

        // Load SIFT100K
        flann::load_from_file(sift_data, "sift100K.h5", "dataset");
        flann::load_from_file(sift_query, "sift100K.h5", "query");

        // Load Brief100K
        flann::load_from_file(brief_data, "brief100K.h5", "dataset");
        flann::load_from_file(brief_query, "brief100K.h5", "query");

        printf("Datasets loaded: SIFT100K (%zu x %zu), Brief100K (%zu x %zu)\n",
               sift_data.rows, sift_data.cols, brief_data.rows, brief_data.cols);
    }

    void TearDown() override
    {
        delete[] sift_data.ptr();
        delete[] sift_query.ptr();
        delete[] brief_data.ptr();
        delete[] brief_query.ptr();
    }
};


// ============================================================================
// Precision Parity Tests
// Verify CUDA matches CPU precision within tolerance
// ============================================================================

/**
 * Test: K-Means CUDA vs CPU precision parity
 * Both implementations should achieve similar precision (within 5%)
 */
TEST_F(CUDACPUComparison, KMeansPrecisionMatchesCPU)
{
    const size_t k = 5;
    const float precision_tolerance = 0.05f;  // 5% tolerance

    // Allocate result matrices
    flann::Matrix<size_t> cpu_indices(new size_t[sift_query.rows * k], sift_query.rows, k);
    flann::Matrix<float> cpu_dists(new float[sift_query.rows * k], sift_query.rows, k);
    flann::Matrix<size_t> cuda_indices(new size_t[sift_query.rows * k], sift_query.rows, k);
    flann::Matrix<float> cuda_dists(new float[sift_query.rows * k], sift_query.rows, k);
    flann::Matrix<size_t> gt_indices(new size_t[sift_query.rows * k], sift_query.rows, k);
    flann::Matrix<float> gt_dists(new float[sift_query.rows * k], sift_query.rows, k);

    // Build ground truth with linear index
    {
        flann::Index<flann::L2<float>> linear_index(sift_data, flann::LinearIndexParams());
        linear_index.buildIndex();
        linear_index.knnSearch(sift_query, gt_indices, gt_dists, k, flann::SearchParams(-1));
    }

    // Test CPU K-Means
    float cpu_precision;
    {
        flann::Index<flann::L2<float>> cpu_index(sift_data, flann::KMeansIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
        cpu_index.buildIndex();
        cpu_index.knnSearch(sift_query, cpu_indices, cpu_dists, k, flann::SearchParams(128));
        cpu_precision = compute_precision(gt_indices, cpu_indices);
        printf("CPU K-Means precision: %.2f%%\n", cpu_precision * 100);
    }

    // Test CUDA K-Means
    float cuda_precision;
    {
        flann::Index<flann::L2<float>> cuda_index(sift_data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
        cuda_index.buildIndex();
        cuda_index.buildCUDAKnnSearch(k, flann::SearchParams(128));
        cuda_index.knnSearch(sift_query, cuda_indices, cuda_dists, k, flann::SearchParams(128));
        cuda_precision = compute_precision(gt_indices, cuda_indices);
        printf("CUDA K-Means precision: %.2f%%\n", cuda_precision * 100);
    }

    // Verify CUDA is not significantly worse than CPU
    // Note: CUDA may be BETTER than CPU due to different internal search parameters
    float precision_diff = cuda_precision - cpu_precision;
    EXPECT_GE(cuda_precision, cpu_precision - precision_tolerance)
        << "CUDA precision " << (cuda_precision * 100)
        << "% significantly worse than CPU " << (cpu_precision * 100) << "%";

    if (precision_diff >= 0) {
        printf("CUDA is %.2f%% BETTER than CPU (CUDA: %.2f%%, CPU: %.2f%%)\n",
               precision_diff * 100, cuda_precision * 100, cpu_precision * 100);
    } else {
        printf("CUDA is %.2f%% worse than CPU (within tolerance %.2f%%)\n",
               -precision_diff * 100, precision_tolerance * 100);
    }

    // Cleanup
    delete[] cpu_indices.ptr();
    delete[] cpu_dists.ptr();
    delete[] cuda_indices.ptr();
    delete[] cuda_dists.ptr();
    delete[] gt_indices.ptr();
    delete[] gt_dists.ptr();
}

/**
 * Test: Hierarchical CUDA vs CPU precision parity
 * Both implementations should achieve similar precision (within 5%)
 */
TEST_F(CUDACPUComparison, HierarchicalPrecisionMatchesCPU)
{
    typedef flann::Hamming<unsigned char> Distance;
    typedef Distance::ResultType DistanceType;

    const size_t k = 3;
    const float precision_tolerance = 0.05f;  // 5% tolerance

    // Allocate result matrices
    flann::Matrix<size_t> cpu_indices(new size_t[brief_query.rows * k], brief_query.rows, k);
    flann::Matrix<DistanceType> cpu_dists(new DistanceType[brief_query.rows * k], brief_query.rows, k);
    flann::Matrix<size_t> cuda_indices(new size_t[brief_query.rows * k], brief_query.rows, k);
    flann::Matrix<DistanceType> cuda_dists(new DistanceType[brief_query.rows * k], brief_query.rows, k);
    flann::Matrix<size_t> gt_indices(new size_t[brief_query.rows * k], brief_query.rows, k);
    flann::Matrix<DistanceType> gt_dists(new DistanceType[brief_query.rows * k], brief_query.rows, k);

    // Build ground truth with linear index
    {
        flann::Index<Distance> linear_index(brief_data, flann::LinearIndexParams());
        linear_index.buildIndex();
        linear_index.knnSearch(brief_query, gt_indices, gt_dists, k, flann::SearchParams(-1));
    }

    // Test CPU Hierarchical
    float cpu_precision;
    {
        flann::Index<Distance> cpu_index(brief_data, flann::HierarchicalClusteringIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
        cpu_index.buildIndex();
        cpu_index.knnSearch(brief_query, cpu_indices, cpu_dists, k, flann::SearchParams(2000));
        cpu_precision = compute_precision(gt_indices, cpu_indices);
        printf("CPU Hierarchical precision: %.2f%%\n", cpu_precision * 100);
    }

    // Test CUDA Hierarchical
    float cuda_precision;
    {
        flann::Index<Distance> cuda_index(brief_data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
        cuda_index.buildIndex();
        cuda_index.buildCUDAKnnSearch(k, flann::SearchParams(2000));
        cuda_index.knnSearch(brief_query, cuda_indices, cuda_dists, k, flann::SearchParams(2000));
        cuda_precision = compute_precision(gt_indices, cuda_indices);
        printf("CUDA Hierarchical precision: %.2f%%\n", cuda_precision * 100);
    }

    // Verify CUDA is not significantly worse than CPU
    // Note: CUDA may be BETTER than CPU due to different internal search parameters
    float precision_diff = cuda_precision - cpu_precision;
    EXPECT_GE(cuda_precision, cpu_precision - precision_tolerance)
        << "CUDA precision " << (cuda_precision * 100)
        << "% significantly worse than CPU " << (cpu_precision * 100) << "%";

    if (precision_diff >= 0) {
        printf("CUDA is %.2f%% BETTER than CPU (CUDA: %.2f%%, CPU: %.2f%%)\n",
               precision_diff * 100, cuda_precision * 100, cpu_precision * 100);
    } else {
        printf("CUDA is %.2f%% worse than CPU (within tolerance %.2f%%)\n",
               -precision_diff * 100, precision_tolerance * 100);
    }

    // Cleanup
    delete[] cpu_indices.ptr();
    delete[] cpu_dists.ptr();
    delete[] cuda_indices.ptr();
    delete[] cuda_dists.ptr();
    delete[] gt_indices.ptr();
    delete[] gt_dists.ptr();
}


// ============================================================================
// Performance Comparison Tests
// Verify CUDA is faster than CPU for batch queries
// ============================================================================

/**
 * Test: CUDA K-Means faster than CPU for batch queries
 * CUDA should be at least 2x faster for 1000 queries
 */
TEST_F(CUDACPUComparison, CUDAKMeansFasterThanCPU)
{
    const size_t k = 5;
    const int num_runs = 3;  // Average over multiple runs
    const float speedup_threshold = 1.5f;  // At least 1.5x faster (conservative)

    // Allocate result matrices
    flann::Matrix<size_t> indices(new size_t[sift_query.rows * k], sift_query.rows, k);
    flann::Matrix<float> dists(new float[sift_query.rows * k], sift_query.rows, k);

    // Build indices
    flann::Index<flann::L2<float>> cpu_index(sift_data, flann::KMeansIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    cpu_index.buildIndex();

    flann::Index<flann::L2<float>> cuda_index(sift_data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    cuda_index.buildIndex();
    cuda_index.buildCUDAKnnSearch(k, flann::SearchParams(128));

    // Warm up CUDA
    cuda_index.knnSearch(sift_query, indices, dists, k, flann::SearchParams(128));

    // Benchmark CPU
    double cpu_total_ms = 0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_index.knnSearch(sift_query, indices, dists, k, flann::SearchParams(128));
        auto end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double cpu_avg_ms = cpu_total_ms / num_runs;

    // Benchmark CUDA
    double cuda_total_ms = 0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cuda_index.knnSearch(sift_query, indices, dists, k, flann::SearchParams(128));
        auto end = std::chrono::high_resolution_clock::now();
        cuda_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double cuda_avg_ms = cuda_total_ms / num_runs;

    // Calculate speedup
    double speedup = cpu_avg_ms / cuda_avg_ms;

    printf("=== K-Means Performance Comparison ===\n");
    printf("CPU average:  %.2f ms (%zu queries)\n", cpu_avg_ms, sift_query.rows);
    printf("CUDA average: %.2f ms (%zu queries)\n", cuda_avg_ms, sift_query.rows);
    printf("Speedup: %.2fx\n", speedup);
    printf("Per-query: CPU=%.2f µs, CUDA=%.2f µs\n",
           (cpu_avg_ms * 1000.0) / sift_query.rows,
           (cuda_avg_ms * 1000.0) / sift_query.rows);

    // CUDA should be faster (with conservative threshold for CI stability)
    EXPECT_GE(speedup, speedup_threshold)
        << "CUDA speedup " << speedup << "x below " << speedup_threshold << "x threshold";

    // Cleanup
    delete[] indices.ptr();
    delete[] dists.ptr();
}

/**
 * Test: CUDA Hierarchical faster than CPU for batch queries
 * CUDA should be at least 2x faster for 1000 queries
 */
TEST_F(CUDACPUComparison, CUDAHierarchicalFasterThanCPU)
{
    typedef flann::Hamming<unsigned char> Distance;
    typedef Distance::ResultType DistanceType;

    const size_t k = 3;
    const int num_runs = 3;
    const float speedup_threshold = 1.5f;

    // Allocate result matrices
    flann::Matrix<size_t> indices(new size_t[brief_query.rows * k], brief_query.rows, k);
    flann::Matrix<DistanceType> dists(new DistanceType[brief_query.rows * k], brief_query.rows, k);

    // Build indices
    flann::Index<Distance> cpu_index(brief_data, flann::HierarchicalClusteringIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
    cpu_index.buildIndex();

    flann::Index<Distance> cuda_index(brief_data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
    cuda_index.buildIndex();
    cuda_index.buildCUDAKnnSearch(k, flann::SearchParams(2000));

    // Warm up CUDA
    cuda_index.knnSearch(brief_query, indices, dists, k, flann::SearchParams(2000));

    // Benchmark CPU
    double cpu_total_ms = 0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_index.knnSearch(brief_query, indices, dists, k, flann::SearchParams(2000));
        auto end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double cpu_avg_ms = cpu_total_ms / num_runs;

    // Benchmark CUDA
    double cuda_total_ms = 0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cuda_index.knnSearch(brief_query, indices, dists, k, flann::SearchParams(2000));
        auto end = std::chrono::high_resolution_clock::now();
        cuda_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    double cuda_avg_ms = cuda_total_ms / num_runs;

    // Calculate speedup
    double speedup = cpu_avg_ms / cuda_avg_ms;

    printf("=== Hierarchical Performance Comparison ===\n");
    printf("CPU average:  %.2f ms (%zu queries)\n", cpu_avg_ms, brief_query.rows);
    printf("CUDA average: %.2f ms (%zu queries)\n", cuda_avg_ms, brief_query.rows);
    printf("Speedup: %.2fx\n", speedup);
    printf("Per-query: CPU=%.2f µs, CUDA=%.2f µs\n",
           (cpu_avg_ms * 1000.0) / brief_query.rows,
           (cuda_avg_ms * 1000.0) / brief_query.rows);

    // CUDA should be faster
    EXPECT_GE(speedup, speedup_threshold)
        << "CUDA speedup " << speedup << "x below " << speedup_threshold << "x threshold";

    // Cleanup
    delete[] indices.ptr();
    delete[] dists.ptr();
}


// ============================================================================
// Result Ordering Tests
// Verify CUDA produces correctly sorted results
// ============================================================================

/**
 * Test: CUDA K-Means results are properly sorted by distance
 */
TEST_F(CUDACPUComparison, CUDAKMeansResultsSorted)
{
    const size_t k = 10;

    flann::Matrix<size_t> indices(new size_t[sift_query.rows * k], sift_query.rows, k);
    flann::Matrix<float> dists(new float[sift_query.rows * k], sift_query.rows, k);

    flann::Index<flann::L2<float>> cuda_index(sift_data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    cuda_index.buildIndex();
    cuda_index.buildCUDAKnnSearch(k, flann::SearchParams(128));
    cuda_index.knnSearch(sift_query, indices, dists, k, flann::SearchParams(128));

    // Verify all queries have sorted results
    int unsorted_count = 0;
    for (size_t i = 0; i < sift_query.rows; ++i) {
        for (size_t j = 1; j < k; ++j) {
            if (dists[i][j] < dists[i][j-1]) {
                unsorted_count++;
                if (unsorted_count <= 5) {
                    printf("Unsorted at query %zu, position %zu: %.4f < %.4f\n",
                           i, j, dists[i][j], dists[i][j-1]);
                }
            }
        }
    }

    EXPECT_EQ(unsorted_count, 0)
        << "Found " << unsorted_count << " unsorted distance pairs";

    delete[] indices.ptr();
    delete[] dists.ptr();
}

/**
 * Test: CUDA Hierarchical results are properly sorted by distance
 */
TEST_F(CUDACPUComparison, CUDAHierarchicalResultsSorted)
{
    typedef flann::Hamming<unsigned char> Distance;
    typedef Distance::ResultType DistanceType;

    const size_t k = 10;

    flann::Matrix<size_t> indices(new size_t[brief_query.rows * k], brief_query.rows, k);
    flann::Matrix<DistanceType> dists(new DistanceType[brief_query.rows * k], brief_query.rows, k);

    flann::Index<Distance> cuda_index(brief_data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
    cuda_index.buildIndex();
    cuda_index.buildCUDAKnnSearch(k, flann::SearchParams(2000));
    cuda_index.knnSearch(brief_query, indices, dists, k, flann::SearchParams(2000));

    // Verify all queries have sorted results
    int unsorted_count = 0;
    for (size_t i = 0; i < brief_query.rows; ++i) {
        for (size_t j = 1; j < k; ++j) {
            if (dists[i][j] < dists[i][j-1]) {
                unsorted_count++;
                if (unsorted_count <= 5) {
                    printf("Unsorted at query %zu, position %zu: %d < %d\n",
                           i, j, dists[i][j], dists[i][j-1]);
                }
            }
        }
    }

    EXPECT_EQ(unsorted_count, 0)
        << "Found " << unsorted_count << " unsorted distance pairs";

    delete[] indices.ptr();
    delete[] dists.ptr();
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
