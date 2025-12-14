/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024 (CUDA implementation)
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

// Note: FLANN_USE_CUDA is defined via CMake target_compile_definitions
#ifndef FLANN_USE_CUDA
#error "CUDA tests require FLANN_USE_CUDA to be defined. Build with -DBUILD_CUDA_LIB=ON"
#endif

#include <gtest/gtest.h>
#include <time.h>
#include <chrono>
#include <set>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "flann_tests.h"

using namespace flann;

/**
 * Test fixture for SIFT 10K dataset with CUDA
 */
class KMeansCUDA_SIFT10K : public DatasetTestFixture<float, float> {
protected:
    KMeansCUDA_SIFT10K() : DatasetTestFixture("sift10K.h5") {}
};

/**
 * Test 1: Basic Search
 * Validate basic k-NN search with default parameters
 * Expected: ≥95% recall@5 (high precision requirement for production)
 */
TEST_F(KMeansCUDA_SIFT10K, TestSearch)
{
    TestSearch<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.96,  // 96% recall (verified actual: ~99.4%)
        gt_indices
    );
}

/**
 * Test 2: Large Branching Factor
 * Higher branching and checks for better precision
 * Expected: ≥92% recall@10
 */
TEST_F(KMeansCUDA_SIFT10K, TestSearch2)
{
    TestSearch2<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(64, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(256),
        0.96,  // 96% recall (verified actual: ~99.4%)
        gt_indices
    );
}

/**
 * Test 3: Incremental Point Addition
 * Add points incrementally and verify search still works
 */
TEST_F(KMeansCUDA_SIFT10K, TestAddIncremental)
{
    TestAddIncremental<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.96,  // 96% recall (verified actual: ~99.4%)
        gt_indices
    );
}

/**
 * Test 4: Incremental Addition with Rebuild
 * Test rebuild threshold triggering
 */
TEST_F(KMeansCUDA_SIFT10K, TestAddIncremental2)
{
    TestAddIncremental2<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.96,  // 96% recall (verified actual: ~99.4%)
        gt_indices
    );
}

/**
 * Test 5: Point Removal
 * Remove points and verify they don't appear in results
 */
TEST_F(KMeansCUDA_SIFT10K, TestRemove)
{
    TestRemove<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128)
    );
}

/**
 * Test 6: Save/Load Index
 * Verify index serialization and deserialization
 */
TEST_F(KMeansCUDA_SIFT10K, TestSave)
{
    TestSave<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.96,  // 96% recall (verified actual: ~99.4%)
        gt_indices
    );
}

/**
 * Test 7: Copy Constructor
 * Verify copy constructor creates independent index
 */
TEST_F(KMeansCUDA_SIFT10K, TestCopy)
{
    TestCopy<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.96,  // 96% recall (verified actual: ~99.4%)
        gt_indices
    );
}

/**
 * Test 8: Move Constructor
 * Verify move constructor transfers ownership correctly
 */
TEST_F(KMeansCUDA_SIFT10K, TestCopy2)
{
    TestCopy2<KMeansCUDAIndex<flann::L2<float> > >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.96,  // 96% recall (verified actual: ~99.4%)
        gt_indices
    );
}

/**
 * Test fixture for SIFT 100K dataset (larger dataset)
 */
class KMeansCUDA_SIFT100K : public DatasetTestFixture<float, float> {
protected:
    KMeansCUDA_SIFT100K() : DatasetTestFixture("sift100K.h5") {}
};

/**
 * Large Dataset Search Test
 * Verify CUDA performance on 100K dataset
 */
TEST_F(KMeansCUDA_SIFT100K, TestSearch)
{
    TestSearch<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.93,  // 93% recall (verified actual: ~95.8%)
        gt_indices
    );
}

/**
 * Large Dataset Incremental Add Test
 */
TEST_F(KMeansCUDA_SIFT100K, TestAddIncremental)
{
    TestAddIncremental<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.93,  // 93% recall (verified actual: ~96.3%)
        gt_indices
    );
}

/**
 * Large Dataset Incremental Add with Rebuild
 */
TEST_F(KMeansCUDA_SIFT100K, TestAddIncremental2)
{
    TestAddIncremental2<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.93,  // 93% recall (verified actual: ~96.3%)
        gt_indices
    );
}

/**
 * Large Dataset Point Removal Test
 * Remove points and verify they don't appear in results
 */
TEST_F(KMeansCUDA_SIFT100K, TestRemove)
{
    TestRemove<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128)
    );
}

/**
 * Large Dataset Save/Load Test
 * Verify index serialization and deserialization
 */
TEST_F(KMeansCUDA_SIFT100K, TestSave)
{
    TestSave<flann::L2<float> >(
        data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2),
        query, indices, dists, knn,
        flann::SearchParams(128),
        0.93,  // 93% recall (verified actual: ~95.8%)
        gt_indices
    );
}


// ============================================================================
// Byte Dataset Tests (L2<unsigned char>)
//
// STATUS: Deferred to future MR
//
// TECHNICAL BARRIER:
// 1. CUDA K-Means kernel (kmeans_search_cooperative.cuh) is hardcoded for float:
//    - Heap stores float distances: `float heap_dists[LOC_SIZE * 2]`
//    - Query buffer is float: `float query[256]`
//    - Distance computation uses `compute_l2_distance()` which returns float
//
// 2. To support unsigned char with L2 distance:
//    - Template kernel on element type (not just K value)
//    - Use int distances in heap (to avoid precision loss for small values)
//    - Add dispatch logic in kmeans_cuda_index.h for different element types
//    - Add template instantiations in kmeans_cuda_kernels.cu
//
// 3. KDTreeCuda3dIndex stubs throw for non-float types, blocking Index<L2<uchar>>
//    factory instantiation (even though K-Means doesn't use KDTree)
//
// WORKAROUND: Distance kernel functions for unsigned char have been added to
// distance_kernels.cuh (compute_l2_distance_uchar) for future use.
//
// OpenCL works because it doesn't go through the same factory/KDTree path.
// ============================================================================


// ============================================================================
// K-Value Coverage Tests
// Comprehensive testing of all supported k-values
// ============================================================================

/**
 * Test all supported k-values for K-Means CUDA
 * K-Means CUDA supports: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100
 */
TEST_F(KMeansCUDA_SIFT10K, TestAllSupportedKValues)
{
    const std::vector<int> k_values = {1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100};

    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // Build ground truth with linear index
    flann::Index<flann::L2<float>> linear_index(data, flann::LinearIndexParams());
    linear_index.buildIndex();

    printf("\n=== K-Value Coverage Test Results ===\n");

    for (int k : k_values) {
        // Skip k values larger than dataset if needed
        if (static_cast<size_t>(k) > data.rows) continue;

        flann::Matrix<size_t> indices_k(new size_t[query.rows * k], query.rows, k);
        flann::Matrix<float> dists_k(new float[query.rows * k], query.rows, k);
        flann::Matrix<size_t> gt_indices_k(new size_t[query.rows * k], query.rows, k);
        flann::Matrix<float> gt_dists_k(new float[query.rows * k], query.rows, k);

        // Compute ground truth
        linear_index.knnSearch(query, gt_indices_k, gt_dists_k, k, flann::SearchParams(-1));

        // Test CUDA search
        ASSERT_NO_THROW(index.buildCUDAKnnSearch(k, flann::SearchParams(128)))
            << "Failed to setup GPU for k=" << k;
        ASSERT_TRUE(index.isGPUSearchReady()) << "GPU not ready for k=" << k;

        index.knnSearch(query, indices_k, dists_k, k, flann::SearchParams(128));

        // Verify results are valid
        for (size_t i = 0; i < query.rows; ++i) {
            for (int j = 0; j < k; ++j) {
                EXPECT_GE(indices_k[i][j], 0u) << "Invalid index at k=" << k << ", query=" << i << ", pos=" << j;
                EXPECT_LT(indices_k[i][j], data.rows) << "Index out of range at k=" << k;
            }
        }

        // Compute and verify precision
        float precision = compute_precision(gt_indices_k, indices_k);
        float threshold = (k <= 10) ? 0.90f : 0.80f;  // Higher threshold for smaller k
        EXPECT_GE(precision, threshold) << "k=" << k << " precision below threshold";
        printf("k=%3d: precision=%.2f%%\n", k, precision * 100);

        delete[] indices_k.ptr();
        delete[] dists_k.ptr();
        delete[] gt_indices_k.ptr();
        delete[] gt_dists_k.ptr();
    }
}


// ============================================================================
// Error Handling Tests
// ============================================================================

/**
 * Test: Unsupported k-value throws exception
 * K-Means CUDA supports: 1, 2, 4, 5, 7, 8, 10, 16, 20, 32, 50, 64, 100
 */
TEST_F(KMeansCUDA_SIFT10K, TestInvalidKValueThrows)
{
    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // k=3 is not supported by cooperative kernel
    EXPECT_THROW(index.buildCUDAKnnSearch(3, flann::SearchParams(128)), FLANNException);

    // k=15 is not supported
    EXPECT_THROW(index.buildCUDAKnnSearch(15, flann::SearchParams(128)), FLANNException);
}

/**
 * Test: GPU setup before buildIndex throws exception
 */
TEST_F(KMeansCUDA_SIFT10K, TestGPUSetupBeforeBuildThrows)
{
    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));

    // Should throw because buildIndex() not called
    EXPECT_THROW(index.buildCUDAKnnSearch(5, flann::SearchParams(128)), FLANNException);
}

/**
 * Test: Supported k-value succeeds
 */
TEST_F(KMeansCUDA_SIFT10K, TestValidKValueSucceeds)
{
    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // k=5 is supported
    EXPECT_NO_THROW(index.buildCUDAKnnSearch(5, flann::SearchParams(128)));
    EXPECT_TRUE(index.isGPUSearchReady());
}

/**
 * Test: Boundary k-values (k=1 minimum, k=100 maximum)
 * Verify that k=1 and k=100 produce correct results with precision verification
 */
TEST_F(KMeansCUDA_SIFT10K, TestBoundaryKValues)
{
    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // Compute ground truth using linear index
    flann::Index<flann::L2<float>> linear_index(data, flann::LinearIndexParams());
    linear_index.buildIndex();

    // Test k=1 (minimum) with precision verification
    {
        const size_t k = 1;
        flann::Matrix<size_t> indices_k1(new size_t[query.rows * k], query.rows, k);
        flann::Matrix<float> dists_k1(new float[query.rows * k], query.rows, k);
        flann::Matrix<size_t> gt_indices_k1(new size_t[query.rows * k], query.rows, k);
        flann::Matrix<float> gt_dists_k1(new float[query.rows * k], query.rows, k);

        // Compute ground truth
        linear_index.knnSearch(query, gt_indices_k1, gt_dists_k1, k, flann::SearchParams(-1));

        // Run CUDA search
        EXPECT_NO_THROW(index.buildCUDAKnnSearch(k, flann::SearchParams(128)));
        EXPECT_TRUE(index.isGPUSearchReady());
        index.knnSearch(query, indices_k1, dists_k1, k, flann::SearchParams(128));

        // Verify validity
        for (size_t i = 0; i < query.rows; ++i) {
            EXPECT_GE(indices_k1[i][0], 0u);
            EXPECT_LT(indices_k1[i][0], data.rows);
            EXPECT_GE(dists_k1[i][0], 0.0f);
        }

        // Verify precision (k=1 should have very high precision)
        float precision = compute_precision(gt_indices_k1, indices_k1);
        EXPECT_GE(precision, 0.95f) << "k=1 precision " << precision << " below 95% threshold";
        printf("k=1 precision: %.2f%%\n", precision * 100);

        delete[] indices_k1.ptr();
        delete[] dists_k1.ptr();
        delete[] gt_indices_k1.ptr();
        delete[] gt_dists_k1.ptr();
    }

    // Test k=100 (maximum supported) with precision verification
    {
        const size_t k = 100;
        flann::Matrix<size_t> indices_k100(new size_t[query.rows * k], query.rows, k);
        flann::Matrix<float> dists_k100(new float[query.rows * k], query.rows, k);
        flann::Matrix<size_t> gt_indices_k100(new size_t[query.rows * k], query.rows, k);
        flann::Matrix<float> gt_dists_k100(new float[query.rows * k], query.rows, k);

        // Compute ground truth
        linear_index.knnSearch(query, gt_indices_k100, gt_dists_k100, k, flann::SearchParams(-1));

        // Run CUDA search
        EXPECT_NO_THROW(index.buildCUDAKnnSearch(k, flann::SearchParams(256)));
        EXPECT_TRUE(index.isGPUSearchReady());
        index.knnSearch(query, indices_k100, dists_k100, k, flann::SearchParams(256));

        // Verify validity and sorting
        for (size_t i = 0; i < query.rows; ++i) {
            for (size_t j = 0; j < k; ++j) {
                EXPECT_GE(indices_k100[i][j], 0u);
                EXPECT_LT(indices_k100[i][j], data.rows);
            }
            for (size_t j = 1; j < k; ++j) {
                EXPECT_GE(dists_k100[i][j], dists_k100[i][j-1])
                    << "Distances not sorted at query " << i << ", position " << j;
            }
        }

        // Verify precision (k=100 may have slightly lower precision)
        float precision = compute_precision(gt_indices_k100, indices_k100);
        EXPECT_GE(precision, 0.85f) << "k=100 precision " << precision << " below 85% threshold";
        printf("k=100 precision: %.2f%%\n", precision * 100);

        delete[] indices_k100.ptr();
        delete[] dists_k100.ptr();
        delete[] gt_indices_k100.ptr();
        delete[] gt_dists_k100.ptr();
    }
}


// ============================================================================
// Additional Edge Case Tests
// ============================================================================

/**
 * Test: k=0 should throw exception
 * Zero neighbors is an invalid request
 */
TEST_F(KMeansCUDA_SIFT10K, TestZeroKThrows)
{
    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    EXPECT_THROW(index.buildCUDAKnnSearch(0, flann::SearchParams(128)), FLANNException);
}

/**
 * Test: k exceeding dataset size should be handled gracefully
 * Should either throw or return all available points
 */
TEST_F(KMeansCUDA_SIFT10K, TestKExceedsDatasetSize)
{
    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // k > dataset.rows should be handled (either throw or clamp)
    size_t excessive_k = data.rows + 100;

    // Expecting this to throw for unsupported k value (not in supported list)
    EXPECT_THROW(index.buildCUDAKnnSearch(excessive_k, flann::SearchParams(128)), FLANNException);
}

/**
 * Test: Multiple buildCUDAKnnSearch calls should work correctly
 * Each call should properly reset GPU state
 */
TEST_F(KMeansCUDA_SIFT10K, TestMultipleGPUSetup)
{
    flann::Index<flann::L2<float>> index(data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // First setup with k=5
    EXPECT_NO_THROW(index.buildCUDAKnnSearch(5, flann::SearchParams(128)));
    EXPECT_TRUE(index.isGPUSearchReady());

    // Run a search
    flann::Matrix<size_t> indices1(new size_t[query.rows * 5], query.rows, 5);
    flann::Matrix<float> dists1(new float[query.rows * 5], query.rows, 5);
    index.knnSearch(query, indices1, dists1, 5, flann::SearchParams(128));

    // Setup with different k=10
    EXPECT_NO_THROW(index.buildCUDAKnnSearch(10, flann::SearchParams(128)));
    EXPECT_TRUE(index.isGPUSearchReady());

    // Run another search
    flann::Matrix<size_t> indices2(new size_t[query.rows * 10], query.rows, 10);
    flann::Matrix<float> dists2(new float[query.rows * 10], query.rows, 10);
    index.knnSearch(query, indices2, dists2, 10, flann::SearchParams(128));

    // Verify results are valid
    for (size_t i = 0; i < query.rows; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_GE(indices2[i][j], 0u);
            EXPECT_LT(indices2[i][j], data.rows);
        }
    }

    delete[] indices1.ptr();
    delete[] dists1.ptr();
    delete[] indices2.ptr();
    delete[] dists2.ptr();
}

/**
 * Test: Search with k that equals dataset size
 * Edge case where we want all neighbors
 */
TEST_F(KMeansCUDA_SIFT10K, TestKEqualsDatasetSize)
{
    // Create a small dataset for this test
    const size_t small_size = 100;
    flann::Matrix<float> small_data(new float[small_size * 128], small_size, 128);
    for (size_t i = 0; i < small_size * 128; ++i) {
        small_data.ptr()[i] = data.ptr()[i];
    }

    flann::Index<flann::L2<float>> index(small_data, flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // k=100 is supported and equals small dataset size
    EXPECT_NO_THROW(index.buildCUDAKnnSearch(100, flann::SearchParams(128)));
    EXPECT_TRUE(index.isGPUSearchReady());

    // Run search - should return all points for each query
    flann::Matrix<size_t> indices(new size_t[query.rows * 100], query.rows, 100);
    flann::Matrix<float> dists(new float[query.rows * 100], query.rows, 100);
    index.knnSearch(query, indices, dists, 100, flann::SearchParams(128));

    // Verify all 100 results are valid indices
    for (size_t i = 0; i < query.rows; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            EXPECT_GE(indices[i][j], 0u);
            EXPECT_LT(indices[i][j], small_size);
        }
    }

    delete[] small_data.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
}

// ============================================================================
// GPU Save/Load Tests
// ============================================================================

/**
 * Test: GPU-Optimized Save/Load
 * Save in GPU format, reload, and verify search results match
 */
TEST_F(KMeansCUDA_SIFT10K, TestGPUSaveLoad)
{
    flann::seed_random(0);
    flann::Index<flann::L2<float>> index(data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));

    index.buildIndex();
    index.buildCUDAKnnSearch(knn, flann::SearchParams(128));

    // Search before save
    flann::Matrix<size_t> indices1(new size_t[query.rows * knn], query.rows, knn);
    flann::Matrix<float> dists1(new float[query.rows * knn], query.rows, knn);
    index.knnSearch(query, indices1, dists1, knn, flann::SearchParams(128));

    // Verify GPU format is available
    EXPECT_TRUE(index.hasGPUFormat());

    // Save in GPU format
    index.save("test_kmeans_gpu_saved.idx");

    // Load from GPU format
    flann::Index<flann::L2<float>> index2(data,
        flann::SavedIndexParams("test_kmeans_gpu_saved.idx"));

    // Need to warm up GPU for loaded index
    index2.buildCUDAKnnSearch(knn, flann::SearchParams(128));

    // Search after load
    flann::Matrix<size_t> indices2(new size_t[query.rows * knn], query.rows, knn);
    flann::Matrix<float> dists2(new float[query.rows * knn], query.rows, knn);
    index2.knnSearch(query, indices2, dists2, knn, flann::SearchParams(128));

    // Verify results match exactly
    for (size_t i = 0; i < query.rows; ++i) {
        for (size_t j = 0; j < knn; ++j) {
            EXPECT_EQ(indices1[i][j], indices2[i][j])
                << "Mismatch at query " << i << " neighbor " << j;
            EXPECT_NEAR(dists1[i][j], dists2[i][j], 0.001)
                << "Distance mismatch at query " << i << " neighbor " << j;
        }
    }

    delete[] indices1.ptr();
    delete[] dists1.ptr();
    delete[] indices2.ptr();
    delete[] dists2.ptr();

    // Cleanup test file
    remove("test_kmeans_gpu_saved.idx");
}

/**
 * Test: Convert to GPU Format
 * Verify convertToGPUFormat() discards CPU tree and enables GPU search
 */
TEST_F(KMeansCUDA_SIFT10K, TestConvertToGPUFormat)
{
    flann::seed_random(0);
    flann::Index<flann::L2<float>> index(data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));

    index.buildIndex();
    index.buildCUDAKnnSearch(knn, flann::SearchParams(128));

    // Before conversion
    EXPECT_TRUE(index.hasGPUFormat());
    EXPECT_TRUE(index.isGPUSearchReady());

    // Convert to GPU format (discards CPU tree)
    index.convertToGPUFormat();

    // After conversion - GPU format still available
    EXPECT_TRUE(index.hasGPUFormat());

    // Save and reload
    index.save("test_kmeans_converted.idx");

    flann::Index<flann::L2<float>> index2(data,
        flann::SavedIndexParams("test_kmeans_converted.idx"));
    index2.buildCUDAKnnSearch(knn, flann::SearchParams(128));

    // Search works after loading GPU format
    flann::Matrix<size_t> indices(new size_t[query.rows * knn], query.rows, knn);
    flann::Matrix<float> dists(new float[query.rows * knn], query.rows, knn);
    index2.knnSearch(query, indices, dists, knn, flann::SearchParams(128));

    // Verify reasonable precision
    float precision = compute_precision(gt_indices, indices);
    EXPECT_GE(precision, 0.95)
        << "Precision too low after GPU save/load: " << precision;

    delete[] indices.ptr();
    delete[] dists.ptr();
    remove("test_kmeans_converted.idx");
}

/**
 * Test: Format Auto-Detection
 * Verify both CPU and GPU formats can be loaded transparently
 */
TEST_F(KMeansCUDA_SIFT10K, TestFormatDetection)
{
    flann::seed_random(0);

    // Create and save CPU format (without GPU init)
    {
        flann::Index<flann::L2<float>> index(data,
            flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
        index.buildIndex();
        // Don't call buildCUDAKnnSearch - saves in CPU format
        EXPECT_FALSE(index.hasGPUFormat());
        index.save("test_kmeans_cpu_format.idx");
    }

    // Create and save GPU format
    {
        flann::Index<flann::L2<float>> index(data,
            flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
        index.buildIndex();
        index.buildCUDAKnnSearch(knn, flann::SearchParams(128));
        EXPECT_TRUE(index.hasGPUFormat());
        index.save("test_kmeans_gpu_format.idx");
    }

    // Load CPU format - should work
    {
        flann::Index<flann::L2<float>> index(data,
            flann::SavedIndexParams("test_kmeans_cpu_format.idx"));
        EXPECT_EQ(index.size(), data.rows);
    }

    // Load GPU format - should work
    {
        flann::Index<flann::L2<float>> index(data,
            flann::SavedIndexParams("test_kmeans_gpu_format.idx"));
        EXPECT_EQ(index.size(), data.rows);
    }

    remove("test_kmeans_cpu_format.idx");
    remove("test_kmeans_gpu_format.idx");
}

// ============================================================================
// GPU Save/Load Gap Coverage Tests
// Additional tests to ensure GPU save/load is at least as comprehensive as CPU
// ============================================================================

/**
 * Test: GPU Format Save After Remove
 * Verify GPU format save/load preserves removed point state
 * (Current TestRemove uses CPU format save)
 */
TEST_F(KMeansCUDA_SIFT100K, TestGPUFormatSaveAfterRemove)
{
    flann::seed_random(0);
    const size_t test_knn = 5;
    flann::Matrix<size_t> indices1(new size_t[query.rows*test_knn], query.rows, test_knn);
    flann::Matrix<float> dists1(new float[query.rows*test_knn], query.rows, test_knn);

    // Build index
    flann::Index<flann::L2<float>> index(data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();
    index.buildCUDAKnnSearch(test_knn, flann::SearchParams(128));

    // Search to find points to remove
    index.knnSearch(query, indices1, dists1, test_knn, flann::SearchParams(128));

    // Remove ~50% of found neighbors (use set to avoid duplicates)
    std::set<size_t> removed_points;
    for (size_t i = 0; i < indices1.rows && removed_points.size() < data.rows / 2; ++i) {
        for (size_t j = 0; j < indices1.cols && removed_points.size() < data.rows / 2; ++j) {
            size_t pt = indices1[i][j];
            if (removed_points.find(pt) == removed_points.end()) {
                index.removePoint(pt);
                removed_points.insert(pt);
            }
        }
    }
    printf("Removed %zu points\n", removed_points.size());

    // Re-init GPU and convert to GPU format
    index.buildCUDAKnnSearch(test_knn, flann::SearchParams(128));
    ASSERT_TRUE(index.hasGPUFormat());
    index.convertToGPUFormat();

    // Save GPU format
    const char* filename = "test_gpu_format_after_remove_kmeans.idx";
    index.save(filename);

    // Load with SavedIndexParams and init GPU
    flann::Index<flann::L2<float>> index2(data, flann::SavedIndexParams(filename));
    index2.buildCUDAKnnSearch(test_knn, flann::SearchParams(128));

    // Search after load
    flann::Matrix<size_t> indices2(new size_t[query.rows*test_knn], query.rows, test_knn);
    flann::Matrix<float> dists2(new float[query.rows*test_knn], query.rows, test_knn);
    index2.knnSearch(query, indices2, dists2, test_knn, flann::SearchParams(128));

    // Verify removed points don't appear in results
    int removed_found = 0;
    for (size_t i = 0; i < indices2.rows; ++i) {
        for (size_t j = 0; j < indices2.cols; ++j) {
            if (removed_points.find(indices2[i][j]) != removed_points.end()) {
                removed_found++;
            }
        }
    }
    EXPECT_EQ(removed_found, 0) << "Found " << removed_found << " removed points in results";

    // Cleanup
    remove(filename);
    delete[] indices1.ptr();
    delete[] dists1.ptr();
    delete[] indices2.ptr();
    delete[] dists2.ptr();
}

/**
 * Test: CPU Format Load With GPU Acceleration
 * Verify CPU-saved indices can be GPU-accelerated after load
 */
TEST_F(KMeansCUDA_SIFT100K, TestCPUFormatLoadWithGPU)
{
    flann::seed_random(0);
    const size_t test_knn = 5;

    // Build index WITHOUT GPU init (saves CPU format)
    flann::Index<flann::L2<float>> index(data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();
    EXPECT_FALSE(index.hasGPUFormat()) << "Should not have GPU format before buildCUDAKnnSearch";

    // CPU baseline search
    flann::Matrix<size_t> indices_cpu(new size_t[query.rows*test_knn], query.rows, test_knn);
    flann::Matrix<float> dists_cpu(new float[query.rows*test_knn], query.rows, test_knn);
    index.knnSearch(query, indices_cpu, dists_cpu, test_knn, flann::SearchParams(128));
    float cpu_precision = compute_precision(gt_indices, indices_cpu);
    printf("CPU baseline precision: %.2f%%\n", cpu_precision * 100);

    // Save CPU format
    const char* filename = "test_cpu_format_load_with_gpu_kmeans.idx";
    index.save(filename);

    // Load and init GPU on CPU-format file
    flann::Index<flann::L2<float>> index2(data, flann::SavedIndexParams(filename));
    EXPECT_FALSE(index2.hasGPUFormat()) << "Loaded CPU format should not have GPU format";
    index2.buildCUDAKnnSearch(test_knn, flann::SearchParams(128));
    EXPECT_TRUE(index2.isGPUSearchReady()) << "GPU search should be ready after buildCUDAKnnSearch";

    // GPU search after loading CPU format
    flann::Matrix<size_t> indices_gpu(new size_t[query.rows*test_knn], query.rows, test_knn);
    flann::Matrix<float> dists_gpu(new float[query.rows*test_knn], query.rows, test_knn);
    index2.knnSearch(query, indices_gpu, dists_gpu, test_knn, flann::SearchParams(128));
    float gpu_precision = compute_precision(gt_indices, indices_gpu);
    printf("GPU precision after CPU format load: %.2f%%\n", gpu_precision * 100);

    // GPU precision should be close to CPU (within 2% for L2 distance)
    EXPECT_GE(gpu_precision, cpu_precision - 0.02f)
        << "GPU precision " << gpu_precision << " too low vs CPU " << cpu_precision;
    EXPECT_GE(gpu_precision, 0.93f) << "GPU precision below threshold";

    // Cleanup
    remove(filename);
    delete[] indices_cpu.ptr();
    delete[] dists_cpu.ptr();
    delete[] indices_gpu.ptr();
    delete[] dists_gpu.ptr();
}

/**
 * Test: GPU Format Save After Incremental Add
 * Ensure GPU format correctly captures incrementally-added data
 */
TEST_F(KMeansCUDA_SIFT100K, TestGPUFormatSaveAfterIncrementalAdd)
{
    flann::seed_random(0);
    const size_t test_knn = 5;

    // Split data 50/50
    size_t size1 = data.rows / 2;
    size_t size2 = data.rows - size1;
    flann::Matrix<float> data1(data[0], size1, data.cols);
    flann::Matrix<float> data2(data[size1], size2, data.cols);

    // Build with 50% data
    flann::Index<flann::L2<float>> index(data1,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();
    index.buildCUDAKnnSearch(test_knn, flann::SearchParams(128));
    EXPECT_EQ(index.size(), size1);

    // Add remaining 50%
    index.addPoints(data2, 2.0f);
    EXPECT_EQ(index.size(), data.rows);
    printf("Added %zu points (total now %zu)\n", size2, index.size());

    // Re-init GPU after add
    index.buildCUDAKnnSearch(test_knn, flann::SearchParams(128));
    EXPECT_TRUE(index.hasGPUFormat());

    // Convert and save GPU format
    index.convertToGPUFormat();
    const char* filename = "test_gpu_format_after_add_kmeans.idx";
    index.save(filename);

    // Load and init GPU
    flann::Index<flann::L2<float>> index2(data, flann::SavedIndexParams(filename));
    index2.buildCUDAKnnSearch(test_knn, flann::SearchParams(128));
    EXPECT_EQ(index2.size(), data.rows);

    // Search and verify precision
    flann::Matrix<size_t> indices(new size_t[query.rows*test_knn], query.rows, test_knn);
    flann::Matrix<float> dists(new float[query.rows*test_knn], query.rows, test_knn);
    index2.knnSearch(query, indices, dists, test_knn, flann::SearchParams(128));
    float precision = compute_precision(gt_indices, indices);
    printf("Precision after incremental add + GPU save/load: %.2f%%\n", precision * 100);
    EXPECT_GE(precision, 0.93f) << "Precision after load: " << precision;

    // Cleanup
    remove(filename);
    delete[] indices.ptr();
    delete[] dists.ptr();
}

// ============================================================================
// Comprehensive E2E Tests
// ============================================================================

/**
 * Helper function to get file size in bytes
 */
inline size_t getFileSize(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fclose(f);
    return size;
}

/**
 * Test: End-to-End GPU Save/Load Cycle
 * Full workflow: CPU baseline → GPU → convert → save → load → verify
 */
TEST_F(KMeansCUDA_SIFT10K, TestEndToEndGPUSaveLoadCycle)
{
    flann::seed_random(0);

    // Build index
    flann::Index<flann::L2<float>> index(data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    // Stage 1: CPU baseline search
    // Note: CPU search uses different algorithm than GPU, so precision may differ
    flann::Matrix<size_t> cpu_indices(new size_t[query.rows * knn], query.rows, knn);
    flann::Matrix<float> cpu_dists(new float[query.rows * knn], query.rows, knn);
    index.knnSearch(query, cpu_indices, cpu_dists, knn, flann::SearchParams(128));
    float cpu_precision = compute_precision(gt_indices, cpu_indices);
    EXPECT_GE(cpu_precision, 0.70f) << "CPU baseline precision too low";

    // Stage 2: GPU search
    index.buildCUDAKnnSearch(knn, flann::SearchParams(128));
    flann::Matrix<size_t> gpu_indices(new size_t[query.rows * knn], query.rows, knn);
    flann::Matrix<float> gpu_dists(new float[query.rows * knn], query.rows, knn);
    index.knnSearch(query, gpu_indices, gpu_dists, knn, flann::SearchParams(128));
    float gpu_precision = compute_precision(gt_indices, gpu_indices);
    EXPECT_GE(gpu_precision, cpu_precision - 0.02f)
        << "GPU precision dropped vs CPU: " << gpu_precision << " vs " << cpu_precision;

    // Stage 3: Convert to GPU-only format and save
    index.convertToGPUFormat();
    EXPECT_TRUE(index.hasGPUFormat());
    const char* filename = "test_e2e_kmeans.idx";
    index.save(filename);
    size_t file_size = getFileSize(filename);
    EXPECT_GT(file_size, 0u) << "File not created";

    // Stage 4: Load and verify
    flann::Index<flann::L2<float>> loaded(data, flann::SavedIndexParams(filename));
    loaded.buildCUDAKnnSearch(knn, flann::SearchParams(128));

    flann::Matrix<size_t> loaded_indices(new size_t[query.rows * knn], query.rows, knn);
    flann::Matrix<float> loaded_dists(new float[query.rows * knn], query.rows, knn);
    loaded.knnSearch(query, loaded_indices, loaded_dists, knn, flann::SearchParams(128));
    float loaded_precision = compute_precision(gt_indices, loaded_indices);

    // Final assertions
    EXPECT_GE(loaded_precision, cpu_precision - 0.02f)
        << "Loaded precision dropped vs CPU baseline: " << loaded_precision << " vs " << cpu_precision;
    EXPECT_GE(loaded_precision, 0.95f)
        << "Loaded index precision too low: " << loaded_precision;

    // Results after load should match results before save (exact)
    for (size_t i = 0; i < query.rows; ++i) {
        for (size_t j = 0; j < knn; ++j) {
            EXPECT_EQ(gpu_indices[i][j], loaded_indices[i][j])
                << "Result mismatch at query " << i << " neighbor " << j;
        }
    }

    // Cleanup
    delete[] cpu_indices.ptr();
    delete[] cpu_dists.ptr();
    delete[] gpu_indices.ptr();
    delete[] gpu_dists.ptr();
    delete[] loaded_indices.ptr();
    delete[] loaded_dists.ptr();
    remove(filename);

    std::cout << "E2E Precision Summary: CPU=" << cpu_precision
              << " GPU=" << gpu_precision
              << " Loaded=" << loaded_precision
              << " FileSize=" << file_size << " bytes\n";
}

/**
 * Test: GPU Format File Size Efficiency
 * Compare GPU format against CPU format WITH save_dataset=true (apples-to-apples)
 * GPU format always saves the dataset since CPU tree is discarded after conversion.
 */
TEST_F(KMeansCUDA_SIFT10K, TestGPUFormatFileSizeEfficiency)
{
    flann::seed_random(0);

    // Build index with save_dataset=true for fair comparison
    flann::KMeansCUDAIndexParams params(32, 11, FLANN_CENTERS_RANDOM, 0.2);
    params["save_dataset"] = true;  // Enable dataset saving for CPU format
    flann::Index<flann::L2<float>> index(data, params);
    index.buildIndex();

    // Save CPU format WITH dataset
    const char* cpu_file = "test_filesize_cpu_kmeans.idx";
    index.save(cpu_file);
    size_t cpu_size = getFileSize(cpu_file);

    // Convert and save GPU format
    index.buildCUDAKnnSearch(knn, flann::SearchParams(128));
    index.convertToGPUFormat();
    const char* gpu_file = "test_filesize_gpu_kmeans.idx";
    index.save(gpu_file);
    size_t gpu_size = getFileSize(gpu_file);

    // Raw data size calculation
    size_t raw_dataset_bytes = data.rows * data.cols * sizeof(float);
    size_t padded_veclen = ((data.cols + 3) / 4) * 4;
    size_t padded_dataset_bytes = data.rows * padded_veclen * sizeof(float);

    std::cout << "File Size Analysis (KMeans, save_dataset=true):\n"
              << "  CPU format (with dataset): " << cpu_size << " bytes\n"
              << "  GPU format:                " << gpu_size << " bytes\n"
              << "  Raw dataset:               " << raw_dataset_bytes << " bytes\n"
              << "  Padded dataset:            " << padded_dataset_bytes << " bytes\n"
              << "  GPU/CPU ratio:             " << (float)gpu_size / cpu_size << "x\n";

    // GPU format should not be significantly larger than CPU format with dataset
    // Allow up to 2x size due to padding and different tree representation
    EXPECT_GT(gpu_size, 0u) << "GPU file is empty";
    EXPECT_LE(gpu_size, cpu_size * 2)
        << "GPU format significantly larger than CPU format with dataset";

    remove(cpu_file);
    remove(gpu_file);
}

/**
 * Test: GPU Format Load Performance
 * Verify GPU format loading is efficient
 */
TEST_F(KMeansCUDA_SIFT10K, TestGPUFormatLoadPerformance)
{
    flann::seed_random(0);
    const int num_runs = 3;

    // Setup: build and save both formats
    flann::Index<flann::L2<float>> index(data,
        flann::KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();

    const char* cpu_file = "test_loadperf_cpu_kmeans.idx";
    index.save(cpu_file);

    index.buildCUDAKnnSearch(knn, flann::SearchParams(128));
    const char* gpu_file = "test_loadperf_gpu_kmeans.idx";
    index.save(gpu_file);

    // Benchmark CPU format load
    double cpu_load_ms = 0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        flann::Index<flann::L2<float>> loaded(data, flann::SavedIndexParams(cpu_file));
        auto end = std::chrono::high_resolution_clock::now();
        cpu_load_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    cpu_load_ms /= num_runs;

    // Benchmark GPU format load + warmup
    double gpu_load_ms = 0, gpu_warmup_ms = 0;
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        flann::Index<flann::L2<float>> loaded(data, flann::SavedIndexParams(gpu_file));
        auto mid = std::chrono::high_resolution_clock::now();
        loaded.buildCUDAKnnSearch(knn, flann::SearchParams(128));
        auto end = std::chrono::high_resolution_clock::now();
        gpu_load_ms += std::chrono::duration<double, std::milli>(mid - start).count();
        gpu_warmup_ms += std::chrono::duration<double, std::milli>(end - mid).count();
    }
    gpu_load_ms /= num_runs;
    gpu_warmup_ms /= num_runs;

    std::cout << "Load Performance (KMeans, avg of " << num_runs << " runs):\n"
              << "  CPU load:     " << cpu_load_ms << " ms\n"
              << "  GPU load:     " << gpu_load_ms << " ms\n"
              << "  GPU warmup:   " << gpu_warmup_ms << " ms\n"
              << "  GPU total:    " << (gpu_load_ms + gpu_warmup_ms) << " ms\n";

    // GPU load includes data upload to GPU, so it may be slower than CPU load
    // The key metric is that it completes in reasonable time (<500ms for small datasets)
    // For larger datasets, GPU load saves time vs CPU tree reconstruction
    EXPECT_LT(gpu_load_ms, 500.0)
        << "GPU load too slow: " << gpu_load_ms << " ms";
    EXPECT_LT(gpu_warmup_ms, 500.0)
        << "GPU warmup too slow: " << gpu_warmup_ms << " ms";

    remove(cpu_file);
    remove(gpu_file);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
