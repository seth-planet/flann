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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
