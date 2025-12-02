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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
