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

#define FLANN_USE_CUDA
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
        0.95,  // 95% recall (high precision requirement)
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
        0.92,  // 92% recall with higher checks
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
        0.95,  // 95% recall (high precision requirement)
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
        0.95,  // 95% recall (high precision requirement)
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
        0.95,  // 95% recall (high precision requirement)
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
        0.95,  // 95% recall (high precision requirement)
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
        0.95,  // 95% recall (high precision requirement)
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
        0.95,  // 95% recall (high precision requirement)
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
        0.95,  // 95% recall (high precision requirement)
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
        0.95,  // 95% recall (high precision requirement)
        gt_indices
    );
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
