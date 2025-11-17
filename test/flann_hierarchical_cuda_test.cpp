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
 * Test fixture for Hierarchical CUDA with BRIEF100K dataset
 * Uses binary descriptors (unsigned char) with Hamming distance
 */
class HierarchicalCUDA_Brief100K : public FLANNTestFixture
{
protected:
	typedef flann::Hamming<unsigned char> Distance;
	typedef Distance::ElementType ElementType;
	typedef Distance::ResultType DistanceType;
	flann::Matrix<unsigned char> data;
	flann::Matrix<unsigned char> query;
	flann::Matrix<size_t> gt_indices;
	flann::Matrix<DistanceType> dists;
	flann::Matrix<DistanceType> gt_dists;
	flann::Matrix<size_t> indices;
	unsigned int k_nn_;

	void SetUp()
	{
		k_nn_ = 3;
		printf("Reading test data...");
		fflush(stdout);
		flann::load_from_file(data, "../datasets/brief100K.h5", "dataset");
		flann::load_from_file(query, "../datasets/brief100K.h5", "query");
		printf("done\n");

		// Compute ground truth with linear index (brief100K.h5 has no pre-computed GT)
		flann::Index<Distance> index(data, flann::LinearIndexParams());
		index.buildIndex();

		start_timer("Searching KNN for ground truth...");
		gt_indices = flann::Matrix<size_t>(new size_t[query.rows * k_nn_], query.rows, k_nn_);
		gt_dists = flann::Matrix<DistanceType>(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
		index.knnSearch(query, gt_indices, gt_dists, k_nn_, flann::SearchParams(-1));
		printf("done (%g seconds)\n", stop_timer());

		dists = flann::Matrix<DistanceType>(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
		indices = flann::Matrix<size_t>(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	}

	void TearDown()
	{
		delete[] data.ptr();
		delete[] query.ptr();
		delete[] dists.ptr();
		delete[] indices.ptr();
		delete[] gt_indices.ptr();
		delete[] gt_dists.ptr();
	}
};

/**
 * Test 1: Basic Search
 * Validate basic k-NN search with default parameters
 * Expected: ≥90% recall@3 (Hierarchical CUDA with Hamming distance)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSearch)
{
	TestSearch<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
}

/**
 * Test 2: High Precision Search
 * Same parameters but validates consistent high precision
 * Expected: ≥90% recall@3
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSearch2)
{
	TestSearch2<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
}

/**
 * Test 3: Incremental Point Addition
 * Add points incrementally and verify search still works
 * Expected: ≥87% recall@3
 */
TEST_F(HierarchicalCUDA_Brief100K, TestAddIncremental)
{
	TestAddIncremental<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists);
}

/**
 * Test 4: Incremental Addition with Rebuild
 * Test rebuild threshold triggering
 * Expected: ≥87% recall@3
 */
TEST_F(HierarchicalCUDA_Brief100K, TestAddIncremental2)
{
	TestAddIncremental2<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists);
}

/**
 * Test 5: Point Removal
 * Remove points and verify they don't appear in results
 */
TEST_F(HierarchicalCUDA_Brief100K, TestRemove)
{
	TestRemove<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000));
}

/**
 * Test 6: Save/Load Index
 * Verify index serialization and deserialization
 * Expected: ≥87% recall@3
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSave)
{
	TestSave<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists);
}

/**
 * Test 7: Copy Constructor
 * Verify copy constructor creates independent index
 * Expected: ≥87% recall@3
 */
TEST_F(HierarchicalCUDA_Brief100K, TestCopy)
{
	TestCopy<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists);
}

/**
 * Test 8: Move Constructor
 * Verify move constructor transfers ownership correctly
 * Expected: ≥87% recall@3
 */
TEST_F(HierarchicalCUDA_Brief100K, TestCopy2)
{
	TestCopy2<flann::cuda::HierarchicalCUDAIndex<Distance> >(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.87, gt_indices, gt_dists);
}


int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
