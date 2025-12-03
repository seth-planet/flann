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
		flann::load_from_file(data, "brief100K.h5", "dataset");
		flann::load_from_file(query, "brief100K.h5", "query");
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
 * Expected: ≥94% recall@3 (Hierarchical CUDA with Hamming distance)
 * Note: Actual precision typically ~97%, threshold set conservatively
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSearch)
{
	TestSearch<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.94, gt_indices, gt_dists);
}

/**
 * Test 2: High Precision Search
 * Same parameters but validates consistent high precision
 * Expected: ≥94% recall@3
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSearch2)
{
	TestSearch2<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.94, gt_indices, gt_dists);
}

/**
 * Test 3: Incremental Point Addition
 * Add points incrementally and verify search still works
 * Expected: ≥94% recall@3 (verified actual: ~97.1%)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestAddIncremental)
{
	TestAddIncremental<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.94, gt_indices, gt_dists);
}

/**
 * Test 4: Incremental Addition with Rebuild
 * Test rebuild threshold triggering
 * Expected: ≥93% recall@3 (verified actual: ~96.3%)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestAddIncremental2)
{
	TestAddIncremental2<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.93, gt_indices, gt_dists);
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
 * Expected: ≥94% recall@3 (verified actual: ~97.17%)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSave)
{
	TestSave<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.94, gt_indices, gt_dists);
}

/**
 * Test 7: Copy Constructor
 * Verify copy constructor creates independent index
 * Expected: ≥94% recall@3 (verified actual: ~97.17%)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestCopy)
{
	TestCopy<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.94, gt_indices, gt_dists);
}

/**
 * Test 8: Move Constructor
 * Verify move constructor transfers ownership correctly
 * Expected: ≥94% recall@3 (verified actual: ~97.17%)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestCopy2)
{
	TestCopy2<flann::cuda::HierarchicalCUDAIndex<Distance> >(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.94, gt_indices, gt_dists);
}


// ============================================================================
// K-Value Coverage Tests
// Comprehensive testing of all supported k-values
// ============================================================================

/**
 * Test all supported k-values for Hierarchical CUDA
 * Hierarchical CUDA supports: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128
 * Performance tiers (by memory alignment):
 *   Tier 1 (optimal): 16, 32, 64, 128 - int4 vectorization + cache-aligned
 *   Tier 2 (good): 8, 24 - int4 vectorization
 *   Tier 3 (vectorized): 4, 12, 20 - int4 vectorization
 *   Tier 4 (compatible): 1, 2, 3, 5, 10, 50, 100 - Scalar fallback
 */
TEST_F(HierarchicalCUDA_Brief100K, TestAllSupportedKValues)
{
	const std::vector<int> k_values = {1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128};

	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// Build ground truth with linear index
	flann::Index<Distance> linear_index(data, flann::LinearIndexParams());
	linear_index.buildIndex();

	printf("\n=== K-Value Coverage Test Results (Hierarchical CUDA) ===\n");

	for (int k : k_values) {
		// Skip k values larger than dataset if needed
		if (static_cast<size_t>(k) > data.rows) continue;

		flann::Matrix<size_t> indices_k(new size_t[query.rows * k], query.rows, k);
		flann::Matrix<DistanceType> dists_k(new DistanceType[query.rows * k], query.rows, k);
		flann::Matrix<size_t> gt_indices_k(new size_t[query.rows * k], query.rows, k);
		flann::Matrix<DistanceType> gt_dists_k(new DistanceType[query.rows * k], query.rows, k);

		// Compute ground truth
		linear_index.knnSearch(query, gt_indices_k, gt_dists_k, k, flann::SearchParams(-1));

		// Test CUDA search
		ASSERT_NO_THROW(index.buildCUDAKnnSearch(k, flann::SearchParams(2000)))
			<< "Failed to setup GPU for k=" << k;
		ASSERT_TRUE(index.isGPUSearchReady()) << "GPU not ready for k=" << k;

		index.knnSearch(query, indices_k, dists_k, k, flann::SearchParams(2000));

		// Verify results are valid
		for (size_t i = 0; i < query.rows; ++i) {
			for (int j = 0; j < k; ++j) {
				EXPECT_LT(indices_k[i][j], data.rows) << "Index out of range at k=" << k;
			}
		}

		// Compute and verify precision
		float precision = compute_precision(gt_indices_k, indices_k);
		// Hamming distance may have ties, so lower threshold
		float threshold = (k <= 10) ? 0.85f : 0.75f;
		EXPECT_GE(precision, threshold) << "k=" << k << " precision below threshold";

		// Classify performance tier for reporting
		const char* tier;
		if (k == 16 || k == 32 || k == 64 || k == 128) tier = "Tier1";
		else if (k == 8 || k == 24) tier = "Tier2";
		else if (k == 4 || k == 12 || k == 20) tier = "Tier3";
		else tier = "Tier4";

		printf("k=%3d [%s]: precision=%.2f%%\n", k, tier, precision * 100);

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
 * Hierarchical CUDA supports: 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 128
 */
TEST_F(HierarchicalCUDA_Brief100K, TestInvalidKValueThrows)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// k=7 is not in the supported list
	EXPECT_THROW(index.buildCUDAKnnSearch(7, flann::SearchParams(2000)), FLANNException);

	// k=17 is not supported
	EXPECT_THROW(index.buildCUDAKnnSearch(17, flann::SearchParams(2000)), FLANNException);
}

/**
 * Test: GPU setup before buildIndex throws exception
 */
TEST_F(HierarchicalCUDA_Brief100K, TestGPUSetupBeforeBuildThrows)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));

	// Should throw because buildIndex() not called
	EXPECT_THROW(index.buildCUDAKnnSearch(3, flann::SearchParams(2000)), FLANNException);
}

/**
 * Test: Supported k-value succeeds
 */
TEST_F(HierarchicalCUDA_Brief100K, TestValidKValueSucceeds)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// k=3 is supported
	EXPECT_NO_THROW(index.buildCUDAKnnSearch(3, flann::SearchParams(2000)));
	EXPECT_TRUE(index.isGPUSearchReady());
}

/**
 * Test: Boundary k-values (minimum and maximum) with precision verification
 */
TEST_F(HierarchicalCUDA_Brief100K, TestBoundaryKValues)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// Compute ground truth using linear index
	flann::Index<Distance> linear_index(data, flann::LinearIndexParams());
	linear_index.buildIndex();

	// Test k=1 (minimum) with precision verification
	{
		const unsigned int k = 1;
		flann::Matrix<size_t> indices_k1(new size_t[query.rows * k], query.rows, k);
		flann::Matrix<DistanceType> dists_k1(new DistanceType[query.rows * k], query.rows, k);
		flann::Matrix<size_t> gt_indices_k1(new size_t[query.rows * k], query.rows, k);
		flann::Matrix<DistanceType> gt_dists_k1(new DistanceType[query.rows * k], query.rows, k);

		// Compute ground truth
		linear_index.knnSearch(query, gt_indices_k1, gt_dists_k1, k, flann::SearchParams(-1));

		// Run CUDA search
		EXPECT_NO_THROW(index.buildCUDAKnnSearch(k, flann::SearchParams(2000)));
		EXPECT_TRUE(index.isGPUSearchReady());
		index.knnSearch(query, indices_k1, dists_k1, k, flann::SearchParams(2000));

		// Verify validity
		for (size_t i = 0; i < query.rows; ++i) {
			EXPECT_LT(indices_k1[i][0], data.rows);
		}

		// Verify precision (k=1 with Hamming distance may have lower precision due to ties)
		// Note: Binary descriptors have many distance ties, so k=1 precision can vary
		float precision = compute_precision(gt_indices_k1, indices_k1);
		EXPECT_GE(precision, 0.84f) << "k=1 precision " << precision << " below 84% threshold";
		printf("k=1 precision: %.2f%%\n", precision * 100);

		delete[] indices_k1.ptr();
		delete[] dists_k1.ptr();
		delete[] gt_indices_k1.ptr();
		delete[] gt_dists_k1.ptr();
	}

	// Test k=128 (maximum supported) with precision verification
	{
		const unsigned int k = 128;
		flann::Matrix<size_t> indices_k128(new size_t[query.rows * k], query.rows, k);
		flann::Matrix<DistanceType> dists_k128(new DistanceType[query.rows * k], query.rows, k);
		flann::Matrix<size_t> gt_indices_k128(new size_t[query.rows * k], query.rows, k);
		flann::Matrix<DistanceType> gt_dists_k128(new DistanceType[query.rows * k], query.rows, k);

		// Compute ground truth
		linear_index.knnSearch(query, gt_indices_k128, gt_dists_k128, k, flann::SearchParams(-1));

		// Run CUDA search
		EXPECT_NO_THROW(index.buildCUDAKnnSearch(k, flann::SearchParams(2000)));
		EXPECT_TRUE(index.isGPUSearchReady());
		index.knnSearch(query, indices_k128, dists_k128, k, flann::SearchParams(2000));

		// Verify validity and sorting
		for (size_t i = 0; i < query.rows; ++i) {
			for (size_t j = 0; j < k; ++j) {
				EXPECT_LT(indices_k128[i][j], data.rows);
			}
			for (size_t j = 1; j < k; ++j) {
				EXPECT_GE(dists_k128[i][j], dists_k128[i][j-1])
					<< "Distances not sorted at query " << i << ", position " << j;
			}
		}

		// Verify precision (k=128 may have slightly lower precision due to larger k)
		float precision = compute_precision(gt_indices_k128, indices_k128);
		EXPECT_GE(precision, 0.80f) << "k=128 precision " << precision << " below 80% threshold";
		printf("k=128 precision: %.2f%%\n", precision * 100);

		delete[] indices_k128.ptr();
		delete[] dists_k128.ptr();
		delete[] gt_indices_k128.ptr();
		delete[] gt_dists_k128.ptr();
	}
}


// ============================================================================
// Additional Edge Case Tests
// ============================================================================

/**
 * Test: k=0 should throw exception
 * Zero neighbors is an invalid request
 */
TEST_F(HierarchicalCUDA_Brief100K, TestZeroKThrows)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	EXPECT_THROW(index.buildCUDAKnnSearch(0, flann::SearchParams(2000)), FLANNException);
}

/**
 * Test: k exceeding dataset size should be handled gracefully
 * Should throw for unsupported k value
 */
TEST_F(HierarchicalCUDA_Brief100K, TestKExceedsDatasetSize)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// k > dataset.rows should throw (not in supported k-values list)
	size_t excessive_k = data.rows + 100;

	EXPECT_THROW(index.buildCUDAKnnSearch(excessive_k, flann::SearchParams(2000)), FLANNException);
}

/**
 * Test: Multiple buildCUDAKnnSearch calls should work correctly
 * Each call should properly reset GPU state
 */
TEST_F(HierarchicalCUDA_Brief100K, TestMultipleGPUSetup)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// First setup with k=3
	EXPECT_NO_THROW(index.buildCUDAKnnSearch(3, flann::SearchParams(2000)));
	EXPECT_TRUE(index.isGPUSearchReady());

	// Run a search
	flann::Matrix<size_t> indices1(new size_t[query.rows * 3], query.rows, 3);
	flann::Matrix<DistanceType> dists1(new DistanceType[query.rows * 3], query.rows, 3);
	index.knnSearch(query, indices1, dists1, 3, flann::SearchParams(2000));

	// Setup with different k=10
	EXPECT_NO_THROW(index.buildCUDAKnnSearch(10, flann::SearchParams(2000)));
	EXPECT_TRUE(index.isGPUSearchReady());

	// Run another search
	flann::Matrix<size_t> indices2(new size_t[query.rows * 10], query.rows, 10);
	flann::Matrix<DistanceType> dists2(new DistanceType[query.rows * 10], query.rows, 10);
	index.knnSearch(query, indices2, dists2, 10, flann::SearchParams(2000));

	// Verify results are valid
	for (size_t i = 0; i < query.rows; ++i) {
		for (size_t j = 0; j < 10; ++j) {
			EXPECT_LT(indices2[i][j], data.rows);
		}
	}

	delete[] indices1.ptr();
	delete[] dists1.ptr();
	delete[] indices2.ptr();
	delete[] dists2.ptr();
}

/**
 * Test: Search with k that approaches dataset size
 * Uses k=128 which is the maximum supported
 */
TEST_F(HierarchicalCUDA_Brief100K, TestMaximumKValue)
{
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// k=128 is the maximum supported for hierarchical
	EXPECT_NO_THROW(index.buildCUDAKnnSearch(128, flann::SearchParams(2000)));
	EXPECT_TRUE(index.isGPUSearchReady());

	// Run search
	flann::Matrix<size_t> indices(new size_t[query.rows * 128], query.rows, 128);
	flann::Matrix<DistanceType> dists(new DistanceType[query.rows * 128], query.rows, 128);
	index.knnSearch(query, indices, dists, 128, flann::SearchParams(2000));

	// Verify results are valid and sorted
	for (size_t i = 0; i < query.rows; ++i) {
		for (size_t j = 0; j < 128; ++j) {
			EXPECT_LT(indices[i][j], data.rows);
		}
		for (size_t j = 1; j < 128; ++j) {
			EXPECT_GE(dists[i][j], dists[i][j-1])
				<< "Distances not sorted at query " << i << ", position " << j;
		}
	}

	delete[] indices.ptr();
	delete[] dists.ptr();
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
