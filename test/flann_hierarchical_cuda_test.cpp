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
 * Expected: ≥90% recall@3 (Hierarchical CUDA with Hamming distance, 128 threads/query)
 * Note: Thread count reduced from 256 to 128 for consistency with K-Means
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSearch)
{
	TestSearch<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
}

/**
 * Test 2: High Precision Search
 * Same parameters but validates consistent high precision
 * Expected: ≥90% recall@3 (128 threads/query)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSearch2)
{
	TestSearch2<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
}

/**
 * Test 3: Incremental Point Addition
 * Add points incrementally and verify search still works
 * Expected: ≥90% recall@3 (128 threads/query)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestAddIncremental)
{
	TestAddIncremental<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
}

/**
 * Test 4: Incremental Addition with Rebuild
 * Test rebuild threshold triggering
 * Expected: ≥88% recall@3 (128 threads/query)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestAddIncremental2)
{
	TestAddIncremental2<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.88, gt_indices, gt_dists);
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
 * Expected: ≥90% recall@3 (128 threads/query)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestSave)
{
	TestSave<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
}

/**
 * Test 7: Copy Constructor
 * Verify copy constructor creates independent index
 * Expected: ≥90% recall@3 (128 threads/query)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestCopy)
{
	TestCopy<Distance>(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
}

/**
 * Test 8: Move Constructor
 * Verify move constructor transfers ownership correctly
 * Expected: ≥90% recall@3 (128 threads/query)
 */
TEST_F(HierarchicalCUDA_Brief100K, TestCopy2)
{
	TestCopy2<flann::cuda::HierarchicalCUDAIndex<Distance> >(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100),
			query, indices, dists, k_nn_, flann::SearchParams(2000), 0.90, gt_indices, gt_dists);
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
		// Hamming distance may have ties, and 128 threads/query reduces precision
		// Larger k values have naturally lower precision
		float threshold = (k <= 10) ? 0.78f : (k <= 64 ? 0.65f : 0.60f);
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
		// Note: Binary descriptors have many distance ties, and 128 threads/query reduces precision
		float precision = compute_precision(gt_indices_k1, indices_k1);
		EXPECT_GE(precision, 0.78f) << "k=1 precision " << precision << " below 78% threshold";
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

		// Verify precision (k=128 may have lower precision due to larger k and 128 threads/query)
		float precision = compute_precision(gt_indices_k128, indices_k128);
		EXPECT_GE(precision, 0.60f) << "k=128 precision " << precision << " below 60% threshold";
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

// ============================================================================
// GPU Save/Load Tests
// ============================================================================

/**
 * Test: GPU-Optimized Save/Load for Hierarchical Index
 * Save in GPU format, reload, and verify search results match
 */
TEST_F(HierarchicalCUDA_Brief100K, TestGPUSaveLoad)
{
	remove("test_hier_gpu_saved.idx");  // Pre-test cleanup for isolation

	flann::Index<Distance> index(data,
		flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));

	index.buildIndex();
	index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));

	// Search before save
	flann::Matrix<size_t> indices1(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	flann::Matrix<DistanceType> dists1(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	index.knnSearch(query, indices1, dists1, k_nn_, flann::SearchParams(2000));

	// Verify GPU format is available
	EXPECT_TRUE(index.hasGPUFormat());

	// Save in GPU format
	index.save("test_hier_gpu_saved.idx");

	// Load from GPU format
	flann::Index<Distance> index2(data,
		flann::SavedIndexParams("test_hier_gpu_saved.idx"));

	// Need to warm up GPU for loaded index
	index2.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));

	// Search after load
	flann::Matrix<size_t> indices2(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	flann::Matrix<DistanceType> dists2(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	index2.knnSearch(query, indices2, dists2, k_nn_, flann::SearchParams(2000));

	// Verify precision is maintained (approximate search may have minor variations)
	// Use precision-based comparison instead of exact match
	float precision1 = compute_precision(gt_indices, indices1);
	float precision2 = compute_precision(gt_indices, indices2);

	// Both should have similar precision (within 5% tolerance)
	EXPECT_GE(precision2, precision1 - 0.05f)
		<< "Precision dropped after GPU save/load: " << precision1 << " -> " << precision2;
	EXPECT_GE(precision2, 0.80f)  // 128 threads/query
		<< "Precision too low after GPU save/load: " << precision2;

	delete[] indices1.ptr();
	delete[] dists1.ptr();
	delete[] indices2.ptr();
	delete[] dists2.ptr();

	// Cleanup test file
	remove("test_hier_gpu_saved.idx");
}

/**
 * Test: Convert to GPU Format for Hierarchical Index
 */
TEST_F(HierarchicalCUDA_Brief100K, TestConvertToGPUFormat)
{
	remove("test_hier_converted.idx");  // Pre-test cleanup for isolation

	flann::Index<Distance> index(data,
		flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));

	index.buildIndex();
	index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));

	// Before conversion
	EXPECT_TRUE(index.hasGPUFormat());

	// Convert to GPU format (discards CPU tree)
	index.convertToGPUFormat();

	// After conversion - GPU format still available
	EXPECT_TRUE(index.hasGPUFormat());

	// Save and reload
	index.save("test_hier_converted.idx");

	flann::Index<Distance> index2(data,
		flann::SavedIndexParams("test_hier_converted.idx"));
	index2.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));

	// Search works after loading GPU format
	flann::Matrix<size_t> indices(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	flann::Matrix<DistanceType> dists(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	index2.knnSearch(query, indices, dists, k_nn_, flann::SearchParams(2000));

	// Verify reasonable precision (128 threads/query)
	float precision = compute_precision(gt_indices, indices);
	EXPECT_GE(precision, 0.80)
		<< "Precision too low after GPU save/load: " << precision;

	delete[] indices.ptr();
	delete[] dists.ptr();
	remove("test_hier_converted.idx");
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
TEST_F(HierarchicalCUDA_Brief100K, TestEndToEndGPUSaveLoadCycle)
{
	// Build index
	flann::Index<Distance> index(data,
		flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	// Stage 1: CPU baseline search
	// Note: CPU search uses different algorithm than GPU, so precision may differ
	flann::Matrix<size_t> cpu_indices(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	flann::Matrix<DistanceType> cpu_dists(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	index.knnSearch(query, cpu_indices, cpu_dists, k_nn_, flann::SearchParams(2000));
	float cpu_precision = compute_precision(gt_indices, cpu_indices);
	EXPECT_GE(cpu_precision, 0.70f) << "CPU baseline precision too low";

	// Stage 2: GPU search
	index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
	flann::Matrix<size_t> gpu_indices(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	flann::Matrix<DistanceType> gpu_dists(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	index.knnSearch(query, gpu_indices, gpu_dists, k_nn_, flann::SearchParams(2000));
	float gpu_precision = compute_precision(gt_indices, gpu_indices);
	EXPECT_GE(gpu_precision, cpu_precision - 0.05f)
		<< "GPU precision dropped vs CPU: " << gpu_precision << " vs " << cpu_precision;

	// Stage 3: Convert to GPU-only format and save
	index.convertToGPUFormat();
	EXPECT_TRUE(index.hasGPUFormat());
	const char* filename = "test_e2e_hier.idx";
	index.save(filename);
	size_t file_size = getFileSize(filename);
	EXPECT_GT(file_size, 0u) << "File not created";

	// Stage 4: Load and verify
	flann::Index<Distance> loaded(data, flann::SavedIndexParams(filename));
	loaded.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));

	flann::Matrix<size_t> loaded_indices(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	flann::Matrix<DistanceType> loaded_dists(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	loaded.knnSearch(query, loaded_indices, loaded_dists, k_nn_, flann::SearchParams(2000));
	float loaded_precision = compute_precision(gt_indices, loaded_indices);

	// Final assertions (5% tolerance for approximate search with binary descriptors)
	EXPECT_GE(loaded_precision, cpu_precision - 0.05f)
		<< "Loaded precision dropped vs CPU baseline: " << loaded_precision << " vs " << cpu_precision;
	EXPECT_GE(loaded_precision, 0.80f)  // 128 threads/query
		<< "Loaded index precision too low: " << loaded_precision;

	// For Hierarchical with Hamming distance, use precision comparison instead of exact match
	// (ties in Hamming distance may cause valid reordering)
	EXPECT_GE(loaded_precision, gpu_precision - 0.05f)
		<< "Loaded precision dropped vs GPU before save: " << loaded_precision << " vs " << gpu_precision;

	// Cleanup
	delete[] cpu_indices.ptr();
	delete[] cpu_dists.ptr();
	delete[] gpu_indices.ptr();
	delete[] gpu_dists.ptr();
	delete[] loaded_indices.ptr();
	delete[] loaded_dists.ptr();
	remove(filename);

	std::cout << "E2E Precision Summary (Hierarchical): CPU=" << cpu_precision
		  << " GPU=" << gpu_precision
		  << " Loaded=" << loaded_precision
		  << " FileSize=" << file_size << " bytes\n";
}

/**
 * Test: GPU Format File Size Efficiency
 * Compare GPU format against CPU format WITH save_dataset=true (apples-to-apples)
 * GPU format always saves the dataset since CPU tree is discarded after conversion.
 */
TEST_F(HierarchicalCUDA_Brief100K, TestGPUFormatFileSizeEfficiency)
{
	// Build index with save_dataset=true for fair comparison
	flann::HierarchicalCUDAIndexParams params(32, FLANN_CENTERS_RANDOM, 4, 100);
	params["save_dataset"] = true;  // Enable dataset saving for CPU format
	flann::Index<Distance> index(data, params);
	index.buildIndex();

	// Save CPU format WITH dataset
	const char* cpu_file = "test_filesize_cpu_hier.idx";
	index.save(cpu_file);
	size_t cpu_size = getFileSize(cpu_file);

	// Convert and save GPU format
	index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
	index.convertToGPUFormat();
	const char* gpu_file = "test_filesize_gpu_hier.idx";
	index.save(gpu_file);
	size_t gpu_size = getFileSize(gpu_file);

	// Raw data size calculation (binary descriptors)
	size_t raw_dataset_bytes = data.rows * data.cols * sizeof(unsigned char);
	size_t padded_veclen = ((data.cols + 3) / 4) * 4;
	size_t padded_dataset_bytes = data.rows * padded_veclen * sizeof(unsigned char);

	std::cout << "File Size Analysis (Hierarchical, save_dataset=true):\n"
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
TEST_F(HierarchicalCUDA_Brief100K, TestGPUFormatLoadPerformance)
{
	const int num_runs = 3;

	// Setup: build and save both formats
	flann::Index<Distance> index(data,
		flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();

	const char* cpu_file = "test_loadperf_cpu_hier.idx";
	index.save(cpu_file);

	index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
	const char* gpu_file = "test_loadperf_gpu_hier.idx";
	index.save(gpu_file);

	// Benchmark CPU format load
	double cpu_load_ms = 0;
	for (int i = 0; i < num_runs; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		flann::Index<Distance> loaded(data, flann::SavedIndexParams(cpu_file));
		auto end = std::chrono::high_resolution_clock::now();
		cpu_load_ms += std::chrono::duration<double, std::milli>(end - start).count();
	}
	cpu_load_ms /= num_runs;

	// Benchmark GPU format load + warmup
	double gpu_load_ms = 0, gpu_warmup_ms = 0;
	for (int i = 0; i < num_runs; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		flann::Index<Distance> loaded(data, flann::SavedIndexParams(gpu_file));
		auto mid = std::chrono::high_resolution_clock::now();
		loaded.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
		auto end = std::chrono::high_resolution_clock::now();
		gpu_load_ms += std::chrono::duration<double, std::milli>(mid - start).count();
		gpu_warmup_ms += std::chrono::duration<double, std::milli>(end - mid).count();
	}
	gpu_load_ms /= num_runs;
	gpu_warmup_ms /= num_runs;

	std::cout << "Load Performance (Hierarchical, avg of " << num_runs << " runs):\n"
		  << "  CPU load:     " << cpu_load_ms << " ms\n"
		  << "  GPU load:     " << gpu_load_ms << " ms\n"
		  << "  GPU warmup:   " << gpu_warmup_ms << " ms\n"
		  << "  GPU total:    " << (gpu_load_ms + gpu_warmup_ms) << " ms\n";

	// GPU load includes data upload to GPU, so it may be slower than CPU load
	// The key metric is that it completes in reasonable time (<500ms for small datasets)
	EXPECT_LT(gpu_load_ms, 500.0)
		<< "GPU load too slow: " << gpu_load_ms << " ms";
	EXPECT_LT(gpu_warmup_ms, 500.0)
		<< "GPU warmup too slow: " << gpu_warmup_ms << " ms";

	remove(cpu_file);
	remove(gpu_file);
}

/**
 * Test: Format Auto-Detection
 * Verify both CPU and GPU formats can be loaded transparently
 */
TEST_F(HierarchicalCUDA_Brief100K, TestFormatDetection)
{
	// Pre-test cleanup for isolation
	remove("test_hier_cpu_format.idx");
	remove("test_hier_gpu_format.idx");

	// Create and save CPU format (without GPU init)
	{
		flann::Index<Distance> index(data,
			flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
		index.buildIndex();
		// Don't call buildCUDAKnnSearch - saves in CPU format
		EXPECT_FALSE(index.hasGPUFormat());
		index.save("test_hier_cpu_format.idx");
	}

	// Create and save GPU format
	{
		flann::Index<Distance> index(data,
			flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
		index.buildIndex();
		index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
		EXPECT_TRUE(index.hasGPUFormat());
		index.save("test_hier_gpu_format.idx");
	}

	// Load CPU format - should work
	{
		flann::Index<Distance> index(data,
			flann::SavedIndexParams("test_hier_cpu_format.idx"));
		EXPECT_EQ(index.size(), data.rows);
		EXPECT_FALSE(index.hasGPUFormat());
	}

	// Load GPU format - should work
	{
		flann::Index<Distance> index(data,
			flann::SavedIndexParams("test_hier_gpu_format.idx"));
		EXPECT_EQ(index.size(), data.rows);
		EXPECT_TRUE(index.hasGPUFormat());
	}

	remove("test_hier_cpu_format.idx");
	remove("test_hier_gpu_format.idx");
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
TEST_F(HierarchicalCUDA_Brief100K, TestGPUFormatSaveAfterRemove)
{
	const char* filename = "test_gpu_format_after_remove_hier.idx";
	remove(filename);  // Pre-test cleanup for isolation

	const int knn = 3;
	flann::Matrix<size_t> indices1(new size_t[query.rows*knn], query.rows, knn);
	flann::Matrix<DistanceType> dists1(new DistanceType[query.rows*knn], query.rows, knn);

	// Build index
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();
	index.buildCUDAKnnSearch(knn, flann::SearchParams(2000));

	// Search to find points to remove
	index.knnSearch(query, indices1, dists1, knn, flann::SearchParams(2000));

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
	index.buildCUDAKnnSearch(knn, flann::SearchParams(2000));
	ASSERT_TRUE(index.hasGPUFormat());
	index.convertToGPUFormat();

	// Save GPU format
	index.save(filename);

	// Load with SavedIndexParams and init GPU
	flann::Index<Distance> index2(data, flann::SavedIndexParams(filename));
	index2.buildCUDAKnnSearch(knn, flann::SearchParams(2000));

	// Search after load
	flann::Matrix<size_t> indices2(new size_t[query.rows*knn], query.rows, knn);
	flann::Matrix<DistanceType> dists2(new DistanceType[query.rows*knn], query.rows, knn);
	index2.knnSearch(query, indices2, dists2, knn, flann::SearchParams(2000));

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
TEST_F(HierarchicalCUDA_Brief100K, TestCPUFormatLoadWithGPU)
{
	const char* filename = "test_cpu_format_load_with_gpu_hier.idx";
	remove(filename);  // Pre-test cleanup for isolation

	const int knn = 3;

	// Build index WITHOUT GPU init (saves CPU format)
	flann::Index<Distance> index(data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();
	EXPECT_FALSE(index.hasGPUFormat()) << "Should not have GPU format before buildCUDAKnnSearch";

	// CPU baseline search
	flann::Matrix<size_t> indices_cpu(new size_t[query.rows*knn], query.rows, knn);
	flann::Matrix<DistanceType> dists_cpu(new DistanceType[query.rows*knn], query.rows, knn);
	index.knnSearch(query, indices_cpu, dists_cpu, knn, flann::SearchParams(2000));
	float cpu_precision = compute_precision(gt_indices, indices_cpu);
	printf("CPU baseline precision: %.2f%%\n", cpu_precision * 100);

	// Save CPU format
	index.save(filename);

	// Load and init GPU on CPU-format file
	flann::Index<Distance> index2(data, flann::SavedIndexParams(filename));
	EXPECT_FALSE(index2.hasGPUFormat()) << "Loaded CPU format should not have GPU format";
	index2.buildCUDAKnnSearch(knn, flann::SearchParams(2000));
	EXPECT_TRUE(index2.isGPUSearchReady()) << "GPU search should be ready after buildCUDAKnnSearch";

	// GPU search after loading CPU format
	flann::Matrix<size_t> indices_gpu(new size_t[query.rows*knn], query.rows, knn);
	flann::Matrix<DistanceType> dists_gpu(new DistanceType[query.rows*knn], query.rows, knn);
	index2.knnSearch(query, indices_gpu, dists_gpu, knn, flann::SearchParams(2000));
	float gpu_precision = compute_precision(gt_indices, indices_gpu);
	printf("GPU precision after CPU format load: %.2f%%\n", gpu_precision * 100);

	// GPU precision should be close to CPU (within 5% for Hamming distance with ties)
	EXPECT_GE(gpu_precision, cpu_precision - 0.05f)
		<< "GPU precision " << gpu_precision << " too low vs CPU " << cpu_precision;
	EXPECT_GE(gpu_precision, 0.80f) << "GPU precision below threshold (128 threads/query)";

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
TEST_F(HierarchicalCUDA_Brief100K, TestGPUFormatSaveAfterIncrementalAdd)
{
	const char* filename = "test_gpu_format_after_add_hier.idx";
	remove(filename);  // Pre-test cleanup for isolation

	const int knn = 3;

	// Split data 50/50
	size_t size1 = data.rows / 2;
	size_t size2 = data.rows - size1;
	flann::Matrix<unsigned char> data1(data[0], size1, data.cols);
	flann::Matrix<unsigned char> data2(data[size1], size2, data.cols);

	// Build with 50% data
	flann::Index<Distance> index(data1, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();
	index.buildCUDAKnnSearch(knn, flann::SearchParams(2000));
	EXPECT_EQ(index.size(), size1);

	// Add remaining 50%
	index.addPoints(data2, 2.0f);
	EXPECT_EQ(index.size(), data.rows);
	printf("Added %zu points (total now %zu)\n", size2, index.size());

	// Re-init GPU after add
	index.buildCUDAKnnSearch(knn, flann::SearchParams(2000));
	EXPECT_TRUE(index.hasGPUFormat());

	// Convert and save GPU format
	index.convertToGPUFormat();
	index.save(filename);

	// Load and init GPU
	flann::Index<Distance> index2(data, flann::SavedIndexParams(filename));
	index2.buildCUDAKnnSearch(knn, flann::SearchParams(2000));
	EXPECT_EQ(index2.size(), data.rows);

	// Search and verify precision
	flann::Matrix<size_t> indices(new size_t[query.rows*knn], query.rows, knn);
	flann::Matrix<DistanceType> dists(new DistanceType[query.rows*knn], query.rows, knn);
	index2.knnSearch(query, indices, dists, knn, flann::SearchParams(2000));
	float precision = compute_precision(gt_indices, indices);
	printf("Precision after incremental add + GPU save/load: %.2f%%\n", precision * 100);
	EXPECT_GE(precision, 0.80f) << "Precision after load (128 threads/query): " << precision;

	// Cleanup
	remove(filename);
	delete[] indices.ptr();
	delete[] dists.ptr();
}

// ============================================================================
// Test Isolation and Regression Tests
// These tests ensure GPU save/load works correctly in isolation, independent
// of test ordering or stale files from previous runs.
// ============================================================================

/**
 * Regression Test: GPU format load via SavedIndexParams in isolation
 *
 * This test verifies that loading a GPU-format index file via SavedIndexParams
 * works correctly WITHOUT any prior test state. The bug (fixed in load_saved_index)
 * was that LoadArchive's destructor tried to read from the file after it had
 * been rewound, causing a crash.
 *
 * This test MUST:
 * 1. Delete any pre-existing test file BEFORE the test
 * 2. Create a fresh GPU format file
 * 3. Close and reopen the file (simulate fresh process)
 * 4. Load via SavedIndexParams (the buggy code path)
 * 5. Verify the loaded index works correctly
 */
TEST_F(HierarchicalCUDA_Brief100K, RegressionTest_GPUFormatLoadIsolation)
{
	const char* filename = "test_regression_gpu_isolation_hier.idx";

	// CRITICAL: Clean up BEFORE test to ensure isolation
	remove(filename);

	// Step 1: Create and save GPU format index
	{
		flann::Index<Distance> index(data,
			flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
		index.buildIndex();
		index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
		ASSERT_TRUE(index.hasGPUFormat()) << "GPU format not available after buildCUDAKnnSearch";
		index.save(filename);
	}  // index destroyed, file closed

	// Step 2: Verify file exists
	FILE* check = fopen(filename, "rb");
	ASSERT_NE(check, nullptr) << "GPU format file was not created";
	fclose(check);

	// Step 3: Load via SavedIndexParams (this was the buggy path)
	// This must work without any prior CUDA context or state
	flann::Index<Distance> loaded(data,
		flann::SavedIndexParams(filename));

	// Step 4: Initialize GPU and search
	loaded.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
	ASSERT_TRUE(loaded.isGPUSearchReady()) << "GPU search not ready after load";

	flann::Matrix<size_t> indices_loaded(new size_t[query.rows * k_nn_], query.rows, k_nn_);
	flann::Matrix<DistanceType> dists_loaded(new DistanceType[query.rows * k_nn_], query.rows, k_nn_);
	loaded.knnSearch(query, indices_loaded, dists_loaded, k_nn_, flann::SearchParams(2000));

	// Step 5: Verify results are valid
	float precision = compute_precision(gt_indices, indices_loaded);
	EXPECT_GE(precision, 0.80f) << "Loaded index precision too low: " << precision;
	printf("Regression test precision: %.2f%%\n", precision * 100);

	delete[] indices_loaded.ptr();
	delete[] dists_loaded.ptr();
	remove(filename);
}

/**
 * Test: GPU format file I/O integrity
 * Verifies the file I/O pattern used in load_saved_index() works correctly:
 * - detectV2Format() peeks at signature and restores position
 * - Header can be read directly after rewind
 * - File can be rewound and re-read multiple times
 */
TEST_F(HierarchicalCUDA_Brief100K, TestGPUFormatFileIO)
{
	const char* filename = "test_gpu_fileio_hier.idx";
	remove(filename);  // Pre-test cleanup

	// Create GPU format file
	{
		flann::Index<Distance> index(data,
			flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
		index.buildIndex();
		index.buildCUDAKnnSearch(k_nn_, flann::SearchParams(2000));
		index.save(filename);
	}

	// Test file I/O pattern manually
	FILE* fin = fopen(filename, "rb");
	ASSERT_NE(fin, nullptr) << "Failed to open GPU format file";

	// Step 1: Detect GPU format (reads signature, should restore position)
	EXPECT_TRUE(flann::GPUIndexHeaderV2::detectV2Format(fin))
		<< "Failed to detect GPU v2.0 format";

	// Step 2: Read header directly (like fixed code does)
	rewind(fin);
	flann::GPUIndexHeaderV2Struct header;
	EXPECT_EQ(fread(&header, sizeof(header), 1, fin), 1u)
		<< "Failed to read GPU header";

	// Step 3: Verify header fields
	EXPECT_EQ(header.rows, data.rows) << "Header rows mismatch";
	EXPECT_EQ(header.cols, data.cols) << "Header cols mismatch";

	// Step 4: Rewind and verify we can read again
	rewind(fin);
	flann::GPUIndexHeaderV2Struct header2;
	EXPECT_EQ(fread(&header2, sizeof(header2), 1, fin), 1u)
		<< "Failed to re-read GPU header after rewind";
	EXPECT_EQ(header.rows, header2.rows) << "Header changed after rewind";
	EXPECT_EQ(header.cols, header2.cols) << "Header changed after rewind";

	fclose(fin);
	remove(filename);
}

/**
 * Test: Verify GPU save/load tests can run in any order
 * This test explicitly verifies test isolation by using a unique filename
 * and creating a fresh index with no reliance on fixture state.
 * Run this test first, last, and in isolation - should always pass.
 */
TEST_F(HierarchicalCUDA_Brief100K, TestIsolation_GPUSaveLoadIndependent)
{
	// Use unique filename unlikely to conflict
	const char* filename = "test_isolation_unique_hier_12345.idx";
	remove(filename);  // Pre-test cleanup

	// Create fresh index - minimal reliance on test fixture
	flann::Index<Distance> index(data,
		flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	index.buildIndex();
	index.buildCUDAKnnSearch(10, flann::SearchParams(2000));
	ASSERT_TRUE(index.hasGPUFormat());
	index.save(filename);

	// Load from scratch via SavedIndexParams
	flann::Index<Distance> loaded(data,
		flann::SavedIndexParams(filename));
	loaded.buildCUDAKnnSearch(10, flann::SearchParams(2000));

	EXPECT_TRUE(loaded.isGPUSearchReady())
		<< "GPU search not ready - test isolation may be broken";
	EXPECT_EQ(loaded.size(), data.rows)
		<< "Loaded index size mismatch";

	remove(filename);
}

/**
 * Lightweight test class for CPU format loading tests.
 * Uses minimal dataset to avoid fixture overhead issues.
 */
class HierarchicalCUDA_LoadTests : public ::testing::Test
{
protected:
	typedef flann::Hamming<unsigned char> Distance;
	typedef Distance::ElementType ElementType;
	typedef Distance::ResultType DistanceType;
	static constexpr size_t N = 1000;  // Small dataset
	static constexpr size_t D = 32;    // Descriptor dimension
	static constexpr size_t Q = 10;    // Query count
	static constexpr unsigned int K = 3;

	flann::Matrix<unsigned char> data;
	flann::Matrix<unsigned char> query;

	void SetUp() override
	{
		// Create synthetic data
		unsigned char* data_ptr = new unsigned char[N * D];
		unsigned char* query_ptr = new unsigned char[Q * D];

		for (size_t i = 0; i < N * D; ++i) {
			data_ptr[i] = static_cast<unsigned char>(rand() % 256);
		}
		for (size_t i = 0; i < Q * D; ++i) {
			query_ptr[i] = static_cast<unsigned char>(rand() % 256);
		}

		data = flann::Matrix<unsigned char>(data_ptr, N, D);
		query = flann::Matrix<unsigned char>(query_ptr, Q, D);
	}

	void TearDown() override
	{
		delete[] data.ptr();
		delete[] query.ptr();
	}
};

/**
 * Test: Load CPU v1.1 format index saved by HierarchicalCUDAIndex
 *
 * When HierarchicalCUDAIndex saves WITHOUT GPU initialization (gpu_initialized_=false),
 * it falls back to BaseClass::saveIndex() which writes CPU v1.1 format.
 * This test verifies we can load that format back.
 */
TEST_F(HierarchicalCUDA_LoadTests, LoadCPUFormatSavedByCUDAIndex)
{
	const char* filename = "test_cuda_cpu_format.idx";
	remove(filename);

	// Step 1: Create CUDA index, build (CPU tree only), save WITHOUT GPU init
	{
		flann::HierarchicalCUDAIndex<Distance> cuda_index(
			data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
		cuda_index.buildIndex();  // Builds CPU tree only

		// Save - since gpu_initialized_ is false, falls back to CPU format
		FILE* fout = fopen(filename, "wb");
		ASSERT_NE(fout, nullptr);
		cuda_index.saveIndex(fout);
		fclose(fout);
	}

	// Step 2: Verify it's CPU format (not GPU v2.0)
	{
		FILE* fin = fopen(filename, "rb");
		ASSERT_NE(fin, nullptr);
		EXPECT_FALSE(flann::GPUIndexHeaderV2::detectV2Format(fin))
			<< "Index should be CPU format when saved without GPU init";
		fclose(fin);
	}

	// Step 3: Load back with CUDA index (tests our rewind fix)
	{
		flann::HierarchicalCUDAIndex<Distance> loaded_index(
			data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));

		FILE* fin = fopen(filename, "rb");
		ASSERT_NE(fin, nullptr);
		ASSERT_NO_THROW(loaded_index.loadIndex(fin))
			<< "loadIndex() should work with rewind() fix";
		fclose(fin);

		// Step 4: Verify GPU search can be enabled on loaded index
		loaded_index.buildCUDAKnnSearch(K, flann::SearchParams(256));
		EXPECT_TRUE(loaded_index.isGPUSearchReady());

		// Step 5: Search and verify results
		flann::Matrix<int> indices(new int[Q * K], Q, K);
		flann::Matrix<DistanceType> dists(new DistanceType[Q * K], Q, K);
		loaded_index.knnSearch(query, indices, dists, K, flann::SearchParams(256));

		// Check we got valid indices
		bool valid = true;
		for (size_t i = 0; i < Q && valid; ++i) {
			for (size_t j = 0; j < K && valid; ++j) {
				if (indices[i][j] < 0 || indices[i][j] >= (int)N) {
					valid = false;
				}
			}
		}
		EXPECT_TRUE(valid) << "Search returned invalid indices";

		delete[] indices.ptr();
		delete[] dists.ptr();
	}

	remove(filename);
}

/**
 * Test: Load CPU v1.1 HIERARCHICAL format (saved by HierarchicalClusteringIndex)
 *
 * This tests the core Issue 2 fix: loading a file saved by CPU-only
 * HierarchicalClusteringIndex (index_type = FLANN_INDEX_HIERARCHICAL = 5)
 * into HierarchicalCUDAIndex (which returns FLANN_INDEX_HIERARCHICAL_CUDA = 12).
 *
 * Before the fix, this would fail with:
 * "Saved index type is different then the current index type"
 */
TEST_F(HierarchicalCUDA_LoadTests, LoadPureCPUHierarchicalFormat)
{
	const char* filename = "test_pure_cpu_hierarchical.idx";
	remove(filename);

	// Step 1: Create and save using pure CPU HierarchicalClusteringIndex
	{
		// Create params with save_dataset=true so the dataset is embedded in the file
		flann::HierarchicalClusteringIndexParams params(32, FLANN_CENTERS_RANDOM, 4, 100);
		params["save_dataset"] = true;

		flann::HierarchicalClusteringIndex<Distance> cpu_index(data, params);
		cpu_index.buildIndex();

		FILE* fout = fopen(filename, "wb");
		ASSERT_NE(fout, nullptr);
		cpu_index.saveIndex(fout);
		fclose(fout);
	}

	// Step 2: Verify it's CPU v1.1 format with HIERARCHICAL type (not HIERARCHICAL_CUDA)
	{
		FILE* fin = fopen(filename, "rb");
		ASSERT_NE(fin, nullptr);

		// Should NOT be GPU v2.0 format
		EXPECT_FALSE(flann::GPUIndexHeaderV2::detectV2Format(fin))
			<< "Index should be CPU format";

		// Read header and verify index type
		rewind(fin);
		flann::IndexHeader header = flann::load_header(fin);
		EXPECT_EQ(header.h.index_type, FLANN_INDEX_HIERARCHICAL)
			<< "Index type should be FLANN_INDEX_HIERARCHICAL (5), not HIERARCHICAL_CUDA (12)";
		fclose(fin);
	}

	// Step 3: Load with HierarchicalCUDAIndex (this tests the fix!)
	{
		flann::HierarchicalCUDAIndex<Distance> cuda_index(
			data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));

		FILE* fin = fopen(filename, "rb");
		ASSERT_NE(fin, nullptr);
		ASSERT_NO_THROW(cuda_index.loadIndex(fin))
			<< "HierarchicalCUDAIndex should accept FLANN_INDEX_HIERARCHICAL files";
		fclose(fin);

		// Step 4: Verify index loaded correctly
		EXPECT_EQ(cuda_index.size(), N);
		EXPECT_EQ(cuda_index.veclen(), D);

		// Step 5: Enable GPU search and verify it works
		cuda_index.buildCUDAKnnSearch(K, flann::SearchParams(256));
		EXPECT_TRUE(cuda_index.isGPUSearchReady());

		// Step 6: Search and verify results
		flann::Matrix<int> indices(new int[Q * K], Q, K);
		flann::Matrix<DistanceType> dists(new DistanceType[Q * K], Q, K);
		cuda_index.knnSearch(query, indices, dists, K, flann::SearchParams(256));

		// Check we got valid indices
		bool valid = true;
		for (size_t i = 0; i < Q && valid; ++i) {
			for (size_t j = 0; j < K && valid; ++j) {
				if (indices[i][j] < 0 || indices[i][j] >= (int)N) {
					valid = false;
				}
			}
		}
		EXPECT_TRUE(valid) << "Search returned invalid indices";

		delete[] indices.ptr();
		delete[] dists.ptr();
	}

	remove(filename);
}

/**
 * Test: Verify loadIndex() handles null stream gracefully
 */
TEST_F(HierarchicalCUDA_LoadTests, LoadNullStream)
{
	flann::HierarchicalCUDAIndex<Distance> cuda_index(
		data, flann::HierarchicalCUDAIndexParams(32, FLANN_CENTERS_RANDOM, 4, 100));
	EXPECT_THROW(cuda_index.loadIndex(nullptr), flann::FLANNException);
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
