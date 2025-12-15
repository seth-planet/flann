/**
 * @file flann_kmeans_opencl_test.cpp
 * @brief K-Means OpenCL GPU acceleration tests
 *
 * TEST COVERAGE NOTES:
 * --------------------
 * These tests provide basic coverage for OpenCL K-Means functionality.
 * Compared to CUDA tests (flann_kmeans_cuda_test.cpp), the following are NOT covered:
 *
 *   - K-value coverage tests (only default k tested, not k=1,2,4,5,7,8,10,16,20,32,50,64,100)
 *   - Edge case tests (k=0, k > dataset size)
 *   - GPU format save/load tests (GPU v2.0 format)
 *   - Test isolation/regression tests
 *   - Performance benchmarks
 *
 * PRECISION THRESHOLD:
 * --------------------
 * OpenCL tests use 75% precision threshold vs CUDA's 96%. This is due to:
 *   - OpenCL uses branching=7 (not power-of-2) vs CUDA's branching=32
 *   - Different cooperative kernel optimizations
 *   - OpenCL precision parity investigation deferred to future MR
 *
 * FUTURE WORK:
 * ------------
 * A future MR should port comprehensive test coverage from CUDA tests.
 */

#define FLANN_USE_OPENCL
#include <gtest/gtest.h>
#include <time.h>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "flann_tests.h"

using namespace flann;

/**
 * Test fixture for SIFT 10K dataset
 */
class KMeansOpenCL_SIFT10K : public DatasetTestFixture<float, float> {
protected:
	KMeansOpenCL_SIFT10K() : DatasetTestFixture("sift10K.h5") {}
};




TEST_F(KMeansOpenCL_SIFT10K, TestSearch)
{
	TestSearch<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT10K, TestSearch2)
{
	TestSearch2<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT10K, TestAddIncremental)
{
	TestAddIncremental<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(110), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT10K, TestAddIncremental2)
{
	TestAddIncremental2<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(110), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT10K, TestRemove)
{
	TestRemove<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128));
}



TEST_F(KMeansOpenCL_SIFT10K, TestSave)
{
	TestSave<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT10K, TestCopy)
{
	TestCopy<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT10K, TestCopy2)
{
	TestCopy2<KMeansOpenCLIndex<flann::L2<float> > >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}

/**
 * Test fixture for SIFT 100K dataset
 */
class KMeansOpenCL_SIFT100K : public DatasetTestFixture<float, float> {
protected:
	KMeansOpenCL_SIFT100K() : DatasetTestFixture("sift100K.h5") {}
};


TEST_F(KMeansOpenCL_SIFT100K, TestSearch)
{
	TestSearch<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(96), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT100K, TestAddIncremental)
{
	TestAddIncremental<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}

TEST_F(KMeansOpenCL_SIFT100K, TestAddIncremental2)
{
	TestAddIncremental2<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}


TEST_F(KMeansOpenCL_SIFT100K, TestRemove)
{
	TestRemove<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4), query, indices,
			dists, knn, flann::SearchParams(128) );
}

TEST_F(KMeansOpenCL_SIFT100K, TestSave)
{
	TestSave<flann::L2<float> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(96), 0.75, gt_indices);
}



/**
 * Test fixture for SIFT 10K dataset with byte feature elements
 */
class KMeansOpenCL_SIFT10K_byte :  public DatasetTestFixture<unsigned char, float> {
protected:
	KMeansOpenCL_SIFT10K_byte() : DatasetTestFixture("sift10K_byte.h5") {}
};

TEST_F(KMeansOpenCL_SIFT10K_byte, TestSearch)
{
	TestSearch<flann::L2<unsigned char> >(data, flann::KMeansOpenCLIndexParams(7, 3, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(128), 0.75, gt_indices);
}



class KMeansOpenCL_SIFT100K_byte : public DatasetTestFixture<unsigned char, float> {
protected:
	KMeansOpenCL_SIFT100K_byte() : DatasetTestFixture("sift100K_byte.h5") {}
};

TEST_F(KMeansOpenCL_SIFT100K_byte, TestSearch)
{
	TestSearch<flann::L2<unsigned char> >(data, flann::KMeansOpenCLIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.4),
			query, indices, dists, knn, flann::SearchParams(96), 0.75, gt_indices);
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
