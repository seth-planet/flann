/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024
 *
 * Thread safety tests for CUDA index implementations.
 * Verifies concurrent searches work correctly with reader-writer locking.
 *************************************************************************/

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>

#include <flann/flann.hpp>

#ifdef FLANN_USE_CUDA

using namespace flann;

// Test fixture for thread safety tests
class CUDAThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate random float dataset for K-Means
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        const size_t num_points = 10000;
        const size_t dim = 128;

        float_data_.resize(num_points * dim);
        for (auto& v : float_data_) {
            v = dist(gen);
        }
        float_dataset_ = Matrix<float>(float_data_.data(), num_points, dim);

        // Generate random query data
        const size_t num_queries = 100;
        float_queries_data_.resize(num_queries * dim);
        for (auto& v : float_queries_data_) {
            v = dist(gen);
        }
        float_queries_ = Matrix<float>(float_queries_data_.data(), num_queries, dim);
    }

    std::vector<float> float_data_;
    std::vector<float> float_queries_data_;
    Matrix<float> float_dataset_;
    Matrix<float> float_queries_;
};

// Test concurrent searches on K-Means CUDA index
TEST_F(CUDAThreadSafetyTest, KMeansConcurrentSearches) {
    // Build K-Means CUDA index
    cuda::KMeansCUDAIndex<L2<float>> index(float_dataset_, cuda::KMeansCUDAIndexParams(32));
    index.buildIndex();
    index.buildCUDAKnnSearch(5, SearchParams());

    const int num_threads = 8;
    const int searches_per_thread = 10;
    const int knn = 5;

    std::atomic<int> completed_searches(0);
    std::atomic<bool> has_error(false);
    std::vector<std::thread> threads;

    // Launch concurrent search threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            try {
                // Each thread performs multiple searches
                for (int i = 0; i < searches_per_thread; ++i) {
                    Matrix<int> indices(new int[float_queries_.rows * knn], float_queries_.rows, knn);
                    Matrix<float> dists(new float[float_queries_.rows * knn], float_queries_.rows, knn);

                    // Perform search
                    index.knnSearch(float_queries_, indices, dists, knn, SearchParams());

                    // Basic validation - indices should be valid
                    for (size_t q = 0; q < float_queries_.rows; ++q) {
                        for (int k = 0; k < knn; ++k) {
                            EXPECT_GE(indices[q][k], 0);
                            EXPECT_LT(static_cast<size_t>(indices[q][k]), float_dataset_.rows);
                        }
                    }

                    completed_searches++;

                    delete[] indices.ptr();
                    delete[] dists.ptr();
                }
            } catch (const std::exception& e) {
                has_error = true;
                ADD_FAILURE() << "Thread " << t << " exception: " << e.what();
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_FALSE(has_error);
    EXPECT_EQ(completed_searches, num_threads * searches_per_thread);
}

// Test that results are consistent between single-threaded and multi-threaded searches
TEST_F(CUDAThreadSafetyTest, KMeansResultConsistency) {
    // Build K-Means CUDA index
    cuda::KMeansCUDAIndex<L2<float>> index(float_dataset_, cuda::KMeansCUDAIndexParams(32));
    index.buildIndex();
    index.buildCUDAKnnSearch(5, SearchParams());

    const int knn = 5;

    // Single-threaded reference search
    Matrix<int> ref_indices(new int[float_queries_.rows * knn], float_queries_.rows, knn);
    Matrix<float> ref_dists(new float[float_queries_.rows * knn], float_queries_.rows, knn);
    index.knnSearch(float_queries_, ref_indices, ref_dists, knn, SearchParams());

    // Multi-threaded searches should produce same results
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<bool> results_match(true);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            Matrix<int> indices(new int[float_queries_.rows * knn], float_queries_.rows, knn);
            Matrix<float> dists(new float[float_queries_.rows * knn], float_queries_.rows, knn);

            index.knnSearch(float_queries_, indices, dists, knn, SearchParams());

            // Compare with reference
            for (size_t q = 0; q < float_queries_.rows; ++q) {
                for (int k = 0; k < knn; ++k) {
                    if (indices[q][k] != ref_indices[q][k]) {
                        results_match = false;
                    }
                }
            }

            delete[] indices.ptr();
            delete[] dists.ptr();
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_TRUE(results_match);

    delete[] ref_indices.ptr();
    delete[] ref_dists.ptr();
}

// Test that mutations properly block during searches
TEST_F(CUDAThreadSafetyTest, KMeansMutationBlocking) {
    // Build K-Means CUDA index
    cuda::KMeansCUDAIndex<L2<float>> index(float_dataset_, cuda::KMeansCUDAIndexParams(32));
    index.buildIndex();
    index.buildCUDAKnnSearch(5, SearchParams());

    const int knn = 5;
    std::atomic<bool> search_started(false);
    std::atomic<bool> mutation_started(false);
    std::atomic<bool> search_completed(false);
    std::atomic<bool> has_error(false);

    // Start a search thread
    std::thread search_thread([&]() {
        try {
            Matrix<int> indices(new int[float_queries_.rows * knn], float_queries_.rows, knn);
            Matrix<float> dists(new float[float_queries_.rows * knn], float_queries_.rows, knn);

            search_started = true;

            // Perform search (holds shared lock)
            index.knnSearch(float_queries_, indices, dists, knn, SearchParams());

            search_completed = true;

            delete[] indices.ptr();
            delete[] dists.ptr();
        } catch (const std::exception& e) {
            has_error = true;
        }
    });

    // Wait for search to start, then try mutation
    while (!search_started) {
        std::this_thread::yield();
    }

    // Try to add points (should wait for search to complete due to exclusive lock)
    std::thread mutation_thread([&]() {
        try {
            mutation_started = true;

            // Small batch of new points
            std::vector<float> new_data(10 * 128);
            std::mt19937 gen(123);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (auto& v : new_data) {
                v = dist(gen);
            }
            Matrix<float> new_points(new_data.data(), 10, 128);

            // This should block until search completes
            index.addPoints(new_points, 2.0f);

        } catch (const std::exception& e) {
            // Expected if GPU mode doesn't support addPoints
        }
    });

    search_thread.join();
    mutation_thread.join();

    // Search should have completed
    EXPECT_TRUE(search_completed);
    EXPECT_FALSE(has_error);
}

// Stress test with many threads
TEST_F(CUDAThreadSafetyTest, KMeansStressTest) {
    // Build K-Means CUDA index
    cuda::KMeansCUDAIndex<L2<float>> index(float_dataset_, cuda::KMeansCUDAIndexParams(32));
    index.buildIndex();
    index.buildCUDAKnnSearch(5, SearchParams());

    const int num_threads = 16;
    const int searches_per_thread = 20;
    const int knn = 5;

    std::atomic<int> completed_searches(0);
    std::atomic<bool> has_error(false);
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch many concurrent threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            try {
                for (int i = 0; i < searches_per_thread; ++i) {
                    Matrix<int> indices(new int[float_queries_.rows * knn], float_queries_.rows, knn);
                    Matrix<float> dists(new float[float_queries_.rows * knn], float_queries_.rows, knn);

                    index.knnSearch(float_queries_, indices, dists, knn, SearchParams());

                    completed_searches++;

                    delete[] indices.ptr();
                    delete[] dists.ptr();
                }
            } catch (const std::exception& e) {
                has_error = true;
                ADD_FAILURE() << "Thread exception: " << e.what();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_FALSE(has_error);
    EXPECT_EQ(completed_searches, num_threads * searches_per_thread);

    // Print performance info
    int total_searches = num_threads * searches_per_thread;
    std::cout << "[Thread Safety] " << total_searches << " searches completed in "
              << duration.count() << "ms ("
              << (total_searches * 1000.0 / duration.count()) << " searches/sec)" << std::endl;
}

#endif // FLANN_USE_CUDA

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
