/**
 * @file flann_cuda_device_dataset_test.cpp
 * @brief deviceDataset() exposes the device rows in point order, after build and after a v2 load.
 *
 * A consumer that reads descriptors by neighbour id through this view depends on row i being
 * point i, row_bytes apart. Each arm copies the view back and compares it with the input, so a
 * reordered or repacked device dataset fails here rather than as wrong distances downstream.
 */

#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <flann/flann.hpp>

namespace
{

typedef flann::Hamming<unsigned char> HammingDist;
typedef flann::cuda::HierarchicalCUDAIndex<HammingDist> HierarchicalIndex;

const size_t kNumPoints = 4000;
const size_t kVeclen = 64;

std::vector<unsigned char> random_bytes(size_t count, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<unsigned char> v(count);
    for (auto& x : v) {
        x = static_cast<unsigned char>(dist(gen));
    }
    return v;
}

// Copies the view to the host and returns the index of the first row that differs from the
// input, or -1 when every row matches.
long first_mismatching_row(const HierarchicalIndex::DeviceDatasetView& view,
                           const std::vector<unsigned char>& input, size_t veclen)
{
    std::vector<unsigned char> host(view.rows * view.row_bytes);
    EXPECT_EQ(cudaMemcpy(host.data(), view.data, host.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < view.rows; ++i) {
        if (std::memcmp(&host[i * view.row_bytes], &input[i * veclen], veclen) != 0) {
            return static_cast<long>(i);
        }
    }
    return -1;
}

flann::HierarchicalCUDAIndexParams shipped_params()
{
    return flann::HierarchicalCUDAIndexParams(32, flann::FLANN_CENTERS_RANDOM, 4, 100);
}

}  // namespace

TEST(DeviceDataset, RowsArePointsInOrderAfterBuild)
{
    std::vector<unsigned char> data = random_bytes(kNumPoints * kVeclen, 2468);
    flann::Matrix<unsigned char> dataset(data.data(), kNumPoints, kVeclen);
    HierarchicalIndex index(dataset, shipped_params());
    index.buildIndex();
    index.buildCUDAKnnSearch(16);

    const HierarchicalIndex::DeviceDatasetView view = index.deviceDataset();
    EXPECT_EQ(view.rows, kNumPoints);
    EXPECT_EQ(view.row_bytes, kVeclen);
    ASSERT_NE(view.data, nullptr);
    EXPECT_EQ(first_mismatching_row(view, data, kVeclen), -1);
}

TEST(DeviceDataset, RowsArePointsInOrderAfterAV2SaveAndGpuOnlyLoad)
{
    const std::string filename = "test_device_dataset_v2.idx";
    std::remove(filename.c_str());
    std::vector<unsigned char> data = random_bytes(kNumPoints * kVeclen, 1357);
    {
        flann::Matrix<unsigned char> dataset(data.data(), kNumPoints, kVeclen);
        HierarchicalIndex index(dataset, shipped_params());
        index.buildIndex();
        index.buildCUDAKnnSearch(16);
        FILE* fout = std::fopen(filename.c_str(), "wb");
        ASSERT_NE(fout, nullptr);
        index.saveIndex(fout);
        std::fclose(fout);
    }

    // The consumer's load: a default-constructed index filled from the v2 file (GPU-only mode),
    // then prepareGPUIndex(), as cuda_pipeline's FLANNQueryEngine::EnsureFlannLoaded does.
    HierarchicalIndex loaded;
    FILE* fin = std::fopen(filename.c_str(), "rb");
    ASSERT_NE(fin, nullptr);
    ASSERT_TRUE(flann::GPUIndexHeaderV2::detectV2Format(fin)) << "the save did not write the v2 format";
    loaded.loadIndex(fin);
    std::fclose(fin);
    std::remove(filename.c_str());
    loaded.prepareGPUIndex();
    ASSERT_TRUE(loaded.isGPUOnlyMode());

    const HierarchicalIndex::DeviceDatasetView view = loaded.deviceDataset();
    EXPECT_EQ(view.rows, kNumPoints);
    EXPECT_EQ(view.row_bytes, kVeclen);
    EXPECT_EQ(first_mismatching_row(view, data, kVeclen), -1);
}

TEST(DeviceDataset, RefusesBeforeTheGpuIndexExists)
{
    std::vector<unsigned char> data = random_bytes(kNumPoints * kVeclen, 97);
    flann::Matrix<unsigned char> dataset(data.data(), kNumPoints, kVeclen);
    HierarchicalIndex index(dataset, shipped_params());
    index.buildIndex();
    try {
        index.deviceDataset();
        FAIL() << "deviceDataset() returned before the GPU index was built";
    } catch (const flann::FLANNException& e) {
        EXPECT_NE(std::string(e.what()).find("not initialized"), std::string::npos) << e.what();
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
