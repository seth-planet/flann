/**
 * @file flann_cuda_launch_status_test.cpp
 * @brief A search that throws because a kernel launcher returned false names the CUDA status.
 *
 * Callers classify a failed search by its message text, and only the status name separates a
 * latched context from a transient failure. One arm per throw site that turns a launcher's
 * false into a FLANNException. The grid-0 arm makes the search launch itself fail; the others
 * leave a failed cudaMalloc's status pending, which the launcher's error check then reports as
 * its own, so each shows that its site reads the status and renders it. Every arm also checks
 * the site consumed the status, since one left pending would be thrown by the next healthy
 * CUDA_CHECK_LAST on the thread.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <flann/flann.hpp>

namespace
{

const std::string kPadText = "Failed to launch GPU padding kernel";
const std::string kSearchText = "Kernel launch failed for k=16 at search width ";
const size_t kNumPoints = 4000;
const size_t kNumQueries = 8;
const size_t kKnn = 16;

template <typename T>
std::vector<T> random_data(size_t count, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<T> v(count);
    for (auto& x : v) {
        x = static_cast<T>(dist(gen));
    }
    return v;
}

// Leaves a failed cudaMalloc's status in the last-error slot and returns it. The slot is
// drained first so that the status the search reports can only be this one.
cudaError_t leave_status_pending()
{
    cudaGetLastError();
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, SIZE_MAX);
    return e;
}

std::string status_suffix(cudaError_t e)
{
    return std::string(": ") + cudaGetErrorString(e) + " (" + cudaGetErrorName(e) + ")";
}

bool ends_with(const std::string& s, const std::string& tail)
{
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

template <typename T>
struct DeviceArray
{
    T* ptr = nullptr;
    explicit DeviceArray(size_t count) { EXPECT_EQ(cudaMalloc(&ptr, count * sizeof(T)), cudaSuccess); }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

typedef flann::Hamming<unsigned char> HammingDist;
typedef flann::cuda::HierarchicalCUDAIndex<HammingDist> HierarchicalIndex;
typedef flann::L2<float> L2Dist;
typedef flann::cuda::KMeansCUDAIndex<L2Dist> KMeansIndex;

// Hamming descriptors of veclen bytes; a veclen that is not a multiple of 4 makes every
// search pad its queries first.
class HierarchicalLaunchStatus : public ::testing::Test
{
protected:
    void build(size_t veclen)
    {
        veclen_ = veclen;
        data_ = random_data<unsigned char>(kNumPoints * veclen, 1234);
        queries_ = random_data<unsigned char>(kNumQueries * veclen, 5678);
        flann::Matrix<unsigned char> dataset(data_.data(), kNumPoints, veclen);
        index_.reset(new HierarchicalIndex(dataset, flann::HierarchicalCUDAIndexParams(32, flann::FLANN_CENTERS_RANDOM, 4, 100)));
        index_->buildIndex();
        index_->buildCUDAKnnSearch(static_cast<int>(kKnn));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    size_t veclen_ = 0;
    std::vector<unsigned char> data_;
    std::vector<unsigned char> queries_;
    std::unique_ptr<HierarchicalIndex> index_;
};

// Float descriptors of veclen elements; KMeansCUDAIndex pads any veclen that is not a
// multiple of 4.
class KMeansLaunchStatus : public ::testing::Test
{
protected:
    void SetUp() override
    {
        data_ = random_data<float>(kNumPoints * kVeclen, 4321);
        queries_ = random_data<float>(kNumQueries * kVeclen, 8765);
        flann::Matrix<float> dataset(data_.data(), kNumPoints, kVeclen);
        index_.reset(new KMeansIndex(dataset, flann::KMeansCUDAIndexParams(32, 11, flann::FLANN_CENTERS_RANDOM, 0.2f)));
        index_->buildIndex();
        index_->buildCUDAKnnSearch(static_cast<int>(kKnn));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    static const size_t kVeclen = 30;
    std::vector<float> data_;
    std::vector<float> queries_;
    std::unique_ptr<KMeansIndex> index_;
};

}  // namespace

TEST_F(HierarchicalLaunchStatus, DirectSearchNamesTheStatusOfAFailedLaunch)
{
    build(32);
    DeviceArray<unsigned char> d_queries(kNumQueries * veclen_);
    DeviceArray<int> d_indices(kNumQueries * kKnn);
    DeviceArray<int> d_dists(kNumQueries * kKnn);
    cudaMemcpy(d_queries.ptr, queries_.data(), queries_.size(), cudaMemcpyHostToDevice);
    cudaGetLastError();
    // The kernel's grid is one block per query in a 32-bit dimension, so 2^32 queries launch
    // a grid of 0 blocks, which the runtime refuses. Nothing earlier in the search sizes an
    // allocation from the query count when no padding is needed.
    const size_t num_queries = size_t(1) << 32;
    std::string what = "<no FLANNException>";
    try {
        index_->knnSearchGPUDirect(d_queries.ptr, d_indices.ptr, d_dists.ptr, num_queries, kKnn,
                                   flann::SearchParams(), nullptr);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    std::printf("grid-0 launch: %s\n", what.c_str());
    EXPECT_EQ(what.rfind(kSearchText, 0), 0u) << what;
    const size_t open = what.rfind(" (cudaError");
    ASSERT_NE(open, std::string::npos) << what;
    EXPECT_TRUE(ends_with(what, ")")) << what;
    EXPECT_EQ(what.find("(cudaSuccess)"), std::string::npos) << what;
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(HierarchicalLaunchStatus, DirectSearchPadNamesThePendingStatus)
{
    build(30);
    DeviceArray<unsigned char> d_queries(kNumQueries * veclen_);
    DeviceArray<int> d_indices(kNumQueries * kKnn);
    DeviceArray<int> d_dists(kNumQueries * kKnn);
    cudaMemcpy(d_queries.ptr, queries_.data(), queries_.size(), cudaMemcpyHostToDevice);
    const cudaError_t e = leave_status_pending();
    ASSERT_NE(e, cudaSuccess);
    std::string what = "<no FLANNException>";
    try {
        index_->knnSearchGPUDirect(d_queries.ptr, d_indices.ptr, d_dists.ptr, kNumQueries, kKnn,
                                   flann::SearchParams(), nullptr);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_EQ(what, kPadText + status_suffix(e));
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(HierarchicalLaunchStatus, HostSearchPadNamesThePendingStatus)
{
    build(30);
    flann::Matrix<unsigned char> queries(queries_.data(), kNumQueries, veclen_);
    std::vector<size_t> idx(kNumQueries * kKnn);
    std::vector<unsigned int> dst(kNumQueries * kKnn);
    flann::Matrix<size_t> indices(idx.data(), kNumQueries, kKnn);
    flann::Matrix<unsigned int> dists(dst.data(), kNumQueries, kKnn);
    const cudaError_t e = leave_status_pending();
    ASSERT_NE(e, cudaSuccess);
    std::string what = "<no FLANNException>";
    try {
        index_->knnSearchGPU(queries, indices, dists, kKnn, flann::SearchParams(), nullptr);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_EQ(what, kPadText + status_suffix(e));
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(HierarchicalLaunchStatus, HostSearchNamesThePendingStatus)
{
    build(32);
    flann::Matrix<unsigned char> queries(queries_.data(), kNumQueries, veclen_);
    std::vector<size_t> idx(kNumQueries * kKnn);
    std::vector<unsigned int> dst(kNumQueries * kKnn);
    flann::Matrix<size_t> indices(idx.data(), kNumQueries, kKnn);
    flann::Matrix<unsigned int> dists(dst.data(), kNumQueries, kKnn);
    const cudaError_t e = leave_status_pending();
    ASSERT_NE(e, cudaSuccess);
    std::string what = "<no FLANNException>";
    try {
        index_->knnSearchGPU(queries, indices, dists, kKnn, flann::SearchParams(), nullptr);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_EQ(what.rfind(kSearchText, 0), 0u) << what;
    EXPECT_TRUE(ends_with(what, status_suffix(e))) << what;
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(KMeansLaunchStatus, DirectSearchPadNamesThePendingStatus)
{
    DeviceArray<float> d_queries(kNumQueries * kVeclen);
    DeviceArray<int> d_indices(kNumQueries * kKnn);
    DeviceArray<float> d_dists(kNumQueries * kKnn);
    cudaMemcpy(d_queries.ptr, queries_.data(), queries_.size() * sizeof(float), cudaMemcpyHostToDevice);
    const cudaError_t e = leave_status_pending();
    ASSERT_NE(e, cudaSuccess);
    std::string what = "<no FLANNException>";
    try {
        index_->knnSearchGPUDirect(d_queries.ptr, d_indices.ptr, d_dists.ptr, kNumQueries, kKnn,
                                   flann::SearchParams(), nullptr);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_EQ(what, kPadText + status_suffix(e));
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(KMeansLaunchStatus, HostSearchPadNamesThePendingStatus)
{
    flann::Matrix<float> queries(queries_.data(), kNumQueries, kVeclen);
    std::vector<int> idx(kNumQueries * kKnn);
    std::vector<float> dst(kNumQueries * kKnn);
    flann::Matrix<int> indices(idx.data(), kNumQueries, kKnn);
    flann::Matrix<float> dists(dst.data(), kNumQueries, kKnn);
    const cudaError_t e = leave_status_pending();
    ASSERT_NE(e, cudaSuccess);
    std::string what = "<no FLANNException>";
    try {
        index_->knnSearchGPU(queries, indices, dists, static_cast<int>(kKnn), flann::SearchParams(), nullptr);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_EQ(what, kPadText + status_suffix(e));
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

int main(int argc, char** argv)
{
    // A pending-status arm's launch succeeds and the site then throws, which frees the search's
    // buffers with cudaFree. Run asynchronously, the kernel can still be reading them, faults,
    // and poisons the context for every later arm; synchronous launches let it finish first.
    // The runtime reads this when it initialises, so it is set before the first CUDA call.
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
