/**
 * @file flann_cuda_check_message_test.cpp
 * @brief CUDA_CHECK, CUDA_CHECK_LAST and the direct CUDABuffer throws name the status as "(cudaErrorName)".
 *
 * Callers classify a failed index by its message text, and the status name is the part that
 * separates a latched context from an exhausted card. No arm needs a device except where it
 * says so: cudaGetErrorName reads a table, and the runtime calls fail the same way with no card.
 * Each direct-throw arm takes the expected name from the same call it provokes, so it holds
 * on a box with a card and on one without.
 */

#include <cstdint>
#include <cstdio>
#include <string>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "flann/general.h"
#include "flann/algorithms/cuda/cuda_utils.h"

namespace
{

void check_illegal_address() { CUDA_CHECK(cudaErrorIllegalAddress); }

}  // namespace

TEST(CudaCheckMessage, CheckNamesTheStatusBeforeTheNumericCode)
{
    std::string what = "<no FLANNException>";
    try {
        check_illegal_address();
    } catch (const flann::FLANNException& e) {
        what = e.what();
    }
    EXPECT_NE(what.find(" (cudaErrorIllegalAddress)"), std::string::npos) << what;
    // The slot closes before the numeric code opens its own parenthesis.
    EXPECT_NE(what.find("(cudaErrorIllegalAddress) (error code: "), std::string::npos) << what;
}

TEST(CudaCheckMessage, CheckLastNamesTheStatusOfTheFailedCall)
{
    void* p = nullptr;
    // The expected name comes from the failing call, so the arm holds on a box with a card
    // (cudaErrorMemoryAllocation) and on one without (cudaErrorNoDevice).
    const cudaError_t e = cudaMalloc(&p, SIZE_MAX);
    ASSERT_NE(e, cudaSuccess);
    std::printf("CUDA_CHECK_LAST expected name: %s\n", cudaGetErrorName(e));
    std::string what = "<no FLANNException>";
    try {
        CUDA_CHECK_LAST();
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_NE(what.find(" (" + std::string(cudaGetErrorName(e)) + ")"), std::string::npos) << what;
}

TEST(CudaBufferThrowMessage, StreamConstructorNamesTheStatusOfCudaMallocAsync)
{
    void* p = nullptr;
    const cudaError_t e = cudaMallocAsync(&p, SIZE_MAX, nullptr);
    ASSERT_NE(e, cudaSuccess);
    cudaGetLastError();
    std::string what = "<no FLANNException>";
    try {
        flann::cuda::CUDABuffer<unsigned char> buf(SIZE_MAX, nullptr);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_NE(what.find(" (" + std::string(cudaGetErrorName(e)) + ")"), std::string::npos) << what;
}

TEST(CudaBufferThrowMessage, ResizeNamesTheStatusOfCudaMalloc)
{
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, SIZE_MAX);
    ASSERT_NE(e, cudaSuccess);
    cudaGetLastError();
    flann::cuda::CUDABuffer<unsigned char> buf;
    std::string what = "<no FLANNException>";
    try {
        buf.resize(SIZE_MAX);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_NE(what.find(" (" + std::string(cudaGetErrorName(e)) + ")"), std::string::npos) << what;
}

TEST(CudaBufferThrowMessage, UploadNamesTheStatusOfTheCopy)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) {
        GTEST_SKIP() << "a zero-byte copy succeeds when a device is visible";
    }
    cudaGetLastError();
    const cudaError_t e = cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToDevice);
    ASSERT_NE(e, cudaSuccess);
    cudaGetLastError();
    flann::cuda::CUDABuffer<unsigned char> buf;
    std::string what = "<no FLANNException>";
    try {
        buf.upload(nullptr, 0);
    } catch (const flann::FLANNException& ex) {
        what = ex.what();
    }
    EXPECT_NE(what.find(" (" + std::string(cudaGetErrorName(e)) + ")"), std::string::npos) << what;
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
