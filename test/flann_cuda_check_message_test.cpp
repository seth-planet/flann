/**
 * @file flann_cuda_check_message_test.cpp
 * @brief CUDA_CHECK and CUDA_CHECK_LAST name the status as "(cudaErrorName)".
 *
 * Callers classify a failed index by its message text, and the status name is the part that
 * separates a latched context from an exhausted card. Neither arm needs a device:
 * cudaGetErrorName reads a table, and the runtime call in the second arm fails the same way
 * with no card.
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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
