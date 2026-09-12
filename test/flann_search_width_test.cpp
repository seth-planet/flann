/**
 * @file flann_search_width_test.cpp
 * @brief Bounds on the hierarchical CUDA search width.
 *
 * The predicate is pure host code and needs neither a GPU nor HDF5. Its only other
 * coverage sits behind a GPU marker in the consuming repository, which does not run
 * when this library is built on its own.
 */

#include <cstring>

#include <gtest/gtest.h>

#include "flann/algorithms/cuda/hierarchical_search_width.h"

using flann::cuda::hierarchical_search_width_error;
using flann::cuda::kDefaultSearchWidth;

namespace
{

/// The shipped index shape: branching 32, four trees, k 16.
const char* shipped(int width) { return hierarchical_search_width_error(width, 16, 32, 4); }

/// A refusal that names its bound. Which bound fired is the part worth pinning: every
/// refusal is a non-null pointer, so a null check alone passes on the wrong one.
::testing::AssertionResult Names(const char* reason, const char* bound)
{
    if (reason == nullptr) {
        return ::testing::AssertionFailure() << "width accepted; expected a refusal naming \""
                                             << bound << "\"";
    }
    if (std::strstr(reason, bound) == nullptr) {
        return ::testing::AssertionFailure() << "refused with \"" << reason
                                             << "\", which does not name \"" << bound << "\"";
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

TEST(SearchWidth, DefaultRunsTheShippedIndex)
{
    EXPECT_EQ(128, kDefaultSearchWidth);
    EXPECT_EQ(nullptr, shipped(kDefaultSearchWidth));
}

TEST(SearchWidth, EveryPowerOfTwoFromBranchingToTheBlockLimitIsAccepted)
{
    for (int width = 32; width <= 1024; width *= 2) {
        EXPECT_EQ(nullptr, shipped(width)) << "width " << width;
    }
}

TEST(SearchWidth, RefusesAWidthTheBlockCannotHold)
{
    EXPECT_TRUE(Names(shipped(0), "must be positive"));
    EXPECT_TRUE(Names(shipped(-128), "must be positive"));
    EXPECT_TRUE(Names(shipped(100), "power of two"));
    EXPECT_TRUE(Names(shipped(2048), "1024-thread"));
}

TEST(SearchWidth, RefusesAWidthTheTreeWalkCannotUse)
{
    EXPECT_TRUE(Names(hierarchical_search_width_error(32, 16, 64, 4),
                      "below the index's branching"));

    EXPECT_TRUE(Names(hierarchical_search_width_error(128, 16, 48, 4), "not a multiple"));
    // A second width, because one point lets a bound narrowed to `width >= 128 && ...`
    // through: 64 with branching 48 leaves 16, and is the silent-wrong shape the bound
    // exists for. It takes a branching that is not a power of two to reach this at all --
    // a power-of-two branching no larger than a power-of-two width always divides it.
    EXPECT_TRUE(Names(hierarchical_search_width_error(64, 16, 48, 4), "not a multiple"));
    // A negative branching divides evenly whenever its magnitude does, so -32 at width 128
    // clears both the below-branching and the multiple-of bound; only this guard stops it.
    EXPECT_TRUE(Names(hierarchical_search_width_error(128, 16, -32, 4),
                      "branching factor is not positive"));

    EXPECT_TRUE(Names(hierarchical_search_width_error(128, 16, 0, 4),
                      "branching factor is not positive"));
}

TEST(SearchWidth, RefusesAWidthThatWouldLoseNeighboursOrTrees)
{
    EXPECT_TRUE(Names(hierarchical_search_width_error(32, 64, 32, 4), "below k"));
    EXPECT_TRUE(Names(hierarchical_search_width_error(32, 16, 32, 64),
                      "below the index's tree count"));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
