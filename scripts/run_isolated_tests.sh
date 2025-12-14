#!/bin/bash
# Run GPU save/load tests in isolation to catch test-order dependencies
#
# Usage:
#   ./scripts/run_isolated_tests.sh [BUILD_DIR]
#
# Arguments:
#   BUILD_DIR: Optional build directory (default: current directory)
#
# This script runs each GPU save/load test individually, cleaning up test files
# before each run. This catches bugs that only manifest when tests don't have
# stale files from previous runs.

set -e

BUILD_DIR="${1:-.}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test executables
HIER_TEST="$BUILD_DIR/test/flann_hierarchical_cuda_test"
KMEANS_TEST="$BUILD_DIR/test/flann_kmeans_cuda_test"

# Check that test executables exist
if [ ! -f "$HIER_TEST" ]; then
    echo -e "${RED}Error: $HIER_TEST not found${NC}"
    echo "Build with: cmake -DBUILD_CUDA_LIB=ON .. && make flann_hierarchical_cuda_test"
    exit 1
fi

if [ ! -f "$KMEANS_TEST" ]; then
    echo -e "${RED}Error: $KMEANS_TEST not found${NC}"
    echo "Build with: cmake -DBUILD_CUDA_LIB=ON .. && make flann_kmeans_cuda_test"
    exit 1
fi

# Tests to run in isolation
HIER_TESTS=(
    "HierarchicalCUDA_Brief100K.RegressionTest_GPUFormatLoadIsolation"
    "HierarchicalCUDA_Brief100K.TestGPUSaveLoad"
    "HierarchicalCUDA_Brief100K.TestConvertToGPUFormat"
    "HierarchicalCUDA_Brief100K.TestFormatDetection"
    "HierarchicalCUDA_Brief100K.TestGPUFormatSaveAfterRemove"
    "HierarchicalCUDA_Brief100K.TestCPUFormatLoadWithGPU"
    "HierarchicalCUDA_Brief100K.TestGPUFormatSaveAfterIncrementalAdd"
    "HierarchicalCUDA_Brief100K.TestGPUFormatFileIO"
    "HierarchicalCUDA_Brief100K.TestIsolation_GPUSaveLoadIndependent"
)

KMEANS_TESTS=(
    "KMeansCUDA_SIFT10K.RegressionTest_GPUFormatLoadIsolation"
    "KMeansCUDA_SIFT10K.TestGPUSaveLoad"
    "KMeansCUDA_SIFT10K.TestConvertToGPUFormat"
    "KMeansCUDA_SIFT10K.TestFormatDetection"
    "KMeansCUDA_SIFT10K.TestIsolation_GPUSaveLoadIndependent"
    "KMeansCUDA_SIFT100K.TestGPUFormatSaveAfterRemove"
    "KMeansCUDA_SIFT100K.TestCPUFormatLoadWithGPU"
    "KMeansCUDA_SIFT100K.TestGPUFormatSaveAfterIncrementalAdd"
)

PASSED=0
FAILED=0
FAILED_TESTS=()

cleanup_test_files() {
    # Clean up any stale test files before each run
    rm -f test_*.idx *.idx /tmp/flann_*.idx 2>/dev/null || true
}

run_test() {
    local executable="$1"
    local test_name="$2"

    echo -e "${YELLOW}--- Testing: $test_name ---${NC}"
    cleanup_test_files

    if $executable --gtest_filter="$test_name" 2>&1; then
        echo -e "${GREEN}PASSED: $test_name${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}FAILED: $test_name${NC}"
        ((FAILED++))
        FAILED_TESTS+=("$test_name")
        return 1
    fi
}

echo "=========================================="
echo " GPU Save/Load Isolation Tests"
echo "=========================================="
echo ""
echo "Build directory: $BUILD_DIR"
echo ""

# Run Hierarchical CUDA tests
echo ""
echo "=== Hierarchical CUDA Tests ==="
for test in "${HIER_TESTS[@]}"; do
    run_test "$HIER_TEST" "$test" || true
done

# Run K-Means CUDA tests
echo ""
echo "=== K-Means CUDA Tests ==="
for test in "${KMEANS_TESTS[@]}"; do
    run_test "$KMEANS_TEST" "$test" || true
done

# Final cleanup
cleanup_test_files

# Summary
echo ""
echo "=========================================="
echo " Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}- $test${NC}"
    done
    exit 1
fi

echo ""
echo -e "${GREEN}All isolation tests passed!${NC}"
exit 0
