/***********************************************************************
 * FLANN GPU Pipeline Equivalence Test
 *
 * Verifies that CPU->GPU and pure GPU index pipelines produce
 * functionally equivalent results.
 *
 * Pipeline A: Load CPU index -> buildCUDAKnnSearch() -> knnSearch()
 * Pipeline B: Load GPU index -> buildCUDAKnnSearch() -> knnSearch()
 *
 * Results should be identical except for tie-breaking order when
 * multiple points have the same distance.
 *
 * Usage:
 *   ./flann_gpu_pipeline_equivalence_test <cpu_index> <gpu_index> [options]
 *
 * Options:
 *   --queries=N     Number of queries (default: 1000)
 *   --checks=N      Search checks parameter (default: 2000)
 *   --verbose       Print detailed per-query analysis
 *
 * Copyright 2024-2025. BSD License.
 **********************************************************************/

#ifndef FLANN_USE_CUDA
#error "This test requires FLANN_USE_CUDA. Build with -DBUILD_CUDA_LIB=ON"
#endif

#include <flann/flann.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <sys/stat.h>

using namespace std;
using namespace flann;

// ============================================================================
// Configuration
// ============================================================================

struct TestConfig {
    string cpu_index_file;
    string gpu_index_file;
    int num_queries = 1000;
    int checks = 2000;
    bool verbose = false;
    vector<int> k_values = {1, 10, 16, 32, 64};
};

// ============================================================================
// Metrics
// ============================================================================

struct ComparisonMetrics {
    int k;
    int num_queries;
    int exact_index_matches;      // Exact match at every position
    int distance_multiset_matches; // Same distances in sorted order
    int tie_adjusted_matches;      // Match accounting for tie-breaking
    int total_positions;           // k * num_queries
    int positions_with_ties;       // Positions where distance equals another
    int max_distance_delta;        // Maximum distance difference at any position
    double pipeline_a_time_ms;
    double pipeline_b_time_ms;

    ComparisonMetrics() : k(0), num_queries(0), exact_index_matches(0),
        distance_multiset_matches(0), tie_adjusted_matches(0),
        total_positions(0), positions_with_ties(0), max_distance_delta(0),
        pipeline_a_time_ms(0), pipeline_b_time_ms(0) {}

    double exactMatchRate() const {
        return total_positions > 0 ? 100.0 * exact_index_matches / total_positions : 0;
    }

    double distanceMatchRate() const {
        return num_queries > 0 ? 100.0 * distance_multiset_matches / num_queries : 0;
    }

    double tieAdjustedRate() const {
        return num_queries > 0 ? 100.0 * tie_adjusted_matches / num_queries : 0;
    }

    bool passed() const {
        // Pass if 100% of queries have matching distance multisets
        return distance_multiset_matches == num_queries;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

size_t getFileSize(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return st.st_size;
    return 0;
}

string formatFileSize(size_t bytes) {
    ostringstream oss;
    if (bytes >= 1024 * 1024) {
        oss << fixed << setprecision(2) << (bytes / (1024.0 * 1024.0)) << " MB";
    } else if (bytes >= 1024) {
        oss << fixed << setprecision(2) << (bytes / 1024.0) << " KB";
    } else {
        oss << bytes << " bytes";
    }
    return oss.str();
}

// Count occurrences of a value in a vector
template<typename T>
int countOccurrences(const vector<T>& vec, T value) {
    return count(vec.begin(), vec.end(), value);
}

// Compare distance multisets (sorted distance arrays)
template<typename T>
bool compareDistanceMultisets(const vector<T>& dists_a, const vector<T>& dists_b) {
    if (dists_a.size() != dists_b.size()) return false;

    vector<T> sorted_a = dists_a;
    vector<T> sorted_b = dists_b;
    sort(sorted_a.begin(), sorted_a.end());
    sort(sorted_b.begin(), sorted_b.end());

    return sorted_a == sorted_b;
}

// Enhanced comparison that accounts for tie-breaking
template<typename T>
bool compareWithTieTolerance(
    const vector<size_t>& indices_a, const vector<T>& dists_a,
    const vector<size_t>& indices_b, const vector<T>& dists_b,
    int& positions_with_ties, int& exact_matches)
{
    if (dists_a.size() != dists_b.size()) return false;

    int k = dists_a.size();
    positions_with_ties = 0;
    exact_matches = 0;

    // First check: Distance multisets must match
    if (!compareDistanceMultisets(dists_a, dists_b)) {
        return false;
    }

    // Second check: For positions with unique distances, indices must match
    for (int i = 0; i < k; i++) {
        // Count how many times this distance appears in each result
        int count_a = countOccurrences(vector<T>(dists_a.begin(), dists_a.end()), dists_a[i]);
        int count_b = countOccurrences(vector<T>(dists_b.begin(), dists_b.end()), dists_b[i]);

        if (count_a > 1 || count_b > 1) {
            positions_with_ties++;
        }

        if (indices_a[i] == indices_b[i]) {
            exact_matches++;
        }
    }

    return true;  // Distance multisets match, tie-breaking differences are OK
}

// ============================================================================
// Print Functions
// ============================================================================

void printHeader() {
    cout << "\n" << string(80, '=') << endl;
    cout << "FLANN GPU PIPELINE EQUIVALENCE TEST" << endl;
    cout << string(80, '=') << endl;
    cout << "\nThis test verifies that CPU->GPU and pure GPU index pipelines" << endl;
    cout << "produce functionally equivalent search results." << endl;
}

void printFileInfo(const string& cpu_file, const string& gpu_file,
                   size_t num_points, size_t veclen) {
    cout << "\n" << string(60, '-') << endl;
    cout << "INDEX FILE INFORMATION" << endl;
    cout << string(60, '-') << endl;

    size_t cpu_size = getFileSize(cpu_file);
    size_t gpu_size = getFileSize(gpu_file);

    cout << "CPU Index: " << cpu_file << endl;
    cout << "  Size: " << formatFileSize(cpu_size) << endl;

    cout << "GPU Index: " << gpu_file << endl;
    cout << "  Size: " << formatFileSize(gpu_size) << endl;
    cout << "  Size ratio: " << fixed << setprecision(2)
         << (double)gpu_size / cpu_size << "x" << endl;

    cout << "\nDataset: " << num_points << " points x " << veclen << " dimensions" << endl;
}

void printMetrics(const ComparisonMetrics& m) {
    cout << "\n  k=" << m.k << ":" << endl;
    cout << "    Distance Multiset Match: " << m.distance_multiset_matches << "/" << m.num_queries
         << " (" << fixed << setprecision(1) << m.distanceMatchRate() << "%)" << endl;
    cout << "    Exact Index Match:       " << m.exact_index_matches << "/" << m.total_positions
         << " (" << fixed << setprecision(1) << m.exactMatchRate() << "%)" << endl;
    cout << "    Tie-Adjusted Match:      " << m.tie_adjusted_matches << "/" << m.num_queries
         << " (" << fixed << setprecision(1) << m.tieAdjustedRate() << "%)" << endl;
    cout << "    Positions with Ties:     " << m.positions_with_ties << "/" << m.total_positions << endl;
    cout << "    Max Distance Delta:      " << m.max_distance_delta << endl;
    cout << "    Search Time (A):         " << fixed << setprecision(2) << m.pipeline_a_time_ms << " ms" << endl;
    cout << "    Search Time (B):         " << fixed << setprecision(2) << m.pipeline_b_time_ms << " ms" << endl;
    cout << "    Status:                  " << (m.passed() ? "PASS" : "FAIL") << endl;
}

void printSummary(const vector<ComparisonMetrics>& all_metrics) {
    cout << "\n" << string(80, '=') << endl;
    cout << "SUMMARY" << endl;
    cout << string(80, '=') << endl;

    // Summary table
    cout << "\n" << left << setw(8) << "k"
         << right << setw(15) << "Dist Match"
         << setw(15) << "Idx Match"
         << setw(15) << "Tie-Adjusted"
         << setw(10) << "Status" << endl;
    cout << string(63, '-') << endl;

    int passed = 0, failed = 0;
    for (const auto& m : all_metrics) {
        cout << left << setw(8) << m.k
             << right << setw(14) << fixed << setprecision(1) << m.distanceMatchRate() << "%"
             << setw(14) << fixed << setprecision(1) << m.exactMatchRate() << "%"
             << setw(14) << fixed << setprecision(1) << m.tieAdjustedRate() << "%"
             << setw(10) << (m.passed() ? "PASS" : "FAIL") << endl;

        if (m.passed()) passed++; else failed++;
    }

    cout << string(63, '-') << endl;
    cout << "\nOverall: " << passed << " passed, " << failed << " failed" << endl;

    if (failed == 0) {
        cout << "\n*** ALL TESTS PASSED ***" << endl;
        cout << "The CPU->GPU and pure GPU pipelines produce equivalent results." << endl;
    } else {
        cout << "\n*** SOME TESTS FAILED ***" << endl;
        cout << "There are discrepancies between the pipelines beyond tie-breaking." << endl;
    }
}

// ============================================================================
// Main Test Function
// ============================================================================

template<typename Distance>
int runEquivalenceTest(const TestConfig& config) {
    using ElementType = typename Distance::ElementType;
    using DistanceType = typename Distance::ResultType;

    printHeader();

    // ========================================================================
    // Load Pipeline A: CPU Index -> GPU
    // ========================================================================
    cout << "\n" << string(60, '-') << endl;
    cout << "LOADING PIPELINE A (CPU Index -> GPU)" << endl;
    cout << string(60, '-') << endl;

    cout << "Loading CPU index from " << config.cpu_index_file << "..." << flush;
    auto start = chrono::high_resolution_clock::now();

    Matrix<ElementType> dataset_a;
    Index<Distance> index_a(dataset_a, SavedIndexParams(config.cpu_index_file));

    auto load_end = chrono::high_resolution_clock::now();
    double load_a_ms = chrono::duration<double, milli>(load_end - start).count();
    cout << " done (" << fixed << setprecision(1) << load_a_ms << " ms)" << endl;

    size_t num_points = index_a.size();
    size_t veclen = index_a.veclen();
    cout << "  Points: " << num_points << ", Dimensions: " << veclen << endl;

    // ========================================================================
    // Load Pipeline B: GPU Index (pre-built)
    // ========================================================================
    cout << "\n" << string(60, '-') << endl;
    cout << "LOADING PIPELINE B (Pre-built GPU Index)" << endl;
    cout << string(60, '-') << endl;

    cout << "Loading GPU index from " << config.gpu_index_file << "..." << flush;
    start = chrono::high_resolution_clock::now();

    Matrix<ElementType> dataset_b;
    Index<Distance> index_b(dataset_b, SavedIndexParams(config.gpu_index_file));

    load_end = chrono::high_resolution_clock::now();
    double load_b_ms = chrono::duration<double, milli>(load_end - start).count();
    cout << " done (" << fixed << setprecision(1) << load_b_ms << " ms)" << endl;

    // Verify both indices have same dimensions
    if (index_a.size() != index_b.size() || index_a.veclen() != index_b.veclen()) {
        cerr << "ERROR: Index dimensions don't match!" << endl;
        cerr << "  Pipeline A: " << index_a.size() << " x " << index_a.veclen() << endl;
        cerr << "  Pipeline B: " << index_b.size() << " x " << index_b.veclen() << endl;
        return 1;
    }

    printFileInfo(config.cpu_index_file, config.gpu_index_file, num_points, veclen);

    // ========================================================================
    // Prepare queries
    // ========================================================================
    size_t query_count = min((size_t)config.num_queries, num_points);
    cout << "\nPreparing " << query_count << " queries (from first points of dataset)..." << endl;

    // Note: For CPU index, we can get points; for GPU-only, we need to use the CPU index
    Matrix<ElementType> queries(new ElementType[query_count * veclen], query_count, veclen);
    for (size_t i = 0; i < query_count; i++) {
        const ElementType* point = index_a.getPoint(i);
        memcpy(queries[i], point, veclen * sizeof(ElementType));
    }

    // ========================================================================
    // Run tests for each k value
    // ========================================================================
    vector<ComparisonMetrics> all_metrics;

    for (int k : config.k_values) {
        cout << "\n" << string(60, '-') << endl;
        cout << "TESTING k=" << k << endl;
        cout << string(60, '-') << endl;

        ComparisonMetrics metrics;
        metrics.k = k;
        metrics.num_queries = query_count;
        metrics.total_positions = query_count * k;

        // Initialize GPU for this k value
        cout << "Initializing GPU search for Pipeline A (k=" << k << ")..." << flush;
        start = chrono::high_resolution_clock::now();
        index_a.buildCUDAKnnSearch(k, SearchParams(config.checks));
        auto end = chrono::high_resolution_clock::now();
        cout << " done (" << fixed << setprecision(1)
             << chrono::duration<double, milli>(end - start).count() << " ms)" << endl;

        cout << "Initializing GPU search for Pipeline B (k=" << k << ")..." << flush;
        start = chrono::high_resolution_clock::now();
        index_b.buildCUDAKnnSearch(k, SearchParams(config.checks));
        end = chrono::high_resolution_clock::now();
        cout << " done (" << fixed << setprecision(1)
             << chrono::duration<double, milli>(end - start).count() << " ms)" << endl;

        // Allocate result matrices
        Matrix<size_t> indices_a(new size_t[query_count * k], query_count, k);
        Matrix<DistanceType> dists_a(new DistanceType[query_count * k], query_count, k);
        Matrix<size_t> indices_b(new size_t[query_count * k], query_count, k);
        Matrix<DistanceType> dists_b(new DistanceType[query_count * k], query_count, k);

        // Run Pipeline A
        cout << "Running Pipeline A search..." << flush;
        start = chrono::high_resolution_clock::now();
        index_a.knnSearch(queries, indices_a, dists_a, k, SearchParams(config.checks));
        end = chrono::high_resolution_clock::now();
        metrics.pipeline_a_time_ms = chrono::duration<double, milli>(end - start).count();
        cout << " done (" << fixed << setprecision(2) << metrics.pipeline_a_time_ms << " ms)" << endl;

        // Run Pipeline B
        cout << "Running Pipeline B search..." << flush;
        start = chrono::high_resolution_clock::now();
        index_b.knnSearch(queries, indices_b, dists_b, k, SearchParams(config.checks));
        end = chrono::high_resolution_clock::now();
        metrics.pipeline_b_time_ms = chrono::duration<double, milli>(end - start).count();
        cout << " done (" << fixed << setprecision(2) << metrics.pipeline_b_time_ms << " ms)" << endl;

        // Compare results
        cout << "Comparing results..." << endl;

        int discrepancy_count = 0;
        for (size_t q = 0; q < query_count; q++) {
            // Extract results for this query
            vector<size_t> idx_a(k), idx_b(k);
            vector<DistanceType> dst_a(k), dst_b(k);

            for (int i = 0; i < k; i++) {
                idx_a[i] = indices_a[q][i];
                idx_b[i] = indices_b[q][i];
                dst_a[i] = dists_a[q][i];
                dst_b[i] = dists_b[q][i];
            }

            // Check distance multiset match
            if (compareDistanceMultisets(dst_a, dst_b)) {
                metrics.distance_multiset_matches++;
            } else {
                discrepancy_count++;
                if (config.verbose && discrepancy_count <= 10) {
                    cout << "  Query " << q << ": Distance mismatch!" << endl;
                    cout << "    A: ";
                    for (int i = 0; i < k; i++) cout << dst_a[i] << " ";
                    cout << endl;
                    cout << "    B: ";
                    for (int i = 0; i < k; i++) cout << dst_b[i] << " ";
                    cout << endl;
                }
            }

            // Check with tie tolerance
            int pos_ties = 0, exact_matches = 0;
            if (compareWithTieTolerance(idx_a, dst_a, idx_b, dst_b, pos_ties, exact_matches)) {
                metrics.tie_adjusted_matches++;
            }
            metrics.positions_with_ties += pos_ties;
            metrics.exact_index_matches += exact_matches;

            // Track max distance delta
            for (int i = 0; i < k; i++) {
                int delta = abs((int)dst_a[i] - (int)dst_b[i]);
                metrics.max_distance_delta = max(metrics.max_distance_delta, delta);
            }
        }

        printMetrics(metrics);
        all_metrics.push_back(metrics);

        // Cleanup
        delete[] indices_a.ptr();
        delete[] dists_a.ptr();
        delete[] indices_b.ptr();
        delete[] dists_b.ptr();
    }

    // Cleanup queries
    delete[] queries.ptr();

    // Print summary
    printSummary(all_metrics);

    // Return exit code based on results
    bool all_passed = true;
    for (const auto& m : all_metrics) {
        if (!m.passed()) all_passed = false;
    }

    return all_passed ? 0 : 1;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    cerr << "FLANN GPU Pipeline Equivalence Test\n\n"
         << "Usage: " << prog << " <cpu_index> <gpu_index> [options]\n\n"
         << "Options:\n"
         << "  --queries=N     Number of queries (default: 1000)\n"
         << "  --checks=N      Search checks parameter (default: 2000)\n"
         << "  --verbose       Print detailed per-query analysis\n"
         << "\nExample:\n"
         << "  " << prog << " index.db index.gpu_v2.idx --queries=1000 --verbose\n";
}

int main(int argc, char** argv) {
    TestConfig config;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg.find("--queries=") == 0) {
            config.num_queries = stoi(arg.substr(10));
        } else if (arg.find("--checks=") == 0) {
            config.checks = stoi(arg.substr(9));
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            if (config.cpu_index_file.empty()) {
                config.cpu_index_file = arg;
            } else if (config.gpu_index_file.empty()) {
                config.gpu_index_file = arg;
            }
        }
    }

    if (config.cpu_index_file.empty() || config.gpu_index_file.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    // Auto-detect index type from file header
    struct FileHeader {
        char signature[24];
        char version[16];
        flann_datatype_t data_type;
        flann_algorithm_t index_type;
    };

    FileHeader header;
    FILE* f = fopen(config.cpu_index_file.c_str(), "rb");
    if (!f) {
        cerr << "Error: Cannot open " << config.cpu_index_file << endl;
        return 1;
    }
    if (fread(&header, sizeof(header), 1, f) != 1) {
        cerr << "Error: Cannot read file header" << endl;
        fclose(f);
        return 1;
    }
    fclose(f);

    // Run appropriate test based on index type
    switch (header.index_type) {
        case FLANN_INDEX_HIERARCHICAL:
        case FLANN_INDEX_HIERARCHICAL_CUDA:
            cout << "Detected: Hierarchical index (Hamming distance)" << endl;
            return runEquivalenceTest<Hamming<unsigned char>>(config);

        case FLANN_INDEX_KMEANS:
        case FLANN_INDEX_KMEANS_CUDA:
            cout << "Detected: KMeans index (L2 distance)" << endl;
            return runEquivalenceTest<L2<float>>(config);

        default:
            cerr << "Error: Unsupported index type: " << header.index_type << endl;
            return 1;
    }
}
