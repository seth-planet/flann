/***********************************************************************
 * FLANN CPU vs GPU Index Benchmark
 *
 * Comprehensive benchmark comparing CPU-only, CPU-to-GPU, and GPU v2.0
 * index file loading and search performance.
 *
 * Usage:
 *   ./benchmark_cpu_gpu_comparison <cpu_index_file> <gpu_index_file> [options]
 *   ./benchmark_cpu_gpu_comparison --dataset=<h5_file> [options]
 *
 * Options:
 *   --k=N           Number of neighbors (default: 10)
 *   --checks=N      Search quality parameter (default: 128)
 *   --runs=N        Number of timed runs (default: 5)
 *   --warmup=N      Number of warmup runs (default: 2)
 *   --queries=N     Number of queries (default: 1000)
 *   --csv=FILE      Output CSV file
 *   --verbose       Verbose output
 *
 * Copyright 2024-2025. BSD License.
 **********************************************************************/

#ifndef FLANN_USE_CUDA
#error "This benchmark requires FLANN_USE_CUDA. Build with -DBUILD_CUDA_LIB=ON"
#endif

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>

using namespace flann;
using namespace std;
using namespace std::chrono;

// ============================================================================
// Configuration
// ============================================================================

struct BenchmarkConfig {
    string cpu_index_file;    // CPU format index file
    string gpu_index_file;    // GPU v2.0 format index file
    string dataset_file;      // HDF5 dataset (alternative to index files)
    string index_type;        // "hierarchical" or "kmeans" (auto-detected)
    int k;
    int checks;
    int num_runs;
    int warmup_runs;
    int num_queries;
    string csv_file;
    bool verbose;

    BenchmarkConfig() :
        index_type("auto"),
        k(10), checks(128),
        num_runs(5), warmup_runs(2),
        num_queries(1000),
        verbose(false) {}
};

// ============================================================================
// Timer and Statistics
// ============================================================================

class Timer {
    high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double, milli>>(end - start_time).count();
    }

    void reset() { start_time = high_resolution_clock::now(); }
};

struct Stats {
    double mean;
    double stddev;
    double min_val;
    double max_val;

    Stats() : mean(0), stddev(0), min_val(0), max_val(0) {}

    Stats(const vector<double>& values) {
        if (values.empty()) {
            mean = stddev = min_val = max_val = 0;
            return;
        }
        mean = 0;
        for (double v : values) mean += v;
        mean /= values.size();

        stddev = 0;
        for (double v : values) {
            double diff = v - mean;
            stddev += diff * diff;
        }
        stddev = sqrt(stddev / values.size());

        min_val = *min_element(values.begin(), values.end());
        max_val = *max_element(values.begin(), values.end());
    }

    string format_ms() const {
        ostringstream oss;
        oss << fixed << setprecision(1) << mean << " +/- " << setprecision(1) << stddev;
        return oss.str();
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

template<typename T>
double calculate_precision(const Matrix<T>& gt_indices,
                          const Matrix<T>& test_indices,
                          int k) {
    if (gt_indices.rows != test_indices.rows) return 0.0;

    int matches = 0;
    int total = gt_indices.rows * k;

    for (size_t i = 0; i < gt_indices.rows; i++) {
        for (int j = 0; j < k; j++) {
            for (int m = 0; m < k; m++) {
                if (gt_indices[i][j] == test_indices[i][m]) {
                    matches++;
                    break;
                }
            }
        }
    }
    return static_cast<double>(matches) / total;
}

// ============================================================================
// Scenario Result Structure
// ============================================================================

struct ScenarioResult {
    string name;
    Stats load_time_ms;
    Stats gpu_setup_time_ms;
    Stats total_ready_time_ms;
    Stats first_search_ms;
    Stats steady_search_ms;
    double precision;
    double queries_per_sec;
    size_t file_size_bytes;

    ScenarioResult() : precision(0), queries_per_sec(0), file_size_bytes(0) {}
};

// ============================================================================
// Binary Hierarchical Benchmark
// ============================================================================

void benchmark_hierarchical_binary(
    const string& cpu_file,
    const string& gpu_file,
    const BenchmarkConfig& config,
    vector<ScenarioResult>& results)
{
    cout << "\n" << string(80, '=') << endl;
    cout << "BINARY HIERARCHICAL BENCHMARK" << endl;
    cout << string(80, '=') << endl;

    // Load CPU index first to get dataset for queries
    cout << "\nLoading dataset from CPU index..." << flush;
    Timer timer;
    Matrix<unsigned char> empty_data;
    Index<Hamming<unsigned char>> base_index(empty_data, SavedIndexParams(cpu_file));
    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

    size_t num_points = base_index.size();
    size_t veclen = base_index.veclen();
    cout << "Dataset: " << num_points << " points x " << veclen << " dimensions" << endl;

    // Create query set from first N points
    size_t query_count = min((size_t)config.num_queries, num_points);
    Matrix<unsigned char> queries(new unsigned char[query_count * veclen], query_count, veclen);
    for (size_t i = 0; i < query_count; i++) {
        const unsigned char* point = base_index.getPoint(i);
        memcpy(queries[i], point, veclen);
    }

    // Compute ground truth
    cout << "Computing ground truth..." << flush;
    timer.reset();
    Matrix<size_t> gt_indices(new size_t[query_count * config.k], query_count, config.k);
    Matrix<unsigned int> gt_dists(new unsigned int[query_count * config.k], query_count, config.k);

    // Use brute force for ground truth
    Index<Hamming<unsigned char>> linear_index(
        Matrix<unsigned char>(const_cast<unsigned char*>(base_index.getPoint(0)), num_points, veclen),
        LinearIndexParams()
    );
    linear_index.buildIndex();
    linear_index.knnSearch(queries, gt_indices, gt_dists, config.k, SearchParams(-1));
    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

    // File sizes
    size_t cpu_file_size = getFileSize(cpu_file);
    size_t gpu_file_size = getFileSize(gpu_file);
    cout << "\nFile sizes:" << endl;
    cout << "  CPU format: " << formatFileSize(cpu_file_size) << endl;
    cout << "  GPU v2.0:   " << formatFileSize(gpu_file_size) << endl;
    cout << "  Ratio:      " << fixed << setprecision(2) << (double)gpu_file_size / cpu_file_size << "x" << endl;

    int total_runs = config.warmup_runs + config.num_runs;

    // ========================================================================
    // Scenario 1: CPU-only
    // ========================================================================
    cout << "\n--- Scenario 1: CPU-only ---" << endl;
    {
        ScenarioResult result;
        result.name = "CPU-only";
        result.file_size_bytes = cpu_file_size;

        vector<double> load_times, first_search, steady_search;

        for (int run = 0; run < total_runs; run++) {
            bool is_warmup = (run < config.warmup_runs);
            if (config.verbose) cout << "  " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;

            timer.reset();
            Matrix<unsigned char> empty;
            Index<Hamming<unsigned char>> index(empty, SavedIndexParams(cpu_file));
            double load_time = timer.elapsed_ms();

            if (!is_warmup) load_times.push_back(load_time);

            // Search performance (last run)
            if (run == total_runs - 1) {
                Matrix<size_t> indices(new size_t[query_count * config.k], query_count, config.k);
                Matrix<unsigned int> distances(new unsigned int[query_count * config.k], query_count, config.k);

                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                first_search.push_back(timer.elapsed_ms());

                for (int s = 0; s < 3; s++) {
                    timer.reset();
                    index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                    steady_search.push_back(timer.elapsed_ms());
                }

                result.precision = calculate_precision(gt_indices, indices, config.k);
                delete[] indices.ptr();
                delete[] distances.ptr();
            }
        }

        result.load_time_ms = Stats(load_times);
        result.gpu_setup_time_ms = Stats({0});  // No GPU setup
        result.total_ready_time_ms = result.load_time_ms;
        result.first_search_ms = Stats(first_search);
        result.steady_search_ms = Stats(steady_search);
        result.queries_per_sec = query_count / (result.steady_search_ms.mean / 1000.0);

        cout << "  Load: " << result.load_time_ms.format_ms() << " ms" << endl;
        cout << "  Search (steady): " << result.steady_search_ms.format_ms() << " ms" << endl;
        cout << "  Precision: " << fixed << setprecision(1) << (result.precision * 100) << "%" << endl;

        results.push_back(result);
    }

    // ========================================================================
    // Scenario 2: CPU-to-GPU
    // ========================================================================
    cout << "\n--- Scenario 2: CPU-to-GPU (in-memory conversion) ---" << endl;
    {
        ScenarioResult result;
        result.name = "CPU-to-GPU";
        result.file_size_bytes = cpu_file_size;

        vector<double> load_times, setup_times, total_times, first_search, steady_search;

        for (int run = 0; run < total_runs; run++) {
            bool is_warmup = (run < config.warmup_runs);
            if (config.verbose) cout << "  " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;

            timer.reset();
            Matrix<unsigned char> empty;
            Index<Hamming<unsigned char>> index(empty, SavedIndexParams(cpu_file));
            double load_time = timer.elapsed_ms();

            timer.reset();
            index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
            double setup_time = timer.elapsed_ms();

            if (!is_warmup) {
                load_times.push_back(load_time);
                setup_times.push_back(setup_time);
                total_times.push_back(load_time + setup_time);
            }

            if (run == total_runs - 1) {
                Matrix<size_t> indices(new size_t[query_count * config.k], query_count, config.k);
                Matrix<unsigned int> distances(new unsigned int[query_count * config.k], query_count, config.k);

                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                first_search.push_back(timer.elapsed_ms());

                for (int s = 0; s < 3; s++) {
                    timer.reset();
                    index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                    steady_search.push_back(timer.elapsed_ms());
                }

                result.precision = calculate_precision(gt_indices, indices, config.k);
                delete[] indices.ptr();
                delete[] distances.ptr();
            }
        }

        result.load_time_ms = Stats(load_times);
        result.gpu_setup_time_ms = Stats(setup_times);
        result.total_ready_time_ms = Stats(total_times);
        result.first_search_ms = Stats(first_search);
        result.steady_search_ms = Stats(steady_search);
        result.queries_per_sec = query_count / (result.steady_search_ms.mean / 1000.0);

        cout << "  Load: " << result.load_time_ms.format_ms() << " ms" << endl;
        cout << "  GPU Setup: " << result.gpu_setup_time_ms.format_ms() << " ms" << endl;
        cout << "  Total Ready: " << result.total_ready_time_ms.format_ms() << " ms" << endl;
        cout << "  Search (steady): " << result.steady_search_ms.format_ms() << " ms" << endl;
        cout << "  Precision: " << fixed << setprecision(1) << (result.precision * 100) << "%" << endl;

        results.push_back(result);
    }

    // ========================================================================
    // Scenario 3: GPU v2.0 Format
    // ========================================================================
    cout << "\n--- Scenario 3: GPU v2.0 Format ---" << endl;
    {
        ScenarioResult result;
        result.name = "GPU-v2.0";
        result.file_size_bytes = gpu_file_size;

        vector<double> load_times, setup_times, total_times, first_search, steady_search;

        for (int run = 0; run < total_runs; run++) {
            bool is_warmup = (run < config.warmup_runs);
            if (config.verbose) cout << "  " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;

            timer.reset();
            Matrix<unsigned char> empty;
            Index<Hamming<unsigned char>> index(empty, SavedIndexParams(gpu_file));
            double load_time = timer.elapsed_ms();

            timer.reset();
            index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
            double setup_time = timer.elapsed_ms();

            if (!is_warmup) {
                load_times.push_back(load_time);
                setup_times.push_back(setup_time);
                total_times.push_back(load_time + setup_time);
            }

            if (run == total_runs - 1) {
                Matrix<size_t> indices(new size_t[query_count * config.k], query_count, config.k);
                Matrix<unsigned int> distances(new unsigned int[query_count * config.k], query_count, config.k);

                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                first_search.push_back(timer.elapsed_ms());

                for (int s = 0; s < 3; s++) {
                    timer.reset();
                    index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                    steady_search.push_back(timer.elapsed_ms());
                }

                result.precision = calculate_precision(gt_indices, indices, config.k);
                delete[] indices.ptr();
                delete[] distances.ptr();
            }
        }

        result.load_time_ms = Stats(load_times);
        result.gpu_setup_time_ms = Stats(setup_times);
        result.total_ready_time_ms = Stats(total_times);
        result.first_search_ms = Stats(first_search);
        result.steady_search_ms = Stats(steady_search);
        result.queries_per_sec = query_count / (result.steady_search_ms.mean / 1000.0);

        cout << "  Load: " << result.load_time_ms.format_ms() << " ms" << endl;
        cout << "  GPU Setup: " << result.gpu_setup_time_ms.format_ms() << " ms" << endl;
        cout << "  Total Ready: " << result.total_ready_time_ms.format_ms() << " ms" << endl;
        cout << "  Search (steady): " << result.steady_search_ms.format_ms() << " ms" << endl;
        cout << "  Precision: " << fixed << setprecision(1) << (result.precision * 100) << "%" << endl;

        results.push_back(result);
    }

    // Cleanup
    delete[] queries.ptr();
    delete[] gt_indices.ptr();
    delete[] gt_dists.ptr();
}

// ============================================================================
// Float KMeans Benchmark (from HDF5)
// ============================================================================

void benchmark_kmeans_float(
    const string& dataset_file,
    const BenchmarkConfig& config,
    vector<ScenarioResult>& results)
{
    cout << "\n" << string(80, '=') << endl;
    cout << "FLOAT KMEANS BENCHMARK" << endl;
    cout << string(80, '=') << endl;

    // Load dataset from HDF5
    cout << "\nLoading dataset from " << dataset_file << "..." << flush;
    Timer timer;
    Matrix<float> dataset;
    load_from_file(dataset, dataset_file, "dataset");
    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;
    cout << "Dataset: " << dataset.rows << " points x " << dataset.cols << " dimensions" << endl;

    // Create query set
    size_t query_count = min((size_t)config.num_queries, dataset.rows);
    Matrix<float> queries(new float[query_count * dataset.cols], query_count, dataset.cols);
    for (size_t i = 0; i < query_count; i++) {
        memcpy(queries[i], dataset[i], dataset.cols * sizeof(float));
    }

    // Compute ground truth
    cout << "Computing ground truth..." << flush;
    timer.reset();
    Matrix<size_t> gt_indices(new size_t[query_count * config.k], query_count, config.k);
    Matrix<float> gt_dists(new float[query_count * config.k], query_count, config.k);
    Index<L2<float>> linear_index(dataset, LinearIndexParams());
    linear_index.buildIndex();
    linear_index.knnSearch(queries, gt_indices, gt_dists, config.k, SearchParams(-1));
    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

    // Build and save CPU index
    string cpu_file = "/tmp/flann_bench_kmeans_cpu.idx";
    string gpu_file = "/tmp/flann_bench_kmeans_gpu.idx";

    cout << "\nBuilding KMeans index..." << flush;
    timer.reset();
    KMeansCUDAIndexParams params(32, 11, FLANN_CENTERS_RANDOM, 0.2f);
    params["save_dataset"] = true;
    Index<L2<float>> build_index(dataset, params);
    build_index.buildIndex();
    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

    cout << "Saving CPU format..." << flush;
    timer.reset();
    build_index.save(cpu_file);
    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

    cout << "Converting to GPU format..." << flush;
    timer.reset();
    build_index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
    build_index.convertToGPUFormat();
    build_index.save(gpu_file);
    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

    size_t cpu_file_size = getFileSize(cpu_file);
    size_t gpu_file_size = getFileSize(gpu_file);
    cout << "\nFile sizes:" << endl;
    cout << "  CPU format: " << formatFileSize(cpu_file_size) << endl;
    cout << "  GPU v2.0:   " << formatFileSize(gpu_file_size) << endl;
    cout << "  Ratio:      " << fixed << setprecision(2) << (double)gpu_file_size / cpu_file_size << "x" << endl;

    int total_runs = config.warmup_runs + config.num_runs;

    // ========================================================================
    // Scenario 1: CPU-only
    // ========================================================================
    cout << "\n--- Scenario 1: CPU-only ---" << endl;
    {
        ScenarioResult result;
        result.name = "CPU-only (KMeans)";
        result.file_size_bytes = cpu_file_size;

        vector<double> load_times, first_search, steady_search;

        for (int run = 0; run < total_runs; run++) {
            bool is_warmup = (run < config.warmup_runs);
            if (config.verbose) cout << "  " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;

            timer.reset();
            Matrix<float> empty;
            Index<L2<float>> index(empty, SavedIndexParams(cpu_file));
            double load_time = timer.elapsed_ms();

            if (!is_warmup) load_times.push_back(load_time);

            if (run == total_runs - 1) {
                Matrix<size_t> indices(new size_t[query_count * config.k], query_count, config.k);
                Matrix<float> distances(new float[query_count * config.k], query_count, config.k);

                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                first_search.push_back(timer.elapsed_ms());

                for (int s = 0; s < 3; s++) {
                    timer.reset();
                    index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                    steady_search.push_back(timer.elapsed_ms());
                }

                result.precision = calculate_precision(gt_indices, indices, config.k);
                delete[] indices.ptr();
                delete[] distances.ptr();
            }
        }

        result.load_time_ms = Stats(load_times);
        result.gpu_setup_time_ms = Stats({0});
        result.total_ready_time_ms = result.load_time_ms;
        result.first_search_ms = Stats(first_search);
        result.steady_search_ms = Stats(steady_search);
        result.queries_per_sec = query_count / (result.steady_search_ms.mean / 1000.0);

        cout << "  Load: " << result.load_time_ms.format_ms() << " ms" << endl;
        cout << "  Search (steady): " << result.steady_search_ms.format_ms() << " ms" << endl;
        cout << "  Precision: " << fixed << setprecision(1) << (result.precision * 100) << "%" << endl;

        results.push_back(result);
    }

    // ========================================================================
    // Scenario 2: CPU-to-GPU
    // ========================================================================
    cout << "\n--- Scenario 2: CPU-to-GPU ---" << endl;
    {
        ScenarioResult result;
        result.name = "CPU-to-GPU (KMeans)";
        result.file_size_bytes = cpu_file_size;

        vector<double> load_times, setup_times, total_times, first_search, steady_search;

        for (int run = 0; run < total_runs; run++) {
            bool is_warmup = (run < config.warmup_runs);
            if (config.verbose) cout << "  " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;

            timer.reset();
            Matrix<float> empty;
            Index<L2<float>> index(empty, SavedIndexParams(cpu_file));
            double load_time = timer.elapsed_ms();

            timer.reset();
            index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
            double setup_time = timer.elapsed_ms();

            if (!is_warmup) {
                load_times.push_back(load_time);
                setup_times.push_back(setup_time);
                total_times.push_back(load_time + setup_time);
            }

            if (run == total_runs - 1) {
                Matrix<size_t> indices(new size_t[query_count * config.k], query_count, config.k);
                Matrix<float> distances(new float[query_count * config.k], query_count, config.k);

                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                first_search.push_back(timer.elapsed_ms());

                for (int s = 0; s < 3; s++) {
                    timer.reset();
                    index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                    steady_search.push_back(timer.elapsed_ms());
                }

                result.precision = calculate_precision(gt_indices, indices, config.k);
                delete[] indices.ptr();
                delete[] distances.ptr();
            }
        }

        result.load_time_ms = Stats(load_times);
        result.gpu_setup_time_ms = Stats(setup_times);
        result.total_ready_time_ms = Stats(total_times);
        result.first_search_ms = Stats(first_search);
        result.steady_search_ms = Stats(steady_search);
        result.queries_per_sec = query_count / (result.steady_search_ms.mean / 1000.0);

        cout << "  Load: " << result.load_time_ms.format_ms() << " ms" << endl;
        cout << "  GPU Setup: " << result.gpu_setup_time_ms.format_ms() << " ms" << endl;
        cout << "  Total Ready: " << result.total_ready_time_ms.format_ms() << " ms" << endl;
        cout << "  Search (steady): " << result.steady_search_ms.format_ms() << " ms" << endl;
        cout << "  Precision: " << fixed << setprecision(1) << (result.precision * 100) << "%" << endl;

        results.push_back(result);
    }

    // ========================================================================
    // Scenario 3: GPU v2.0
    // ========================================================================
    cout << "\n--- Scenario 3: GPU v2.0 Format ---" << endl;
    {
        ScenarioResult result;
        result.name = "GPU-v2.0 (KMeans)";
        result.file_size_bytes = gpu_file_size;

        vector<double> load_times, setup_times, total_times, first_search, steady_search;

        for (int run = 0; run < total_runs; run++) {
            bool is_warmup = (run < config.warmup_runs);
            if (config.verbose) cout << "  " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;

            timer.reset();
            Matrix<float> empty;
            Index<L2<float>> index(empty, SavedIndexParams(gpu_file));
            double load_time = timer.elapsed_ms();

            timer.reset();
            index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
            double setup_time = timer.elapsed_ms();

            if (!is_warmup) {
                load_times.push_back(load_time);
                setup_times.push_back(setup_time);
                total_times.push_back(load_time + setup_time);
            }

            if (run == total_runs - 1) {
                Matrix<size_t> indices(new size_t[query_count * config.k], query_count, config.k);
                Matrix<float> distances(new float[query_count * config.k], query_count, config.k);

                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                first_search.push_back(timer.elapsed_ms());

                for (int s = 0; s < 3; s++) {
                    timer.reset();
                    index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                    steady_search.push_back(timer.elapsed_ms());
                }

                result.precision = calculate_precision(gt_indices, indices, config.k);
                delete[] indices.ptr();
                delete[] distances.ptr();
            }
        }

        result.load_time_ms = Stats(load_times);
        result.gpu_setup_time_ms = Stats(setup_times);
        result.total_ready_time_ms = Stats(total_times);
        result.first_search_ms = Stats(first_search);
        result.steady_search_ms = Stats(steady_search);
        result.queries_per_sec = query_count / (result.steady_search_ms.mean / 1000.0);

        cout << "  Load: " << result.load_time_ms.format_ms() << " ms" << endl;
        cout << "  GPU Setup: " << result.gpu_setup_time_ms.format_ms() << " ms" << endl;
        cout << "  Total Ready: " << result.total_ready_time_ms.format_ms() << " ms" << endl;
        cout << "  Search (steady): " << result.steady_search_ms.format_ms() << " ms" << endl;
        cout << "  Precision: " << fixed << setprecision(1) << (result.precision * 100) << "%" << endl;

        results.push_back(result);
    }

    // Cleanup
    delete[] queries.ptr();
    delete[] gt_indices.ptr();
    delete[] gt_dists.ptr();
    delete[] dataset.ptr();
    remove(cpu_file.c_str());
    remove(gpu_file.c_str());
}

// ============================================================================
// Report Generation
// ============================================================================

void print_report(const vector<ScenarioResult>& hier_results,
                  const vector<ScenarioResult>& kmeans_results,
                  const BenchmarkConfig& config) {
    cout << "\n" << string(80, '=') << endl;
    cout << "COMPREHENSIVE BENCHMARK REPORT" << endl;
    cout << string(80, '=') << endl;

    auto print_table = [&](const string& title, const vector<ScenarioResult>& results) {
        if (results.empty()) return;

        cout << "\n" << title << endl;
        cout << string(80, '-') << endl;

        // Loading timing table
        cout << "\nLOADING + GPU SETUP TIMING (ms)" << endl;
        cout << left << setw(20) << "Scenario"
             << right << setw(15) << "Load"
             << setw(15) << "GPU Setup"
             << setw(15) << "Total Ready" << endl;
        cout << string(65, '-') << endl;

        for (const auto& r : results) {
            cout << left << setw(20) << r.name
                 << right << setw(15) << r.load_time_ms.format_ms()
                 << setw(15) << r.gpu_setup_time_ms.format_ms()
                 << setw(15) << r.total_ready_time_ms.format_ms() << endl;
        }

        // Search performance table
        cout << "\nSEARCH PERFORMANCE" << endl;
        cout << left << setw(20) << "Scenario"
             << right << setw(15) << "First (ms)"
             << setw(15) << "Steady (ms)"
             << setw(12) << "Precision"
             << setw(15) << "Throughput" << endl;
        cout << string(77, '-') << endl;

        for (const auto& r : results) {
            ostringstream prec, qps;
            prec << fixed << setprecision(1) << (r.precision * 100) << "%";
            qps << fixed << setprecision(0) << r.queries_per_sec << " q/s";

            cout << left << setw(20) << r.name
                 << right << setw(15) << r.first_search_ms.format_ms()
                 << setw(15) << r.steady_search_ms.format_ms()
                 << setw(12) << prec.str()
                 << setw(15) << qps.str() << endl;
        }

        // Speedup analysis
        if (results.size() >= 3) {
            double load_speedup = results[1].total_ready_time_ms.mean / results[2].total_ready_time_ms.mean;
            double search_speedup = results[0].steady_search_ms.mean / results[2].steady_search_ms.mean;
            double time_saved = results[1].total_ready_time_ms.mean - results[2].total_ready_time_ms.mean;

            cout << "\nSPEEDUP ANALYSIS" << endl;
            cout << "  GPU v2.0 vs CPU-to-GPU load speedup: " << fixed << setprecision(2) << load_speedup << "x" << endl;
            cout << "  GPU vs CPU search speedup: " << fixed << setprecision(1) << search_speedup << "x" << endl;
            cout << "  Time saved per load: " << fixed << setprecision(1) << time_saved << " ms" << endl;
        }
    };

    print_table("=== BINARY HIERARCHICAL (Real-World Data) ===", hier_results);
    print_table("=== FLOAT KMEANS (SIFT Dataset) ===", kmeans_results);

    // Summary table
    cout << "\n" << string(80, '=') << endl;
    cout << "SUMMARY" << endl;
    cout << string(80, '=') << endl;

    if (!hier_results.empty() && !kmeans_results.empty()) {
        cout << left << setw(25) << ""
             << right << setw(20) << "Binary/Hier"
             << setw(20) << "Float/KMeans" << endl;
        cout << string(65, '-') << endl;

        if (hier_results.size() >= 3 && kmeans_results.size() >= 3) {
            double hier_load_speedup = hier_results[1].total_ready_time_ms.mean / hier_results[2].total_ready_time_ms.mean;
            double kmeans_load_speedup = kmeans_results[1].total_ready_time_ms.mean / kmeans_results[2].total_ready_time_ms.mean;
            double hier_search_speedup = hier_results[0].steady_search_ms.mean / hier_results[2].steady_search_ms.mean;
            double kmeans_search_speedup = kmeans_results[0].steady_search_ms.mean / kmeans_results[2].steady_search_ms.mean;

            ostringstream hier_ls, kmeans_ls, hier_ss, kmeans_ss;
            hier_ls << fixed << setprecision(2) << hier_load_speedup << "x";
            kmeans_ls << fixed << setprecision(2) << kmeans_load_speedup << "x";
            hier_ss << fixed << setprecision(1) << hier_search_speedup << "x";
            kmeans_ss << fixed << setprecision(1) << kmeans_search_speedup << "x";

            cout << left << setw(25) << "Load Speedup (v2.0):"
                 << right << setw(20) << hier_ls.str()
                 << setw(20) << kmeans_ls.str() << endl;
            cout << left << setw(25) << "Search Speedup (GPU):"
                 << right << setw(20) << hier_ss.str()
                 << setw(20) << kmeans_ss.str() << endl;

            ostringstream hier_prec, kmeans_prec;
            hier_prec << fixed << setprecision(1) << (hier_results[2].precision * 100) << "%";
            kmeans_prec << fixed << setprecision(1) << (kmeans_results[2].precision * 100) << "%";

            cout << left << setw(25) << "GPU Precision:"
                 << right << setw(20) << hier_prec.str()
                 << setw(20) << kmeans_prec.str() << endl;
        }
    }
}

void export_csv(const vector<ScenarioResult>& hier_results,
                const vector<ScenarioResult>& kmeans_results,
                const string& filename) {
    ofstream csv(filename);
    if (!csv.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return;
    }

    csv << "test_type,scenario,file_mb,load_ms,load_std,gpu_setup_ms,gpu_setup_std,"
        << "total_ready_ms,total_ready_std,first_search_ms,steady_search_ms,"
        << "precision,queries_per_sec" << endl;

    auto write_results = [&](const string& test_type, const vector<ScenarioResult>& results) {
        for (const auto& r : results) {
            csv << test_type << ","
                << r.name << ","
                << fixed << setprecision(3) << (r.file_size_bytes / (1024.0 * 1024.0)) << ","
                << setprecision(2) << r.load_time_ms.mean << ","
                << r.load_time_ms.stddev << ","
                << r.gpu_setup_time_ms.mean << ","
                << r.gpu_setup_time_ms.stddev << ","
                << r.total_ready_time_ms.mean << ","
                << r.total_ready_time_ms.stddev << ","
                << r.first_search_ms.mean << ","
                << r.steady_search_ms.mean << ","
                << setprecision(4) << r.precision << ","
                << setprecision(1) << r.queries_per_sec << endl;
        }
    };

    write_results("hierarchical_binary", hier_results);
    write_results("kmeans_float", kmeans_results);

    csv.close();
    cout << "\nCSV results saved to: " << filename << endl;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    cerr << "FLANN CPU vs GPU Index Benchmark\n\n"
         << "Usage:\n"
         << "  " << prog << " <cpu_index> <gpu_index> [options]  - Benchmark binary Hierarchical\n"
         << "  " << prog << " --dataset=<h5_file> [options]      - Benchmark float KMeans\n"
         << "  " << prog << " --all [options]                    - Run all benchmarks\n"
         << "\nOptions:\n"
         << "  --k=N         Number of neighbors (default: 10)\n"
         << "  --checks=N    Search quality (default: 128)\n"
         << "  --runs=N      Timed runs (default: 5)\n"
         << "  --warmup=N    Warmup runs (default: 2)\n"
         << "  --queries=N   Number of queries (default: 1000)\n"
         << "  --csv=FILE    Output CSV file\n"
         << "  --verbose     Verbose output\n";
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    bool run_all = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg.find("--k=") == 0) {
            config.k = stoi(arg.substr(4));
        } else if (arg.find("--checks=") == 0) {
            config.checks = stoi(arg.substr(9));
        } else if (arg.find("--runs=") == 0) {
            config.num_runs = stoi(arg.substr(7));
        } else if (arg.find("--warmup=") == 0) {
            config.warmup_runs = stoi(arg.substr(9));
        } else if (arg.find("--queries=") == 0) {
            config.num_queries = stoi(arg.substr(10));
        } else if (arg.find("--csv=") == 0) {
            config.csv_file = arg.substr(6);
        } else if (arg.find("--dataset=") == 0) {
            config.dataset_file = arg.substr(10);
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--all") {
            run_all = true;
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

    cout << "FLANN CPU vs GPU Index Benchmark" << endl;
    cout << "Configuration: k=" << config.k << ", checks=" << config.checks
         << ", runs=" << config.num_runs << ", warmup=" << config.warmup_runs
         << ", queries=" << config.num_queries << endl;

    vector<ScenarioResult> hier_results, kmeans_results;

    // Run binary Hierarchical benchmark
    if (!config.cpu_index_file.empty() && !config.gpu_index_file.empty()) {
        benchmark_hierarchical_binary(config.cpu_index_file, config.gpu_index_file, config, hier_results);
    } else if (run_all) {
        // Use default paths for --all mode
        string cpu_file = "/home/seth/pl/planet_common/cplusplus/cuda_pipeline/test_data/flann/ref_airbus_gpkg-060-034-0289_flann.db";
        string gpu_file = "/home/seth/pl/planet_common/cplusplus/cuda_pipeline/test_data/flann/ref_airbus_gpkg-060-034-0289_flann.gpu_v2.idx";
        benchmark_hierarchical_binary(cpu_file, gpu_file, config, hier_results);
    }

    // Run float KMeans benchmark
    if (!config.dataset_file.empty()) {
        benchmark_kmeans_float(config.dataset_file, config, kmeans_results);
    } else if (run_all) {
        benchmark_kmeans_float("datasets/sift100K.h5", config, kmeans_results);
    }

    // Generate report
    print_report(hier_results, kmeans_results, config);

    // Export CSV
    if (!config.csv_file.empty()) {
        export_csv(hier_results, kmeans_results, config.csv_file);
    }

    return 0;
}
