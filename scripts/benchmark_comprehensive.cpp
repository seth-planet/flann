/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024-2025 FLANN GPU Benchmark Suite
 *
 * Comprehensive benchmark comparing CPU, OpenCL, and CUDA implementations
 * for large-scale binary descriptor datasets (1M points, 512-bit vectors)
 **********************************************************************/

#include <flann/flann.h>
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

using namespace flann;
using namespace std;
using namespace std::chrono;

// ============================================================================
// Configuration
// ============================================================================

struct BenchmarkConfig {
    string dataset_file;
    int k;                    // Number of nearest neighbors
    int checks;               // Search accuracy parameter
    int branching;            // Tree branching factor
    int trees;                // Number of trees (hierarchical)
    int leaf_size;            // Leaf node size
    int num_runs;             // Number of runs for averaging
    int warmup_runs;          // Warmup runs (discarded from timing)
    bool run_cpu;
    bool run_opencl;
    bool run_cuda;
    bool run_kmeans;          // Also test K-Means algorithms
    bool is_binary;           // Binary descriptors (Hamming) or float (L2)
    float cb_index;           // K-Means cb_index parameter
    string output_prefix;

    BenchmarkConfig() :
        dataset_file("datasets/binary1M_512bit.h5"),
        k(16), checks(128), branching(32), trees(4),
        leaf_size(100), num_runs(5), warmup_runs(1),
        run_cpu(true), run_opencl(true), run_cuda(true), run_kmeans(false),
        is_binary(true), cb_index(0.2f),
        output_prefix("benchmark_results/results") {}
};

// ============================================================================
// Timing Utilities
// ============================================================================

class Timer {
    high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(high_resolution_clock::now()) {}

    double elapsed() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double>>(end - start_time).count();
    }

    void reset() {
        start_time = high_resolution_clock::now();
    }
};

// ============================================================================
// Statistics Utilities
// ============================================================================

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

    string format() const {
        ostringstream oss;
        oss << fixed << setprecision(3) << mean << "±" << setprecision(3) << stddev;
        return oss.str();
    }
};

// ============================================================================
// Results Structure
// ============================================================================

struct BenchmarkResult {
    string implementation;
    string algorithm;

    Stats build_time;
    Stats gpu_upload_time;
    Stats search_time;
    double total_time;

    double precision;        // Recall@k
    double queries_per_sec;

    size_t cpu_memory_mb;
    size_t gpu_memory_mb;

    BenchmarkResult() : total_time(0), precision(0), queries_per_sec(0),
                        cpu_memory_mb(0), gpu_memory_mb(0) {}
};

// ============================================================================
// Precision Calculation
// ============================================================================

template<typename T>
double calculate_precision(const Matrix<T>& gt_indices,
                          const Matrix<T>& test_indices,
                          int k)
{
    if (gt_indices.rows != test_indices.rows) {
        cerr << "Error: Ground truth and test results have different number of queries" << endl;
        return 0.0;
    }

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
// Ground Truth Computation
// ============================================================================

template<typename Distance>
void compute_ground_truth(
    const Matrix<typename Distance::ElementType>& dataset,
    const Matrix<typename Distance::ElementType>& queries,
    Matrix<size_t>& gt_indices,
    Matrix<typename Distance::ResultType>& gt_dists,
    int k)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "Computing Ground Truth (Linear Brute-Force Search)" << endl;
    cout << string(70, '=') << endl;

    Timer timer;

    cout << "Building linear index..." << flush;
    Index<Distance> index(dataset, LinearIndexParams());
    index.buildIndex();
    cout << " done" << endl;

    // Allocate memory for ground truth
    gt_indices = Matrix<size_t>(new size_t[queries.rows * k], queries.rows, k);
    gt_dists = Matrix<typename Distance::ResultType>(
        new typename Distance::ResultType[queries.rows * k], queries.rows, k);

    cout << "Searching " << queries.rows << " queries for " << k << " nearest neighbors..." << flush;
    timer.reset();
    index.knnSearch(queries, gt_indices, gt_dists, k, SearchParams(-1));
    double elapsed = timer.elapsed();

    cout << " done" << endl;
    cout << "Ground truth computed in " << fixed << setprecision(2) << elapsed << " seconds" << endl;
    cout << "Throughput: " << fixed << setprecision(1) << (queries.rows / elapsed) << " queries/sec" << endl;
}

// ============================================================================
// CPU Hierarchical Benchmark
// ============================================================================

BenchmarkResult benchmark_cpu_hierarchical(
    const Matrix<unsigned char>& dataset,
    const Matrix<unsigned char>& queries,
    const Matrix<size_t>& gt_indices,
    const BenchmarkConfig& config)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "CPU Hierarchical Clustering Benchmark" << endl;
    cout << string(70, '=') << endl;

    BenchmarkResult result;
    result.implementation = "CPU";
    result.algorithm = "Hierarchical";

    vector<double> build_times, search_times;

    HierarchicalClusteringIndexParams params(
        config.branching,
        FLANN_CENTERS_RANDOM,
        config.trees,
        config.leaf_size
    );

    for (int run = 0; run < config.num_runs; run++) {
        cout << "\nRun " << (run + 1) << "/" << config.num_runs << ":" << endl;

        Timer timer;

        // Build index
        cout << "  Building index..." << flush;
        timer.reset();
        Index<Hamming<unsigned char>> index(dataset, params);
        index.buildIndex();
        double build_time = timer.elapsed();
        build_times.push_back(build_time);
        cout << " " << fixed << setprecision(3) << build_time << "s" << endl;

        // Search
        Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
        Matrix<unsigned int> distances(new unsigned int[queries.rows * config.k], queries.rows, config.k);

        cout << "  Searching..." << flush;
        timer.reset();
        index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
        double search_time = timer.elapsed();
        search_times.push_back(search_time);
        cout << " " << fixed << setprecision(3) << search_time << "s" << endl;

        // Calculate precision (last run only)
        if (run == config.num_runs - 1) {
            result.precision = calculate_precision(gt_indices, indices, config.k);
            cout << "  Precision: " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;
        }

        delete[] indices.ptr();
        delete[] distances.ptr();
    }

    result.build_time = Stats(build_times);
    result.search_time = Stats(search_times);
    result.gpu_upload_time = Stats();
    result.total_time = result.build_time.mean + result.search_time.mean;
    result.queries_per_sec = queries.rows / result.search_time.mean;
    result.cpu_memory_mb = (dataset.rows * dataset.cols) / (1024 * 1024);
    result.gpu_memory_mb = 0;

    cout << "\nSummary:" << endl;
    cout << "  Build time:    " << result.build_time.format() << "s" << endl;
    cout << "  Search time:   " << result.search_time.format() << "s" << endl;
    cout << "  Throughput:    " << fixed << setprecision(1) << result.queries_per_sec << " queries/sec" << endl;
    cout << "  Precision:     " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;

    return result;
}

// ============================================================================
// OpenCL Hierarchical Benchmark
// ============================================================================

#ifdef FLANN_USE_OPENCL
BenchmarkResult benchmark_opencl_hierarchical(
    const Matrix<unsigned char>& dataset,
    const Matrix<unsigned char>& queries,
    const Matrix<size_t>& gt_indices,
    const BenchmarkConfig& config)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "OpenCL Hierarchical Clustering Benchmark" << endl;
    cout << string(70, '=') << endl;

    BenchmarkResult result;
    result.implementation = "OpenCL";
    result.algorithm = "Hierarchical";

    vector<double> build_times, upload_times, search_times;

    HierarchicalClusteringOpenCLIndexParams params(
        config.branching,
        FLANN_CENTERS_RANDOM,
        config.trees,
        config.leaf_size
    );

    for (int run = 0; run < config.num_runs; run++) {
        cout << "\nRun " << (run + 1) << "/" << config.num_runs << ":" << endl;

        Timer timer;

        // Build index (CPU)
        cout << "  Building index (CPU)..." << flush;
        timer.reset();
        Index<Hamming<unsigned char>> index(dataset, params);
        index.buildIndex();
        double build_time = timer.elapsed();
        build_times.push_back(build_time);
        cout << " " << fixed << setprecision(3) << build_time << "s" << endl;

        // GPU upload
        cout << "  Uploading to GPU..." << flush;
        timer.reset();
        index.buildCLKnnSearch(config.k, SearchParams(config.checks));
        double upload_time = timer.elapsed();
        upload_times.push_back(upload_time);
        cout << " " << fixed << setprecision(3) << upload_time << "s" << endl;

        // Search (GPU)
        Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
        Matrix<unsigned int> distances(new unsigned int[queries.rows * config.k], queries.rows, config.k);

        cout << "  Searching (GPU)..." << flush;
        timer.reset();
        index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
        double search_time = timer.elapsed();
        search_times.push_back(search_time);
        cout << " " << fixed << setprecision(3) << search_time << "s" << endl;

        // Calculate precision (last run only)
        if (run == config.num_runs - 1) {
            result.precision = calculate_precision(gt_indices, indices, config.k);
            cout << "  Precision: " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;
        }

        delete[] indices.ptr();
        delete[] distances.ptr();
    }

    result.build_time = Stats(build_times);
    result.gpu_upload_time = Stats(upload_times);
    result.search_time = Stats(search_times);
    result.total_time = result.build_time.mean + result.gpu_upload_time.mean + result.search_time.mean;
    result.queries_per_sec = queries.rows / result.search_time.mean;
    result.cpu_memory_mb = (dataset.rows * dataset.cols) / (1024 * 1024);
    result.gpu_memory_mb = (dataset.rows * dataset.cols + queries.rows * queries.cols + dataset.rows * 64) / (1024 * 1024);

    cout << "\nSummary:" << endl;
    cout << "  Build time:    " << result.build_time.format() << "s" << endl;
    cout << "  Upload time:   " << result.gpu_upload_time.format() << "s" << endl;
    cout << "  Search time:   " << result.search_time.format() << "s" << endl;
    cout << "  Throughput:    " << fixed << setprecision(1) << result.queries_per_sec << " queries/sec" << endl;
    cout << "  Precision:     " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;

    return result;
}
#endif

// ============================================================================
// CUDA Hierarchical Benchmark
// ============================================================================

#ifdef FLANN_USE_CUDA
BenchmarkResult benchmark_cuda_hierarchical(
    const Matrix<unsigned char>& dataset,
    const Matrix<unsigned char>& queries,
    const Matrix<size_t>& gt_indices,
    const BenchmarkConfig& config)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "CUDA Hierarchical Clustering Benchmark" << endl;
    cout << string(70, '=') << endl;

    BenchmarkResult result;
    result.implementation = "CUDA";
    result.algorithm = "Hierarchical";

    vector<double> build_times, upload_times, search_times;

    HierarchicalCUDAIndexParams params(
        config.branching,
        FLANN_CENTERS_RANDOM,
        config.trees,
        config.leaf_size
    );

    for (int run = 0; run < config.num_runs; run++) {
        cout << "\nRun " << (run + 1) << "/" << config.num_runs << ":" << endl;

        Timer timer;

        // Build index (CPU)
        cout << "  Building index (CPU)..." << flush;
        timer.reset();
        Index<Hamming<unsigned char>> index(dataset, params);
        index.buildIndex();
        double build_time = timer.elapsed();
        build_times.push_back(build_time);
        cout << " " << fixed << setprecision(3) << build_time << "s" << endl;

        // GPU upload
        cout << "  Uploading to GPU..." << flush;
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        double upload_time = timer.elapsed();
        upload_times.push_back(upload_time);
        cout << " " << fixed << setprecision(3) << upload_time << "s" << endl;

        // Search (GPU)
        Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
        Matrix<unsigned int> distances(new unsigned int[queries.rows * config.k], queries.rows, config.k);

        cout << "  Searching (GPU)..." << flush;
        timer.reset();
        index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
        double search_time = timer.elapsed();
        search_times.push_back(search_time);
        cout << " " << fixed << setprecision(3) << search_time << "s" << endl;

        // Calculate precision (last run only)
        if (run == config.num_runs - 1) {
            result.precision = calculate_precision(gt_indices, indices, config.k);
            cout << "  Precision: " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;
        }

        delete[] indices.ptr();
        delete[] distances.ptr();
    }

    result.build_time = Stats(build_times);
    result.gpu_upload_time = Stats(upload_times);
    result.search_time = Stats(search_times);
    result.total_time = result.build_time.mean + result.gpu_upload_time.mean + result.search_time.mean;
    result.queries_per_sec = queries.rows / result.search_time.mean;
    result.cpu_memory_mb = (dataset.rows * dataset.cols) / (1024 * 1024);
    result.gpu_memory_mb = (dataset.rows * dataset.cols + queries.rows * queries.cols + dataset.rows * 64) / (1024 * 1024);

    cout << "\nSummary:" << endl;
    cout << "  Build time:    " << result.build_time.format() << "s" << endl;
    cout << "  Upload time:   " << result.gpu_upload_time.format() << "s" << endl;
    cout << "  Search time:   " << result.search_time.format() << "s" << endl;
    cout << "  Throughput:    " << fixed << setprecision(1) << result.queries_per_sec << " queries/sec" << endl;
    cout << "  Precision:     " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;

    return result;
}
#endif

// ============================================================================
// K-Means CPU Benchmark (Float/L2)
// ============================================================================

BenchmarkResult benchmark_cpu_kmeans(
    const Matrix<float>& dataset,
    const Matrix<float>& queries,
    const Matrix<size_t>& gt_indices,
    const BenchmarkConfig& config)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "CPU K-Means Tree Benchmark" << endl;
    cout << string(70, '=') << endl;

    BenchmarkResult result;
    result.implementation = "CPU";
    result.algorithm = "K-Means";

    vector<double> build_times, search_times;

    KMeansIndexParams params(
        config.branching,
        11,  // iterations
        FLANN_CENTERS_RANDOM,
        config.cb_index
    );

    int total_runs = config.warmup_runs + config.num_runs;
    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        if (is_warmup) {
            cout << "\nWarmup run " << (run + 1) << "/" << config.warmup_runs << " (discarded):" << endl;
        } else {
            cout << "\nRun " << (run - config.warmup_runs + 1) << "/" << config.num_runs << ":" << endl;
        }

        Timer timer;

        // Build index
        cout << "  Building index..." << flush;
        timer.reset();
        Index<L2<float>> index(dataset, params);
        index.buildIndex();
        double build_time = timer.elapsed();
        if (!is_warmup) build_times.push_back(build_time);
        cout << " " << fixed << setprecision(3) << build_time << "s" << endl;

        // Search
        Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
        Matrix<float> distances(new float[queries.rows * config.k], queries.rows, config.k);

        cout << "  Searching..." << flush;
        timer.reset();
        index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
        double search_time = timer.elapsed();
        if (!is_warmup) search_times.push_back(search_time);
        cout << " " << fixed << setprecision(3) << search_time << "s" << endl;

        // Calculate precision (last run only)
        if (run == total_runs - 1) {
            result.precision = calculate_precision(gt_indices, indices, config.k);
            cout << "  Precision: " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;
        }

        delete[] indices.ptr();
        delete[] distances.ptr();
    }

    result.build_time = Stats(build_times);
    result.search_time = Stats(search_times);
    result.gpu_upload_time = Stats();
    result.total_time = result.build_time.mean + result.search_time.mean;
    result.queries_per_sec = queries.rows / result.search_time.mean;
    result.cpu_memory_mb = (dataset.rows * dataset.cols * sizeof(float)) / (1024 * 1024);
    result.gpu_memory_mb = 0;

    cout << "\nSummary:" << endl;
    cout << "  Build time:    " << result.build_time.format() << "s" << endl;
    cout << "  Search time:   " << result.search_time.format() << "s" << endl;
    cout << "  Per-query:     " << fixed << setprecision(2) << (result.search_time.mean * 1e6 / queries.rows) << " µs" << endl;
    cout << "  Throughput:    " << fixed << setprecision(1) << result.queries_per_sec << " queries/sec" << endl;
    cout << "  Precision:     " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;

    return result;
}

// ============================================================================
// K-Means OpenCL Benchmark
// ============================================================================

#ifdef FLANN_USE_OPENCL
BenchmarkResult benchmark_opencl_kmeans(
    const Matrix<float>& dataset,
    const Matrix<float>& queries,
    const Matrix<size_t>& gt_indices,
    const BenchmarkConfig& config)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "OpenCL K-Means Tree Benchmark" << endl;
    cout << string(70, '=') << endl;

    BenchmarkResult result;
    result.implementation = "OpenCL";
    result.algorithm = "K-Means";

    vector<double> build_times, upload_times, search_times;

    KMeansOpenCLIndexParams params(
        config.branching,
        11,  // iterations
        FLANN_CENTERS_RANDOM,
        config.cb_index
    );

    int total_runs = config.warmup_runs + config.num_runs;
    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        if (is_warmup) {
            cout << "\nWarmup run " << (run + 1) << "/" << config.warmup_runs << " (discarded):" << endl;
        } else {
            cout << "\nRun " << (run - config.warmup_runs + 1) << "/" << config.num_runs << ":" << endl;
        }

        Timer timer;

        // Build index (CPU)
        cout << "  Building index (CPU)..." << flush;
        timer.reset();
        Index<L2<float>> index(dataset, params);
        index.buildIndex();
        double build_time = timer.elapsed();
        if (!is_warmup) build_times.push_back(build_time);
        cout << " " << fixed << setprecision(3) << build_time << "s" << endl;

        // GPU upload
        cout << "  Uploading to GPU..." << flush;
        timer.reset();
        index.buildCLKnnSearch(config.k, SearchParams(config.checks));
        double upload_time = timer.elapsed();
        if (!is_warmup) upload_times.push_back(upload_time);
        cout << " " << fixed << setprecision(3) << upload_time << "s" << endl;

        // Search (GPU)
        Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
        Matrix<float> distances(new float[queries.rows * config.k], queries.rows, config.k);

        cout << "  Searching (GPU)..." << flush;
        timer.reset();
        index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
        double search_time = timer.elapsed();
        if (!is_warmup) search_times.push_back(search_time);
        cout << " " << fixed << setprecision(3) << search_time << "s" << endl;

        // Calculate precision (last run only)
        if (run == total_runs - 1) {
            result.precision = calculate_precision(gt_indices, indices, config.k);
            cout << "  Precision: " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;
        }

        delete[] indices.ptr();
        delete[] distances.ptr();
    }

    result.build_time = Stats(build_times);
    result.gpu_upload_time = Stats(upload_times);
    result.search_time = Stats(search_times);
    result.total_time = result.build_time.mean + result.gpu_upload_time.mean + result.search_time.mean;
    result.queries_per_sec = queries.rows / result.search_time.mean;
    result.cpu_memory_mb = (dataset.rows * dataset.cols * sizeof(float)) / (1024 * 1024);
    result.gpu_memory_mb = result.cpu_memory_mb * 2;  // Estimate

    cout << "\nSummary:" << endl;
    cout << "  Build time:    " << result.build_time.format() << "s" << endl;
    cout << "  Upload time:   " << result.gpu_upload_time.format() << "s" << endl;
    cout << "  Search time:   " << result.search_time.format() << "s" << endl;
    cout << "  Per-query:     " << fixed << setprecision(2) << (result.search_time.mean * 1e6 / queries.rows) << " µs" << endl;
    cout << "  Throughput:    " << fixed << setprecision(1) << result.queries_per_sec << " queries/sec" << endl;
    cout << "  Precision:     " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;

    return result;
}
#endif

// ============================================================================
// K-Means CUDA Benchmark
// ============================================================================

#ifdef FLANN_USE_CUDA
BenchmarkResult benchmark_cuda_kmeans(
    const Matrix<float>& dataset,
    const Matrix<float>& queries,
    const Matrix<size_t>& gt_indices,
    const BenchmarkConfig& config)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "CUDA K-Means Tree Benchmark" << endl;
    cout << string(70, '=') << endl;

    BenchmarkResult result;
    result.implementation = "CUDA";
    result.algorithm = "K-Means";

    vector<double> build_times, upload_times, search_times;

    KMeansCUDAIndexParams params(
        config.branching,
        11,  // iterations
        FLANN_CENTERS_RANDOM,
        config.cb_index
    );

    int total_runs = config.warmup_runs + config.num_runs;
    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        if (is_warmup) {
            cout << "\nWarmup run " << (run + 1) << "/" << config.warmup_runs << " (discarded):" << endl;
        } else {
            cout << "\nRun " << (run - config.warmup_runs + 1) << "/" << config.num_runs << ":" << endl;
        }

        Timer timer;

        // Build index (CPU)
        cout << "  Building index (CPU)..." << flush;
        timer.reset();
        Index<L2<float>> index(dataset, params);
        index.buildIndex();
        double build_time = timer.elapsed();
        if (!is_warmup) build_times.push_back(build_time);
        cout << " " << fixed << setprecision(3) << build_time << "s" << endl;

        // GPU upload
        cout << "  Uploading to GPU..." << flush;
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        double upload_time = timer.elapsed();
        if (!is_warmup) upload_times.push_back(upload_time);
        cout << " " << fixed << setprecision(3) << upload_time << "s" << endl;

        // Search (GPU)
        Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
        Matrix<float> distances(new float[queries.rows * config.k], queries.rows, config.k);

        cout << "  Searching (GPU)..." << flush;
        timer.reset();
        index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
        double search_time = timer.elapsed();
        if (!is_warmup) search_times.push_back(search_time);
        cout << " " << fixed << setprecision(3) << search_time << "s" << endl;

        // Calculate precision (last run only)
        if (run == total_runs - 1) {
            result.precision = calculate_precision(gt_indices, indices, config.k);
            cout << "  Precision: " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;
        }

        delete[] indices.ptr();
        delete[] distances.ptr();
    }

    result.build_time = Stats(build_times);
    result.gpu_upload_time = Stats(upload_times);
    result.search_time = Stats(search_times);
    result.total_time = result.build_time.mean + result.gpu_upload_time.mean + result.search_time.mean;
    result.queries_per_sec = queries.rows / result.search_time.mean;
    result.cpu_memory_mb = (dataset.rows * dataset.cols * sizeof(float)) / (1024 * 1024);
    result.gpu_memory_mb = result.cpu_memory_mb * 2;  // Estimate

    cout << "\nSummary:" << endl;
    cout << "  Build time:    " << result.build_time.format() << "s" << endl;
    cout << "  Upload time:   " << result.gpu_upload_time.format() << "s" << endl;
    cout << "  Search time:   " << result.search_time.format() << "s" << endl;
    cout << "  Per-query:     " << fixed << setprecision(2) << (result.search_time.mean * 1e6 / queries.rows) << " µs" << endl;
    cout << "  Throughput:    " << fixed << setprecision(1) << result.queries_per_sec << " queries/sec" << endl;
    cout << "  Precision:     " << fixed << setprecision(2) << (result.precision * 100.0) << "%" << endl;

    return result;
}
#endif

// ============================================================================
// Results Export
// ============================================================================

void export_results_csv(const vector<BenchmarkResult>& results,
                        const string& filename)
{
    ofstream csv(filename);
    if (!csv.is_open()) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return;
    }

    csv << "Implementation,Algorithm,BuildTime(s),BuildStd,UploadTime(s),UploadStd,SearchTime(s),SearchStd,TotalTime(s),Precision(%),QueriesPerSec,CPUMemory(MB),GPUMemory(MB)" << endl;

    for (const auto& r : results) {
        csv << r.implementation << ","
            << r.algorithm << ","
            << fixed << setprecision(3) << r.build_time.mean << ","
            << fixed << setprecision(3) << r.build_time.stddev << ","
            << fixed << setprecision(3) << r.gpu_upload_time.mean << ","
            << fixed << setprecision(3) << r.gpu_upload_time.stddev << ","
            << fixed << setprecision(3) << r.search_time.mean << ","
            << fixed << setprecision(3) << r.search_time.stddev << ","
            << fixed << setprecision(3) << r.total_time << ","
            << fixed << setprecision(2) << (r.precision * 100.0) << ","
            << fixed << setprecision(1) << r.queries_per_sec << ","
            << r.cpu_memory_mb << ","
            << r.gpu_memory_mb << endl;
    }

    csv.close();
    cout << "\nCSV results saved to: " << filename << endl;
}

void print_summary_table(const vector<BenchmarkResult>& results)
{
    cout << "\n" << string(70, '=') << endl;
    cout << "BENCHMARK RESULTS SUMMARY" << endl;
    cout << string(70, '=') << endl;

    // Print table header
    cout << endl;
    cout << left << setw(12) << "Impl"
         << setw(14) << "Algorithm"
         << right << setw(12) << "Build(s)"
         << setw(12) << "Upload(s)"
         << setw(12) << "Search(s)"
         << setw(12) << "Total(s)"
         << setw(10) << "Prec(%)"
         << setw(10) << "Q/sec" << endl;
    cout << string(92, '-') << endl;

    for (const auto& r : results) {
        cout << left << setw(12) << r.implementation
             << setw(14) << r.algorithm
             << right << setw(12) << r.build_time.format()
             << setw(12) << r.gpu_upload_time.format()
             << setw(12) << r.search_time.format()
             << setw(12) << fixed << setprecision(2) << r.total_time
             << setw(10) << fixed << setprecision(1) << (r.precision * 100.0)
             << setw(10) << fixed << setprecision(0) << r.queries_per_sec << endl;
    }

    // Speedup analysis
    if (results.size() > 1) {
        cout << "\n" << string(70, '-') << endl;
        cout << "SPEEDUP ANALYSIS (vs CPU)" << endl;
        cout << string(70, '-') << endl;

        double cpu_search = results[0].search_time.mean;
        for (size_t i = 1; i < results.size(); i++) {
            double speedup = cpu_search / results[i].search_time.mean;
            cout << results[i].implementation << " " << results[i].algorithm << ": "
                 << fixed << setprecision(2) << speedup << "x faster" << endl;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program) {
    cout << "Usage: " << program << " [dataset.h5] [options]\n\n"
         << "Options:\n"
         << "  --k=N          Number of nearest neighbors (default: 16)\n"
         << "  --checks=N     Search accuracy parameter (default: 128)\n"
         << "  --runs=N       Number of timed runs (default: 5)\n"
         << "  --warmup=N     Number of warmup runs (default: 1)\n"
         << "  --branching=N  Tree branching factor (default: 32)\n"
         << "  --cb-index=F   K-Means cb_index parameter (default: 0.2)\n"
         << "  --kmeans       Run K-Means benchmarks (float/L2 data)\n"
         << "  --cpu-only     Only run CPU benchmarks\n"
         << "  --gpu-only     Only run GPU benchmarks\n"
         << "  --cuda-only    Only run CUDA benchmarks\n"
         << "  --opencl-only  Only run OpenCL benchmarks\n"
         << "  --help         Show this help message\n\n"
         << "Examples:\n"
         << "  " << program << " datasets/binary1M_512bit.h5 --k=16 --runs=5\n"
         << "  " << program << " datasets/sift100K.h5 --kmeans --k=5 --checks=96\n";
}

int main(int argc, char** argv)
{
    BenchmarkConfig config;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg.find(".h5") != string::npos) {
            config.dataset_file = arg;
            // Auto-detect binary vs float
            if (arg.find("sift") != string::npos || arg.find("float") != string::npos) {
                config.is_binary = false;
                config.run_kmeans = true;  // SIFT implies K-Means
            }
        }
        else if (arg.find("--k=") == 0) {
            config.k = stoi(arg.substr(4));
        }
        else if (arg.find("--checks=") == 0) {
            config.checks = stoi(arg.substr(9));
        }
        else if (arg.find("--runs=") == 0) {
            config.num_runs = stoi(arg.substr(7));
        }
        else if (arg.find("--warmup=") == 0) {
            config.warmup_runs = stoi(arg.substr(9));
        }
        else if (arg.find("--branching=") == 0) {
            config.branching = stoi(arg.substr(12));
        }
        else if (arg.find("--cb-index=") == 0) {
            config.cb_index = stof(arg.substr(11));
        }
        else if (arg == "--kmeans") {
            config.run_kmeans = true;
            config.is_binary = false;
        }
        else if (arg == "--cpu-only") {
            config.run_opencl = config.run_cuda = false;
        }
        else if (arg == "--gpu-only") {
            config.run_cpu = false;
        }
        else if (arg == "--cuda-only") {
            config.run_cpu = config.run_opencl = false;
            config.run_cuda = true;
        }
        else if (arg == "--opencl-only") {
            config.run_cpu = config.run_cuda = false;
            config.run_opencl = true;
        }
    }

    cout << string(70, '=') << endl;
    cout << "FLANN Comprehensive GPU Benchmark Suite" << endl;
    if (config.run_kmeans) {
        cout << "K-Means Tree (L2/Float Distance)" << endl;
    } else {
        cout << "Hierarchical Clustering (Hamming/Binary Distance)" << endl;
    }
    cout << string(70, '=') << endl;

    cout << "\nConfiguration:" << endl;
    cout << "  Dataset:       " << config.dataset_file << endl;
    cout << "  Mode:          " << (config.run_kmeans ? "K-Means (L2<float>)" : "Hierarchical (Hamming)") << endl;
    cout << "  k:             " << config.k << endl;
    cout << "  checks:        " << config.checks << endl;
    cout << "  branching:     " << config.branching << endl;
    if (config.run_kmeans) {
        cout << "  cb_index:      " << config.cb_index << endl;
    } else {
        cout << "  trees:         " << config.trees << endl;
    }
    cout << "  warmup_runs:   " << config.warmup_runs << endl;
    cout << "  num_runs:      " << config.num_runs << endl;
    cout << "  Backends:      "
         << (config.run_cpu ? "CPU " : "")
         << (config.run_opencl ? "OpenCL " : "")
         << (config.run_cuda ? "CUDA " : "") << endl;

    vector<BenchmarkResult> results;

    if (config.run_kmeans) {
        // ============================================
        // K-Means mode: Load float data
        // ============================================
        cout << "\nLoading dataset (float)..." << flush;
        Matrix<float> dataset;
        Matrix<float> queries;

        try {
            load_from_file(dataset, config.dataset_file, "dataset");
            load_from_file(queries, config.dataset_file, "query");
        }
        catch (const exception& e) {
            cerr << "\nError loading dataset: " << e.what() << endl;
            return 1;
        }

        cout << " done" << endl;
        cout << "  Index:  " << dataset.rows << " x " << dataset.cols << " floats" << endl;
        cout << "  Query:  " << queries.rows << " x " << queries.cols << " floats" << endl;

        // Compute ground truth
        Matrix<size_t> gt_indices;
        Matrix<float> gt_dists;
        compute_ground_truth<L2<float>>(dataset, queries, gt_indices, gt_dists, config.k);

        // Run K-Means benchmarks
        if (config.run_cpu) {
            try {
                results.push_back(benchmark_cpu_kmeans(dataset, queries, gt_indices, config));
            }
            catch (const exception& e) {
                cerr << "CPU K-Means failed: " << e.what() << endl;
            }
        }

#ifdef FLANN_USE_OPENCL
        if (config.run_opencl) {
            try {
                results.push_back(benchmark_opencl_kmeans(dataset, queries, gt_indices, config));
            }
            catch (const exception& e) {
                cerr << "OpenCL K-Means failed: " << e.what() << endl;
            }
        }
#endif

#ifdef FLANN_USE_CUDA
        if (config.run_cuda) {
            try {
                results.push_back(benchmark_cuda_kmeans(dataset, queries, gt_indices, config));
            }
            catch (const exception& e) {
                cerr << "CUDA K-Means failed: " << e.what() << endl;
            }
        }
#endif

        // Cleanup
        delete[] dataset.ptr();
        delete[] queries.ptr();
        delete[] gt_indices.ptr();
        delete[] gt_dists.ptr();

    } else {
        // ============================================
        // Hierarchical mode: Load binary data
        // ============================================
        cout << "\nLoading dataset (binary)..." << flush;
        Matrix<unsigned char> dataset;
        Matrix<unsigned char> queries;

        try {
            load_from_file(dataset, config.dataset_file, "dataset");
            load_from_file(queries, config.dataset_file, "query");
        }
        catch (const exception& e) {
            cerr << "\nError loading dataset: " << e.what() << endl;
            return 1;
        }

        cout << " done" << endl;
        cout << "  Index:  " << dataset.rows << " x " << dataset.cols << " bytes" << endl;
        cout << "  Query:  " << queries.rows << " x " << queries.cols << " bytes" << endl;

        // Compute ground truth
        Matrix<size_t> gt_indices;
        Matrix<unsigned int> gt_dists;
        compute_ground_truth<Hamming<unsigned char>>(dataset, queries, gt_indices, gt_dists, config.k);

        // Run Hierarchical benchmarks
        if (config.run_cpu) {
            try {
                results.push_back(benchmark_cpu_hierarchical(dataset, queries, gt_indices, config));
            }
            catch (const exception& e) {
                cerr << "CPU Hierarchical failed: " << e.what() << endl;
            }
        }

#ifdef FLANN_USE_OPENCL
        if (config.run_opencl) {
            try {
                results.push_back(benchmark_opencl_hierarchical(dataset, queries, gt_indices, config));
            }
            catch (const exception& e) {
                cerr << "OpenCL Hierarchical failed: " << e.what() << endl;
            }
        }
#endif

#ifdef FLANN_USE_CUDA
        if (config.run_cuda) {
            try {
                results.push_back(benchmark_cuda_hierarchical(dataset, queries, gt_indices, config));
            }
            catch (const exception& e) {
                cerr << "CUDA Hierarchical failed: " << e.what() << endl;
            }
        }
#endif

        // Cleanup
        delete[] dataset.ptr();
        delete[] queries.ptr();
        delete[] gt_indices.ptr();
        delete[] gt_dists.ptr();
    }

    // Print summary
    print_summary_table(results);

    // Export results
    string csv_file = config.output_prefix + "_" +
                      (config.run_kmeans ? "kmeans" : "hierarchical") + ".csv";
    export_results_csv(results, csv_file);

    cout << "\nBenchmark complete!" << endl;

    return 0;
}
