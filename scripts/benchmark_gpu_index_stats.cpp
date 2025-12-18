/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024-2025 FLANN GPU Index Statistics Utility
 *
 * Benchmark utility comparing CPU vs GPU index file formats.
 * Measures load times, file sizes, and search performance.
 **********************************************************************/

#ifndef FLANN_USE_CUDA
#error "This utility requires FLANN_USE_CUDA to be defined. Build with -DBUILD_CUDA_LIB=ON"
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
#include <cstdio>

using namespace flann;
using namespace std;
using namespace std::chrono;

// ============================================================================
// Configuration
// ============================================================================

struct BenchmarkConfig {
    string dataset_file;
    string index_type;        // "kmeans" or "hierarchical" or "auto"
    int k;
    int checks;
    int branching;
    int trees;
    int leaf_size;
    int iterations;
    float cb_index;
    int num_runs;
    int warmup_runs;
    string output_file;
    bool verbose;
    bool run_all;

    BenchmarkConfig() :
        dataset_file(""),
        index_type("auto"),
        k(10), checks(128),
        branching(32), trees(4), leaf_size(100),
        iterations(11), cb_index(0.2f),
        num_runs(5), warmup_runs(2),
        output_file(""),
        verbose(false), run_all(false) {}
};

// ============================================================================
// Timer Utility
// ============================================================================

class Timer {
    high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double, milli>>(end - start_time).count();
    }

    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }

    void reset() {
        start_time = high_resolution_clock::now();
    }
};

// ============================================================================
// Statistics
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

    string format_ms() const {
        ostringstream oss;
        oss << fixed << setprecision(1) << mean << " +/- " << setprecision(1) << stddev << " ms";
        return oss.str();
    }
};

// ============================================================================
// File Utilities
// ============================================================================

size_t getFileSize(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return st.st_size;
    }
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

// ============================================================================
// Precision Calculation
// ============================================================================

template<typename T>
double calculate_precision(const Matrix<T>& gt_indices,
                          const Matrix<T>& test_indices,
                          int k)
{
    if (gt_indices.rows != test_indices.rows) {
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
    if (gt_indices.ptr() != nullptr) return;  // Already computed

    cout << "  Computing ground truth..." << flush;
    Timer timer;

    Index<Distance> index(dataset, LinearIndexParams());
    index.buildIndex();

    gt_indices = Matrix<size_t>(new size_t[queries.rows * k], queries.rows, k);
    gt_dists = Matrix<typename Distance::ResultType>(
        new typename Distance::ResultType[queries.rows * k], queries.rows, k);

    index.knnSearch(queries, gt_indices, gt_dists, k, SearchParams(-1));

    cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;
}

// ============================================================================
// Format Result Structure
// ============================================================================

struct FormatResult {
    string format_name;
    size_t file_size_bytes;

    // Timing statistics from multiple runs
    Stats load_time;         // Time to load index from file
    Stats gpu_setup_time;    // Time for buildCUDAKnnSearch
    Stats total_ready_time;  // load + gpu_setup

    // Search performance
    Stats first_search_time;
    Stats steady_search_time;
    double precision;
    double queries_per_sec;
};

struct DatasetResult {
    string dataset_name;
    string index_type;
    size_t rows;
    size_t cols;

    FormatResult cpu_format;
    FormatResult gpu_format;

    double load_speedup;
    double size_ratio;
};

// ============================================================================
// Hierarchical Index Benchmark
// ============================================================================

void benchmark_hierarchical(
    const Matrix<unsigned char>& dataset,
    const Matrix<unsigned char>& queries,
    Matrix<size_t>& gt_indices,
    Matrix<unsigned int>& gt_dists,
    const BenchmarkConfig& config,
    DatasetResult& result)
{
    result.index_type = "Hierarchical";

    // Create temporary file paths
    string cpu_file = "/tmp/flann_bench_cpu_hier.idx";
    string gpu_file = "/tmp/flann_bench_gpu_hier.idx";

    // ========================================================================
    // Step 1: Create CPU and GPU format index files
    // ========================================================================
    cout << "\n  Creating index files..." << endl;

    HierarchicalCUDAIndexParams params(
        config.branching,
        FLANN_CENTERS_RANDOM,
        config.trees,
        config.leaf_size
    );
    params["save_dataset"] = true;  // Embed dataset in saved file

    {
        cout << "    Building index..." << flush;
        Timer timer;
        Index<Hamming<unsigned char>> index(dataset, params);
        index.buildIndex();
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        // Save CPU format (with dataset embedded for standalone loading)
        cout << "    Saving CPU format..." << flush;
        timer.reset();
        index.save(cpu_file);
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        // Prepare GPU and save GPU format
        cout << "    Setting up GPU..." << flush;
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        cout << "    Converting to GPU format..." << flush;
        timer.reset();
        index.convertToGPUFormat();
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        cout << "    Saving GPU format..." << flush;
        timer.reset();
        index.save(gpu_file);
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;
    }

    // Get file sizes
    result.cpu_format.file_size_bytes = getFileSize(cpu_file);
    result.gpu_format.file_size_bytes = getFileSize(gpu_file);
    result.cpu_format.format_name = "CPU";
    result.gpu_format.format_name = "GPU";

    cout << "\n  File sizes:" << endl;
    cout << "    CPU format: " << formatFileSize(result.cpu_format.file_size_bytes) << endl;
    cout << "    GPU format: " << formatFileSize(result.gpu_format.file_size_bytes) << endl;
    result.size_ratio = (double)result.gpu_format.file_size_bytes / result.cpu_format.file_size_bytes;
    cout << "    Ratio: " << fixed << setprecision(2) << result.size_ratio << "x" << endl;

    // Compute ground truth for precision verification
    compute_ground_truth<Hamming<unsigned char>>(dataset, queries, gt_indices, gt_dists, config.k);

    // ========================================================================
    // Step 2: Benchmark CPU format loading
    // ========================================================================
    cout << "\n  Benchmarking CPU format load (" << config.warmup_runs << " warmup + "
         << config.num_runs << " timed runs)..." << endl;

    vector<double> cpu_load_times, cpu_setup_times, cpu_total_times;
    vector<double> cpu_first_search, cpu_steady_search;

    int total_runs = config.warmup_runs + config.num_runs;
    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        if (config.verbose) {
            cout << "    " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;
        }

        Timer timer;

        // Load from CPU format file
        timer.reset();
        Matrix<unsigned char> empty_data;
        Index<Hamming<unsigned char>> index(empty_data, SavedIndexParams(cpu_file));
        double load_time = timer.elapsed_ms();

        // Setup GPU (includes tree flattening and upload)
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        double setup_time = timer.elapsed_ms();

        double total_time = load_time + setup_time;

        if (!is_warmup) {
            cpu_load_times.push_back(load_time);
            cpu_setup_times.push_back(setup_time);
            cpu_total_times.push_back(total_time);
        }

        // Search performance (on last run)
        if (run == total_runs - 1) {
            Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
            Matrix<unsigned int> distances(new unsigned int[queries.rows * config.k], queries.rows, config.k);

            // First search (may include remaining JIT)
            timer.reset();
            index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
            cpu_first_search.push_back(timer.elapsed_ms());

            // Steady-state searches
            for (int s = 0; s < 3; s++) {
                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                cpu_steady_search.push_back(timer.elapsed_ms());
            }

            result.cpu_format.precision = calculate_precision(gt_indices, indices, config.k);

            delete[] indices.ptr();
            delete[] distances.ptr();
        }
    }

    result.cpu_format.load_time = Stats(cpu_load_times);
    result.cpu_format.gpu_setup_time = Stats(cpu_setup_times);
    result.cpu_format.total_ready_time = Stats(cpu_total_times);
    result.cpu_format.first_search_time = Stats(cpu_first_search);
    result.cpu_format.steady_search_time = Stats(cpu_steady_search);
    result.cpu_format.queries_per_sec = queries.rows / (result.cpu_format.steady_search_time.mean / 1000.0);

    cout << "    CPU format load: " << result.cpu_format.load_time.format_ms() << endl;
    cout << "    CPU format GPU setup: " << result.cpu_format.gpu_setup_time.format_ms() << endl;
    cout << "    CPU format total ready: " << result.cpu_format.total_ready_time.format_ms() << endl;

    // ========================================================================
    // Step 3: Benchmark GPU format loading
    // ========================================================================
    cout << "\n  Benchmarking GPU format load (" << config.warmup_runs << " warmup + "
         << config.num_runs << " timed runs)..." << endl;

    vector<double> gpu_load_times, gpu_setup_times, gpu_total_times;
    vector<double> gpu_first_search, gpu_steady_search;

    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        if (config.verbose) {
            cout << "    " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;
        }

        Timer timer;

        // Load from GPU format file (direct GPU upload, no tree reconstruction)
        timer.reset();
        Matrix<unsigned char> empty_data;
        Index<Hamming<unsigned char>> index(empty_data, SavedIndexParams(gpu_file));
        double load_time = timer.elapsed_ms();

        // Setup GPU (already has GPU data, just warmup)
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        double setup_time = timer.elapsed_ms();

        double total_time = load_time + setup_time;

        if (!is_warmup) {
            gpu_load_times.push_back(load_time);
            gpu_setup_times.push_back(setup_time);
            gpu_total_times.push_back(total_time);
        }

        // Search performance (on last run)
        if (run == total_runs - 1) {
            Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
            Matrix<unsigned int> distances(new unsigned int[queries.rows * config.k], queries.rows, config.k);

            // First search
            timer.reset();
            index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
            gpu_first_search.push_back(timer.elapsed_ms());

            // Steady-state searches
            for (int s = 0; s < 3; s++) {
                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                gpu_steady_search.push_back(timer.elapsed_ms());
            }

            result.gpu_format.precision = calculate_precision(gt_indices, indices, config.k);

            delete[] indices.ptr();
            delete[] distances.ptr();
        }
    }

    result.gpu_format.load_time = Stats(gpu_load_times);
    result.gpu_format.gpu_setup_time = Stats(gpu_setup_times);
    result.gpu_format.total_ready_time = Stats(gpu_total_times);
    result.gpu_format.first_search_time = Stats(gpu_first_search);
    result.gpu_format.steady_search_time = Stats(gpu_steady_search);
    result.gpu_format.queries_per_sec = queries.rows / (result.gpu_format.steady_search_time.mean / 1000.0);

    cout << "    GPU format load: " << result.gpu_format.load_time.format_ms() << endl;
    cout << "    GPU format GPU setup: " << result.gpu_format.gpu_setup_time.format_ms() << endl;
    cout << "    GPU format total ready: " << result.gpu_format.total_ready_time.format_ms() << endl;

    // Calculate speedup
    result.load_speedup = result.cpu_format.total_ready_time.mean / result.gpu_format.total_ready_time.mean;

    // Cleanup temp files
    remove(cpu_file.c_str());
    remove(gpu_file.c_str());
}

// ============================================================================
// K-Means Index Benchmark
// ============================================================================

void benchmark_kmeans(
    const Matrix<float>& dataset,
    const Matrix<float>& queries,
    Matrix<size_t>& gt_indices,
    Matrix<float>& gt_dists,
    const BenchmarkConfig& config,
    DatasetResult& result)
{
    result.index_type = "K-Means";

    // Create temporary file paths
    string cpu_file = "/tmp/flann_bench_cpu_kmeans.idx";
    string gpu_file = "/tmp/flann_bench_gpu_kmeans.idx";

    // ========================================================================
    // Step 1: Create CPU and GPU format index files
    // ========================================================================
    cout << "\n  Creating index files..." << endl;

    KMeansCUDAIndexParams params(
        config.branching,
        config.iterations,
        FLANN_CENTERS_RANDOM,
        config.cb_index
    );
    params["save_dataset"] = true;  // Embed dataset in saved file

    {
        cout << "    Building index..." << flush;
        Timer timer;
        Index<L2<float>> index(dataset, params);
        index.buildIndex();
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        // Save CPU format (with dataset embedded for standalone loading)
        cout << "    Saving CPU format..." << flush;
        timer.reset();
        index.save(cpu_file);
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        // Prepare GPU and save GPU format
        cout << "    Setting up GPU..." << flush;
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        cout << "    Converting to GPU format..." << flush;
        timer.reset();
        index.convertToGPUFormat();
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

        cout << "    Saving GPU format..." << flush;
        timer.reset();
        index.save(gpu_file);
        cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;
    }

    // Get file sizes
    result.cpu_format.file_size_bytes = getFileSize(cpu_file);
    result.gpu_format.file_size_bytes = getFileSize(gpu_file);
    result.cpu_format.format_name = "CPU";
    result.gpu_format.format_name = "GPU";

    cout << "\n  File sizes:" << endl;
    cout << "    CPU format: " << formatFileSize(result.cpu_format.file_size_bytes) << endl;
    cout << "    GPU format: " << formatFileSize(result.gpu_format.file_size_bytes) << endl;
    result.size_ratio = (double)result.gpu_format.file_size_bytes / result.cpu_format.file_size_bytes;
    cout << "    Ratio: " << fixed << setprecision(2) << result.size_ratio << "x" << endl;

    // Compute ground truth for precision verification
    compute_ground_truth<L2<float>>(dataset, queries, gt_indices, gt_dists, config.k);

    // ========================================================================
    // Step 2: Benchmark CPU format loading
    // ========================================================================
    cout << "\n  Benchmarking CPU format load (" << config.warmup_runs << " warmup + "
         << config.num_runs << " timed runs)..." << endl;

    vector<double> cpu_load_times, cpu_setup_times, cpu_total_times;
    vector<double> cpu_first_search, cpu_steady_search;

    int total_runs = config.warmup_runs + config.num_runs;
    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        if (config.verbose) {
            cout << "    " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;
        }

        Timer timer;

        // Load from CPU format file
        timer.reset();
        Matrix<float> empty_data;
        Index<L2<float>> index(empty_data, SavedIndexParams(cpu_file));
        double load_time = timer.elapsed_ms();

        // Setup GPU
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        double setup_time = timer.elapsed_ms();

        double total_time = load_time + setup_time;

        if (!is_warmup) {
            cpu_load_times.push_back(load_time);
            cpu_setup_times.push_back(setup_time);
            cpu_total_times.push_back(total_time);
        }

        // Search performance (on last run)
        if (run == total_runs - 1) {
            Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
            Matrix<float> distances(new float[queries.rows * config.k], queries.rows, config.k);

            // First search
            timer.reset();
            index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
            cpu_first_search.push_back(timer.elapsed_ms());

            // Steady-state searches
            for (int s = 0; s < 3; s++) {
                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                cpu_steady_search.push_back(timer.elapsed_ms());
            }

            result.cpu_format.precision = calculate_precision(gt_indices, indices, config.k);

            delete[] indices.ptr();
            delete[] distances.ptr();
        }
    }

    result.cpu_format.load_time = Stats(cpu_load_times);
    result.cpu_format.gpu_setup_time = Stats(cpu_setup_times);
    result.cpu_format.total_ready_time = Stats(cpu_total_times);
    result.cpu_format.first_search_time = Stats(cpu_first_search);
    result.cpu_format.steady_search_time = Stats(cpu_steady_search);
    result.cpu_format.queries_per_sec = queries.rows / (result.cpu_format.steady_search_time.mean / 1000.0);

    cout << "    CPU format load: " << result.cpu_format.load_time.format_ms() << endl;
    cout << "    CPU format GPU setup: " << result.cpu_format.gpu_setup_time.format_ms() << endl;
    cout << "    CPU format total ready: " << result.cpu_format.total_ready_time.format_ms() << endl;

    // ========================================================================
    // Step 3: Benchmark GPU format loading
    // ========================================================================
    cout << "\n  Benchmarking GPU format load (" << config.warmup_runs << " warmup + "
         << config.num_runs << " timed runs)..." << endl;

    vector<double> gpu_load_times, gpu_setup_times, gpu_total_times;
    vector<double> gpu_first_search, gpu_steady_search;

    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        if (config.verbose) {
            cout << "    " << (is_warmup ? "Warmup" : "Run") << " " << (run + 1) << "..." << endl;
        }

        Timer timer;

        // Load from GPU format file
        timer.reset();
        Matrix<float> empty_data;
        Index<L2<float>> index(empty_data, SavedIndexParams(gpu_file));
        double load_time = timer.elapsed_ms();

        // Setup GPU
        timer.reset();
        index.buildCUDAKnnSearch(config.k, SearchParams(config.checks));
        double setup_time = timer.elapsed_ms();

        double total_time = load_time + setup_time;

        if (!is_warmup) {
            gpu_load_times.push_back(load_time);
            gpu_setup_times.push_back(setup_time);
            gpu_total_times.push_back(total_time);
        }

        // Search performance (on last run)
        if (run == total_runs - 1) {
            Matrix<size_t> indices(new size_t[queries.rows * config.k], queries.rows, config.k);
            Matrix<float> distances(new float[queries.rows * config.k], queries.rows, config.k);

            // First search
            timer.reset();
            index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
            gpu_first_search.push_back(timer.elapsed_ms());

            // Steady-state searches
            for (int s = 0; s < 3; s++) {
                timer.reset();
                index.knnSearch(queries, indices, distances, config.k, SearchParams(config.checks));
                gpu_steady_search.push_back(timer.elapsed_ms());
            }

            result.gpu_format.precision = calculate_precision(gt_indices, indices, config.k);

            delete[] indices.ptr();
            delete[] distances.ptr();
        }
    }

    result.gpu_format.load_time = Stats(gpu_load_times);
    result.gpu_format.gpu_setup_time = Stats(gpu_setup_times);
    result.gpu_format.total_ready_time = Stats(gpu_total_times);
    result.gpu_format.first_search_time = Stats(gpu_first_search);
    result.gpu_format.steady_search_time = Stats(gpu_steady_search);
    result.gpu_format.queries_per_sec = queries.rows / (result.gpu_format.steady_search_time.mean / 1000.0);

    cout << "    GPU format load: " << result.gpu_format.load_time.format_ms() << endl;
    cout << "    GPU format GPU setup: " << result.gpu_format.gpu_setup_time.format_ms() << endl;
    cout << "    GPU format total ready: " << result.gpu_format.total_ready_time.format_ms() << endl;

    // Calculate speedup
    result.load_speedup = result.cpu_format.total_ready_time.mean / result.gpu_format.total_ready_time.mean;

    // Cleanup temp files
    remove(cpu_file.c_str());
    remove(gpu_file.c_str());
}

// ============================================================================
// Output Functions
// ============================================================================

void print_result(const DatasetResult& result) {
    cout << "\n" << string(80, '=') << endl;
    cout << "RESULTS: " << result.dataset_name << endl;
    cout << string(80, '=') << endl;

    cout << "\nDataset: " << result.dataset_name << " (" << result.rows << " x " << result.cols << ")" << endl;
    cout << "Index Type: " << result.index_type << endl;

    cout << "\n" << string(80, '-') << endl;
    cout << "FILE SIZE COMPARISON" << endl;
    cout << string(80, '-') << endl;
    cout << "  CPU Format: " << formatFileSize(result.cpu_format.file_size_bytes) << endl;
    cout << "  GPU Format: " << formatFileSize(result.gpu_format.file_size_bytes) << endl;
    cout << "  Size Ratio: " << fixed << setprecision(2) << result.size_ratio << "x";
    if (result.size_ratio > 1.0) {
        cout << " (GPU " << fixed << setprecision(0) << ((result.size_ratio - 1.0) * 100) << "% larger)";
    } else {
        cout << " (GPU " << fixed << setprecision(0) << ((1.0 - result.size_ratio) * 100) << "% smaller)";
    }
    cout << endl;

    cout << "\n" << string(80, '-') << endl;
    cout << "LOAD TIME COMPARISON" << endl;
    cout << string(80, '-') << endl;
    cout << left << setw(25) << "  Phase" << setw(25) << "CPU Format" << setw(25) << "GPU Format" << endl;
    cout << "  " << string(73, '-') << endl;
    cout << left << setw(25) << "  File Load"
         << setw(25) << result.cpu_format.load_time.format_ms()
         << setw(25) << result.gpu_format.load_time.format_ms() << endl;
    cout << left << setw(25) << "  GPU Setup"
         << setw(25) << result.cpu_format.gpu_setup_time.format_ms()
         << setw(25) << result.gpu_format.gpu_setup_time.format_ms() << endl;
    cout << "  " << string(73, '-') << endl;
    cout << left << setw(25) << "  TOTAL READY"
         << setw(25) << result.cpu_format.total_ready_time.format_ms()
         << setw(25) << result.gpu_format.total_ready_time.format_ms() << endl;
    cout << endl;
    cout << "  Load Speedup: " << fixed << setprecision(2) << result.load_speedup << "x faster with GPU format" << endl;
    double time_saved = result.cpu_format.total_ready_time.mean - result.gpu_format.total_ready_time.mean;
    cout << "  Time Saved: " << fixed << setprecision(1) << time_saved << " ms per load" << endl;

    cout << "\n" << string(80, '-') << endl;
    cout << "SEARCH PERFORMANCE (after load)" << endl;
    cout << string(80, '-') << endl;
    cout << left << setw(25) << "  Metric" << setw(25) << "CPU Format Load" << setw(25) << "GPU Format Load" << endl;
    cout << "  " << string(73, '-') << endl;
    cout << left << setw(25) << "  First Search"
         << setw(25) << result.cpu_format.first_search_time.format_ms()
         << setw(25) << result.gpu_format.first_search_time.format_ms() << endl;
    cout << left << setw(25) << "  Steady State"
         << setw(25) << result.cpu_format.steady_search_time.format_ms()
         << setw(25) << result.gpu_format.steady_search_time.format_ms() << endl;

    ostringstream cpu_qps, gpu_qps;
    cpu_qps << fixed << setprecision(0) << result.cpu_format.queries_per_sec << " q/s";
    gpu_qps << fixed << setprecision(0) << result.gpu_format.queries_per_sec << " q/s";
    cout << left << setw(25) << "  Throughput"
         << setw(25) << cpu_qps.str()
         << setw(25) << gpu_qps.str() << endl;

    ostringstream cpu_prec, gpu_prec;
    cpu_prec << fixed << setprecision(1) << (result.cpu_format.precision * 100) << "%";
    gpu_prec << fixed << setprecision(1) << (result.gpu_format.precision * 100) << "%";
    cout << left << setw(25) << "  Precision"
         << setw(25) << cpu_prec.str()
         << setw(25) << gpu_prec.str() << endl;

    cout << "\n" << string(80, '-') << endl;
    cout << "SUMMARY" << endl;
    cout << string(80, '-') << endl;
    cout << "  GPU format provides " << fixed << setprecision(2) << result.load_speedup
         << "x faster startup with " << fixed << setprecision(0) << ((result.size_ratio - 1.0) * 100)
         << "% larger files." << endl;

    // Break-even analysis: how many loads to amortize storage overhead
    if (result.size_ratio > 1.0) {
        double extra_storage_mb = (result.gpu_format.file_size_bytes - result.cpu_format.file_size_bytes) / (1024.0 * 1024.0);
        cout << "  Storage overhead: " << fixed << setprecision(2) << extra_storage_mb << " MB extra" << endl;
    }

    cout << "  Recommended for: Frequently loaded indices, containerized deployments, cold starts." << endl;
}

void export_csv(const vector<DatasetResult>& results, const string& filename) {
    ofstream csv(filename);
    if (!csv.is_open()) {
        cerr << "Error: Could not open " << filename << " for writing" << endl;
        return;
    }

    csv << "dataset,index_type,rows,cols,cpu_file_mb,gpu_file_mb,size_ratio,"
        << "cpu_load_ms,cpu_load_std,cpu_setup_ms,cpu_setup_std,cpu_total_ms,cpu_total_std,"
        << "gpu_load_ms,gpu_load_std,gpu_setup_ms,gpu_setup_std,gpu_total_ms,gpu_total_std,"
        << "load_speedup,cpu_precision,gpu_precision,cpu_qps,gpu_qps" << endl;

    for (const auto& r : results) {
        csv << r.dataset_name << ","
            << r.index_type << ","
            << r.rows << ","
            << r.cols << ","
            << fixed << setprecision(3) << (r.cpu_format.file_size_bytes / (1024.0 * 1024.0)) << ","
            << fixed << setprecision(3) << (r.gpu_format.file_size_bytes / (1024.0 * 1024.0)) << ","
            << fixed << setprecision(3) << r.size_ratio << ","
            << fixed << setprecision(2) << r.cpu_format.load_time.mean << ","
            << fixed << setprecision(2) << r.cpu_format.load_time.stddev << ","
            << fixed << setprecision(2) << r.cpu_format.gpu_setup_time.mean << ","
            << fixed << setprecision(2) << r.cpu_format.gpu_setup_time.stddev << ","
            << fixed << setprecision(2) << r.cpu_format.total_ready_time.mean << ","
            << fixed << setprecision(2) << r.cpu_format.total_ready_time.stddev << ","
            << fixed << setprecision(2) << r.gpu_format.load_time.mean << ","
            << fixed << setprecision(2) << r.gpu_format.load_time.stddev << ","
            << fixed << setprecision(2) << r.gpu_format.gpu_setup_time.mean << ","
            << fixed << setprecision(2) << r.gpu_format.gpu_setup_time.stddev << ","
            << fixed << setprecision(2) << r.gpu_format.total_ready_time.mean << ","
            << fixed << setprecision(2) << r.gpu_format.total_ready_time.stddev << ","
            << fixed << setprecision(3) << r.load_speedup << ","
            << fixed << setprecision(4) << r.cpu_format.precision << ","
            << fixed << setprecision(4) << r.gpu_format.precision << ","
            << fixed << setprecision(1) << r.cpu_format.queries_per_sec << ","
            << fixed << setprecision(1) << r.gpu_format.queries_per_sec
            << endl;
    }

    csv.close();
    cout << "\nCSV results saved to: " << filename << endl;
}

void print_summary_table(const vector<DatasetResult>& results) {
    cout << "\n" << string(100, '=') << endl;
    cout << "BENCHMARK SUMMARY" << endl;
    cout << string(100, '=') << endl;

    cout << left << setw(20) << "Dataset"
         << setw(15) << "Index Type"
         << right << setw(12) << "CPU File"
         << setw(12) << "GPU File"
         << setw(10) << "Ratio"
         << setw(15) << "CPU Ready"
         << setw(15) << "GPU Ready"
         << setw(10) << "Speedup" << endl;
    cout << string(100, '-') << endl;

    for (const auto& r : results) {
        cout << left << setw(20) << r.dataset_name
             << setw(15) << r.index_type
             << right << setw(12) << formatFileSize(r.cpu_format.file_size_bytes)
             << setw(12) << formatFileSize(r.gpu_format.file_size_bytes)
             << setw(10) << fixed << setprecision(2) << r.size_ratio << "x"
             << setw(15) << r.cpu_format.total_ready_time.format_ms()
             << setw(15) << r.gpu_format.total_ready_time.format_ms()
             << setw(10) << fixed << setprecision(2) << r.load_speedup << "x" << endl;
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program) {
    cout << "FLANN GPU Index Format Statistics Utility\n\n"
         << "Usage: " << program << " [options]\n\n"
         << "Options:\n"
         << "  --dataset=<file.h5>    HDF5 dataset file\n"
         << "  --index-type=<type>    Force index type: kmeans, hierarchical, auto\n"
         << "  --k=<value>            Number of nearest neighbors (default: 10)\n"
         << "  --checks=<value>       Search accuracy parameter (default: 128)\n"
         << "  --runs=<N>             Number of timed runs (default: 5)\n"
         << "  --warmup=<N>           Number of warmup runs (default: 2)\n"
         << "  --output=<file.csv>    CSV output file\n"
         << "  --verbose              Print detailed progress\n"
         << "  --all                  Run all available datasets\n"
         << "  --help                 Show this help message\n\n"
         << "Examples:\n"
         << "  " << program << " --dataset=datasets/brief100K.h5\n"
         << "  " << program << " --all --output=results.csv\n"
         << "  " << program << " --dataset=datasets/sift100K.h5 --index-type=kmeans\n";
}

int main(int argc, char** argv) {
    BenchmarkConfig config;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg.find("--dataset=") == 0) {
            config.dataset_file = arg.substr(10);
        }
        else if (arg.find("--index-type=") == 0) {
            config.index_type = arg.substr(13);
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
        else if (arg.find("--output=") == 0) {
            config.output_file = arg.substr(9);
        }
        else if (arg == "--verbose") {
            config.verbose = true;
        }
        else if (arg == "--all") {
            config.run_all = true;
        }
        else {
            cerr << "Unknown option: " << arg << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Build list of datasets to run
    vector<pair<string, string>> datasets;  // (file, type)

    if (config.run_all) {
        // All available datasets
        datasets.push_back({"datasets/sift10K.h5", "kmeans"});
        datasets.push_back({"datasets/sift100K.h5", "kmeans"});
        datasets.push_back({"datasets/cloud.h5", "kmeans"});  // 3D point cloud (177K × 3 float)
        datasets.push_back({"datasets/brief100K.h5", "hierarchical"});
        datasets.push_back({"datasets/binary1M_512bit.h5", "hierarchical"});
    } else if (!config.dataset_file.empty()) {
        // Single dataset
        string type = config.index_type;
        if (type == "auto") {
            // Auto-detect based on filename
            if (config.dataset_file.find("sift") != string::npos ||
                config.dataset_file.find("float") != string::npos ||
                config.dataset_file.find("cloud") != string::npos) {
                type = "kmeans";
            } else {
                type = "hierarchical";
            }
        }
        datasets.push_back({config.dataset_file, type});
    } else {
        // Default: run brief100K as a quick test
        datasets.push_back({"datasets/brief100K.h5", "hierarchical"});
    }

    cout << string(80, '=') << endl;
    cout << "FLANN GPU Index Format Statistics" << endl;
    cout << string(80, '=') << endl;
    cout << "\nConfiguration:" << endl;
    cout << "  k:       " << config.k << endl;
    cout << "  checks:  " << config.checks << endl;
    cout << "  runs:    " << config.num_runs << endl;
    cout << "  warmup:  " << config.warmup_runs << endl;
    cout << "  datasets: " << datasets.size() << endl;

    vector<DatasetResult> results;

    for (const auto& ds : datasets) {
        const string& file = ds.first;
        const string& type = ds.second;

        cout << "\n" << string(80, '=') << endl;
        cout << "Processing: " << file << " (" << type << ")" << endl;
        cout << string(80, '=') << endl;

        DatasetResult result;

        // Extract dataset name from path
        size_t last_slash = file.find_last_of('/');
        result.dataset_name = (last_slash != string::npos) ? file.substr(last_slash + 1) : file;

        try {
            if (type == "kmeans") {
                Matrix<float> dataset, queries;
                cout << "  Loading dataset..." << flush;
                Timer timer;
                load_from_file(dataset, file, "dataset");
                load_from_file(queries, file, "query");
                cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

                result.rows = dataset.rows;
                result.cols = dataset.cols;
                cout << "  Dataset: " << dataset.rows << " x " << dataset.cols << " floats" << endl;
                cout << "  Queries: " << queries.rows << " x " << queries.cols << " floats" << endl;

                Matrix<size_t> gt_indices;
                Matrix<float> gt_dists;

                benchmark_kmeans(dataset, queries, gt_indices, gt_dists, config, result);

                delete[] dataset.ptr();
                delete[] queries.ptr();
                if (gt_indices.ptr()) delete[] gt_indices.ptr();
                if (gt_dists.ptr()) delete[] gt_dists.ptr();

            } else {  // hierarchical
                Matrix<unsigned char> dataset, queries;
                cout << "  Loading dataset..." << flush;
                Timer timer;
                load_from_file(dataset, file, "dataset");
                load_from_file(queries, file, "query");
                cout << " done (" << fixed << setprecision(1) << timer.elapsed_ms() << " ms)" << endl;

                result.rows = dataset.rows;
                result.cols = dataset.cols;
                cout << "  Dataset: " << dataset.rows << " x " << dataset.cols << " bytes" << endl;
                cout << "  Queries: " << queries.rows << " x " << queries.cols << " bytes" << endl;

                Matrix<size_t> gt_indices;
                Matrix<unsigned int> gt_dists;

                benchmark_hierarchical(dataset, queries, gt_indices, gt_dists, config, result);

                delete[] dataset.ptr();
                delete[] queries.ptr();
                if (gt_indices.ptr()) delete[] gt_indices.ptr();
                if (gt_dists.ptr()) delete[] gt_dists.ptr();
            }

            print_result(result);
            results.push_back(result);

        } catch (const exception& e) {
            cerr << "Error processing " << file << ": " << e.what() << endl;
            continue;
        }
    }

    // Print summary table
    if (results.size() > 1) {
        print_summary_table(results);
    }

    // Export CSV if requested
    if (!config.output_file.empty()) {
        export_csv(results, config.output_file);
    }

    cout << "\nBenchmark complete!" << endl;

    return 0;
}
