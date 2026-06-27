/**
 * CUDA K-Means Search Performance Benchmark
 *
 * Measures GPU kernel execution time (excluding H2D/D2H transfers)
 * for knnSearch with SIFT descriptors (128D float).
 *
 * Usage: ./test/benchmark_cuda_search [--dataset sift10K.h5|sift100K.h5] [--k 10] [--runs 20]
 */

#ifndef FLANN_USE_CUDA
#error "Requires FLANN_USE_CUDA"
#endif

#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

using namespace flann;
using namespace std;

struct BenchConfig {
    string dataset = "sift10K.h5";
    int k = 10;
    int runs = 20;
    int warmup = 3;
    int checks = 128;
};

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) cfg.dataset = argv[++i];
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) cfg.k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) cfg.runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) cfg.warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--checks") == 0 && i + 1 < argc) cfg.checks = atoi(argv[++i]);
    }
    return cfg;
}

int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);

    // Load dataset
    Matrix<float> dataset;
    Matrix<float> query;
    Matrix<int> gt_indices;
    Matrix<float> gt_dists;

    load_from_file(dataset, cfg.dataset.c_str(), "dataset");

    Matrix<float> raw_query;
    load_from_file(raw_query, cfg.dataset.c_str(), "query");

    // Tile queries to reach target count (default: use raw queries as-is)
    int query_multiplier = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--query-mult") == 0 && i + 1 < argc)
            query_multiplier = atoi(argv[++i]);
    }

    if (query_multiplier > 1) {
        size_t total_queries = raw_query.rows * query_multiplier;
        float* tiled = new float[total_queries * raw_query.cols];
        for (int m = 0; m < query_multiplier; m++)
            memcpy(tiled + m * raw_query.rows * raw_query.cols,
                   raw_query[0], raw_query.rows * raw_query.cols * sizeof(float));
        query = Matrix<float>(tiled, total_queries, raw_query.cols);
    } else {
        query = raw_query;
    }

    cout << "Dataset: " << cfg.dataset
         << " (" << dataset.rows << " x " << dataset.cols << ")" << endl;
    cout << "Queries: " << query.rows << " x " << query.cols << endl;
    cout << "K=" << cfg.k << ", checks=" << cfg.checks
         << ", runs=" << cfg.runs << ", warmup=" << cfg.warmup << endl;

    // Build index
    auto t0 = chrono::high_resolution_clock::now();
    Index<L2<float>> index(dataset,
        KMeansCUDAIndexParams(32, 11, FLANN_CENTERS_RANDOM, 0.2));
    index.buildIndex();
    auto t1 = chrono::high_resolution_clock::now();
    double build_ms = chrono::duration<double, milli>(t1 - t0).count();
    cout << "Build time: " << fixed << setprecision(1) << build_ms << " ms" << endl;

    // Prepare GPU for k-NN search (uploads tree + dataset to GPU)
    auto t2 = chrono::high_resolution_clock::now();
    index.buildCUDAKnnSearch(cfg.k, SearchParams(cfg.checks));
    auto t3 = chrono::high_resolution_clock::now();
    double gpu_setup_ms = chrono::duration<double, milli>(t3 - t2).count();
    cout << "GPU setup: " << fixed << setprecision(1) << gpu_setup_ms << " ms" << endl;

    // Allocate result matrices
    Matrix<int> indices(new int[query.rows * cfg.k], query.rows, cfg.k);
    Matrix<float> dists(new float[query.rows * cfg.k], query.rows, cfg.k);

    SearchParams params(cfg.checks);

    // Warmup runs
    for (int i = 0; i < cfg.warmup; i++) {
        index.knnSearch(query, indices, dists, cfg.k, params);
    }

    // Timed runs (wall-clock)
    vector<double> times;
    for (int i = 0; i < cfg.runs; i++) {
        auto start = chrono::high_resolution_clock::now();
        index.knnSearch(query, indices, dists, cfg.k, params);
        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start).count();
        times.push_back(ms);
    }

    // Also time with CUDA events to separate kernel from host overhead
    {
        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        // Warmup
        index.knnSearch(query, indices, dists, cfg.k, params);

        cudaEventRecord(start_ev);
        for (int i = 0; i < 5; i++) {
            index.knnSearch(query, indices, dists, cfg.k, params);
        }
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);

        float gpu_ms = 0;
        cudaEventElapsedTime(&gpu_ms, start_ev, stop_ev);
        cout << "CUDA event time (5 runs): " << fixed << setprecision(3) << gpu_ms << " ms ("
             << gpu_ms / 5.0 << " ms/search)" << endl;

        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    // Statistics
    sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    double mean = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double p10 = times[times.size() / 10];
    double p90 = times[times.size() * 9 / 10];

    // Compute recall if ground truth available
    try {
        load_from_file(gt_indices, cfg.dataset.c_str(), "match");
        int correct = 0;
        int total = 0;
        size_t eval_rows = min(query.rows, gt_indices.rows);
        for (size_t i = 0; i < eval_rows; i++) {
            for (int j = 0; j < cfg.k && j < (int)gt_indices.cols; j++) {
                for (int jj = 0; jj < cfg.k; jj++) {
                    if (indices[i][jj] == gt_indices[i][j]) {
                        correct++;
                        break;
                    }
                }
                total++;
            }
        }
        double recall = (double)correct / total;
        cout << "Recall@" << cfg.k << ": " << fixed << setprecision(3) << recall << endl;
    } catch (...) {
        // No ground truth
    }

    cout << "\n=== BENCHMARK RESULTS ===" << endl;
    cout << "Queries: " << query.rows << ", Dataset: " << dataset.rows
         << ", K=" << cfg.k << endl;
    cout << "Median: " << fixed << setprecision(3) << median << " ms" << endl;
    cout << "Mean:   " << fixed << setprecision(3) << mean << " ms" << endl;
    cout << "P10:    " << fixed << setprecision(3) << p10 << " ms" << endl;
    cout << "P90:    " << fixed << setprecision(3) << p90 << " ms" << endl;
    cout << "QPS:    " << fixed << setprecision(0) << (query.rows / (median / 1000.0)) << endl;
    cout << "METRIC: " << fixed << setprecision(3) << median << endl;

    delete[] dataset.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
    if (gt_indices.ptr()) delete[] gt_indices.ptr();
    if (gt_dists.ptr()) delete[] gt_dists.ptr();

    return 0;
}
