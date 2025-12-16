/**
 * Detailed timing breakdown for GPU index format loading
 * Measures each step to understand where time is spent
 */

#ifndef FLANN_USE_CUDA
#error "Requires CUDA"
#endif

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>

using namespace flann;
using namespace std;
using namespace std::chrono;

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

size_t getFileSize(const string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0) ? st.st_size : 0;
}

int main() {
    const string dataset_file = "datasets/sift10K.h5";
    const string cpu_file = "/tmp/detailed_cpu.idx";
    const string gpu_file = "/tmp/detailed_gpu.idx";
    const int k = 10;
    const int checks = 128;

    cout << "=== Detailed Timing Analysis for sift10K ===" << endl << endl;

    // Load dataset
    Matrix<float> dataset, queries;
    load_from_file(dataset, dataset_file, "dataset");
    load_from_file(queries, dataset_file, "query");
    cout << "Dataset: " << dataset.rows << " x " << dataset.cols << " floats" << endl;
    cout << "Raw dataset size: " << (dataset.rows * dataset.cols * sizeof(float)) / 1024.0 << " KB" << endl;
    cout << endl;

    // ========================================================================
    // STEP 1: Create CPU format index file
    // ========================================================================
    cout << "=== Creating CPU Format Index ===" << endl;

    KMeansCUDAIndexParams params(32, 11, FLANN_CENTERS_RANDOM, 0.2f);
    params["save_dataset"] = true;

    {
        Timer t;
        Index<L2<float>> index(dataset, params);
        index.buildIndex();
        cout << "Build index: " << fixed << setprecision(2) << t.elapsed_ms() << " ms" << endl;

        t.reset();
        index.save(cpu_file);
        cout << "Save CPU format: " << t.elapsed_ms() << " ms" << endl;
    }

    // ========================================================================
    // STEP 2: Create GPU format index file (separate build)
    // ========================================================================
    cout << "\n=== Creating GPU Format Index ===" << endl;

    {
        Timer t;
        Index<L2<float>> index(dataset, params);
        index.buildIndex();
        cout << "Build index: " << fixed << setprecision(2) << t.elapsed_ms() << " ms" << endl;

        t.reset();
        index.buildCUDAKnnSearch(k, SearchParams(checks));
        cout << "Setup GPU: " << t.elapsed_ms() << " ms" << endl;

        t.reset();
        index.convertToGPUFormat();
        cout << "Convert to GPU format: " << t.elapsed_ms() << " ms" << endl;

        t.reset();
        index.save(gpu_file);
        cout << "Save GPU format: " << t.elapsed_ms() << " ms" << endl;
    }

    size_t cpu_size = getFileSize(cpu_file);
    size_t gpu_size = getFileSize(gpu_file);
    cout << endl;
    cout << "CPU file size: " << cpu_size / 1024.0 << " KB" << endl;
    cout << "GPU file size: " << gpu_size / 1024.0 << " KB" << endl;
    cout << endl;

    // ========================================================================
    // STEP 3: Detailed CPU format loading
    // ========================================================================
    cout << "=== CPU Format Loading (Detailed) ===" << endl;

    // Warmup run first
    {
        Matrix<float> empty;
        Index<L2<float>> warmup(empty, SavedIndexParams(cpu_file));
        warmup.buildCUDAKnnSearch(k, SearchParams(checks));
    }

    for (int run = 0; run < 3; run++) {
        cout << "Run " << (run+1) << ":" << endl;

        Timer t;

        // This includes: file read, LZ4 decompress, tree reconstruction
        t.reset();
        Matrix<float> empty;
        Index<L2<float>> index(empty, SavedIndexParams(cpu_file));
        double load_time = t.elapsed_ms();

        // This includes: flatten tree, cudaMalloc, cudaMemcpy, warmup kernel
        t.reset();
        index.buildCUDAKnnSearch(k, SearchParams(checks));
        double setup_time = t.elapsed_ms();

        cout << "  File load (read + decompress + tree reconstruct): "
             << fixed << setprecision(2) << load_time << " ms" << endl;
        cout << "  GPU setup (flatten + malloc + memcpy + warmup): "
             << setup_time << " ms" << endl;
        cout << "  TOTAL: " << (load_time + setup_time) << " ms" << endl;
    }
    cout << endl;

    // ========================================================================
    // STEP 4: Detailed GPU format loading
    // ========================================================================
    cout << "=== GPU Format Loading (Detailed) ===" << endl;

    // Warmup run first
    {
        Matrix<float> empty;
        Index<L2<float>> warmup(empty, SavedIndexParams(gpu_file));
        warmup.buildCUDAKnnSearch(k, SearchParams(checks));
    }

    for (int run = 0; run < 3; run++) {
        cout << "Run " << (run+1) << ":" << endl;

        Timer t;

        // This includes: file read, LZ4 decompress, host alloc, cudaMalloc, cudaMemcpy
        t.reset();
        Matrix<float> empty;
        Index<L2<float>> index(empty, SavedIndexParams(gpu_file));
        double load_time = t.elapsed_ms();

        // This should just be warmup kernel (GPU data already loaded)
        t.reset();
        index.buildCUDAKnnSearch(k, SearchParams(checks));
        double setup_time = t.elapsed_ms();

        cout << "  File load (read + decompress + malloc + memcpy): "
             << fixed << setprecision(2) << load_time << " ms" << endl;
        cout << "  GPU setup (warmup only): "
             << setup_time << " ms" << endl;
        cout << "  TOTAL: " << (load_time + setup_time) << " ms" << endl;
    }
    cout << endl;

    // ========================================================================
    // STEP 5: Measure individual components
    // ========================================================================
    cout << "=== Component Analysis ===" << endl;

    // Measure raw file read time (no processing)
    {
        Timer t;
        FILE* f = fopen(cpu_file.c_str(), "rb");
        char* buf = new char[cpu_size];
        size_t r = fread(buf, 1, cpu_size, f);
        (void)r;
        fclose(f);
        delete[] buf;
        cout << "Raw file read (CPU format, " << cpu_size/1024.0 << " KB): "
             << t.elapsed_ms() << " ms" << endl;
    }

    {
        Timer t;
        FILE* f = fopen(gpu_file.c_str(), "rb");
        char* buf = new char[gpu_size];
        size_t r = fread(buf, 1, gpu_size, f);
        (void)r;
        fclose(f);
        delete[] buf;
        cout << "Raw file read (GPU format, " << gpu_size/1024.0 << " KB): "
             << t.elapsed_ms() << " ms" << endl;
    }

    // Measure cudaMemcpy time for dataset
    {
        size_t padded_veclen = 128;  // Already aligned
        size_t data_size = dataset.rows * padded_veclen * sizeof(float);
        float* d_data;
        float* h_data = new float[dataset.rows * padded_veclen];
        memset(h_data, 0, data_size);

        cudaMalloc(&d_data, data_size);
        cudaDeviceSynchronize();

        Timer t;
        cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        double memcpy_time = t.elapsed_ms();

        cout << "cudaMemcpy dataset (" << data_size/1024.0 << " KB): "
             << memcpy_time << " ms" << endl;

        cudaFree(d_data);
        delete[] h_data;
    }

    // Measure cudaMalloc time for various sizes
    {
        float* d_data;

        Timer t;
        cudaMalloc(&d_data, 2 * 1024 * 1024);  // 2 MB
        cudaDeviceSynchronize();
        cout << "cudaMalloc (2 MB): " << t.elapsed_ms() << " ms" << endl;
        cudaFree(d_data);
    }

    // Measure vector allocation time
    {
        Timer t;
        std::vector<float> v(dataset.rows * 128);
        cout << "std::vector<float> alloc (" << (dataset.rows * 128 * 4)/1024.0 << " KB): "
             << t.elapsed_ms() << " ms" << endl;
    }

    cout << endl;
    cout << "=== Summary ===" << endl;
    cout << "CPU format load includes:" << endl;
    cout << "  1. File read + LZ4 decompress" << endl;
    cout << "  2. Tree node reconstruction (malloc nodes, build pointers)" << endl;
    cout << "  3. Dataset unpack to points_" << endl;
    cout << endl;
    cout << "GPU format load includes:" << endl;
    cout << "  1. File read + LZ4 decompress" << endl;
    cout << "  2. Allocate 4 host vectors (node_index, variance, pivots, dataset)" << endl;
    cout << "  3. Read data into host vectors" << endl;
    cout << "  4. cudaMalloc x4" << endl;
    cout << "  5. cudaMemcpy x4 (host to device)" << endl;
    cout << "  6. Reconstruct points_ from padded dataset" << endl;
    cout << endl;
    cout << "For small datasets, the cudaMalloc + cudaMemcpy overhead dominates." << endl;

    // Cleanup
    delete[] dataset.ptr();
    delete[] queries.ptr();
    remove(cpu_file.c_str());
    remove(gpu_file.c_str());

    return 0;
}
