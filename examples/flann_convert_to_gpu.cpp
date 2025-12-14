/**
 * FLANN Index Conversion Utility
 * Converts CPU format index files to GPU format for faster loading.
 *
 * The GPU format stores flattened tree arrays directly, enabling fast
 * GPU uploads without CPU tree reconstruction on load.
 *
 * Usage:
 *   flann_convert_to_gpu <input_file> <output_file> [options]
 *
 * Options:
 *   --index-type=<type>  Force index type: kmeans, hierarchical (auto-detected if not specified)
 *   --k=<value>          K value for GPU search setup (default: 10)
 *   --checks=<value>     Search checks parameter (default: 128 for kmeans, 2000 for hierarchical)
 *   --verify             Verify conversion by comparing search results
 *   --verbose            Print detailed information
 *
 * Requirements:
 *   - Input file must have save_dataset=true (dataset embedded in file)
 *   - Only KMeans and Hierarchical indices are supported for GPU conversion
 *
 * Copyright 2024. BSD License.
 */

#ifndef FLANN_USE_CUDA
#error "This utility requires FLANN_USE_CUDA to be defined. Build with -DBUILD_CUDA_LIB=ON"
#endif

#include <flann/flann.hpp>
#include <iostream>
#include <cstring>
#include <chrono>

using flann::FLANN_INDEX_KMEANS;
using flann::FLANN_INDEX_KMEANS_CUDA;
using flann::FLANN_INDEX_HIERARCHICAL;
using flann::FLANN_INDEX_HIERARCHICAL_CUDA;
using flann::flann_algorithm_t;
using flann::flann_datatype_t;

void printUsage(const char* prog) {
    std::cerr << "FLANN Index GPU Conversion Utility\n\n"
              << "Usage: " << prog << " <input_file> <output_file> [options]\n"
              << "\nOptions:\n"
              << "  --index-type=<type>  Force index type: kmeans, hierarchical\n"
              << "  --k=<value>          K value for GPU search setup (default: 10)\n"
              << "  --checks=<value>     Search checks parameter (default: auto)\n"
              << "  --verify             Verify conversion by comparing search results\n"
              << "  --verbose            Print detailed information\n"
              << "\nExample:\n"
              << "  " << prog << " index.db index.gpu.db --verify --verbose\n";
}

struct FileHeader {
    char signature[24];
    char version[16];
    flann_datatype_t data_type;
    flann_algorithm_t index_type;
    size_t rows;
    size_t cols;
};

bool readFileHeader(const char* filename, FileHeader& header) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    bool ok = fread(&header, sizeof(header), 1, f) == 1;
    fclose(f);
    return ok;
}

size_t getFileSize(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fclose(f);
    return size;
}

template<typename Distance>
bool convertIndex(const char* input, const char* output,
                  int k, int checks, bool verify, bool verbose,
                  size_t expected_rows, size_t expected_cols) {
    using ElementType = typename Distance::ElementType;
    using DistanceType = typename Distance::ResultType;

    auto start = std::chrono::high_resolution_clock::now();

    // Load CPU format (dataset must be embedded in file)
    if (verbose) std::cout << "Loading CPU format index from " << input << "...\n";

    // Create empty dataset - will be populated from saved file
    flann::Matrix<ElementType> dataset;
    flann::Index<Distance> index(dataset, flann::SavedIndexParams(input));

    auto load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - start).count();

    if (verbose) {
        std::cout << "  Loaded: " << index.size() << " points, " << index.veclen() << " dimensions\n";
        std::cout << "  Load time: " << load_ms << " ms\n";
    }

    // Validate dimensions
    if (index.size() != expected_rows || index.veclen() != expected_cols) {
        std::cerr << "Warning: Loaded dimensions (" << index.size() << "x" << index.veclen()
                  << ") differ from header (" << expected_rows << "x" << expected_cols << ")\n";
    }

    // Run search before conversion (for verification)
    flann::Matrix<size_t> indices_before;
    flann::Matrix<DistanceType> dists_before;
    size_t num_queries = std::min(size_t(10), index.size());

    if (verify && num_queries > 0) {
        indices_before = flann::Matrix<size_t>(new size_t[num_queries * k], num_queries, k);
        dists_before = flann::Matrix<DistanceType>(new DistanceType[num_queries * k], num_queries, k);

        // Use first points as queries
        flann::Matrix<ElementType> queries(const_cast<ElementType*>(index.getPoint(0)),
                                           num_queries, index.veclen());
        index.knnSearch(queries, indices_before, dists_before, k, flann::SearchParams(checks));

        if (verbose) std::cout << "  Ran verification search (CPU): " << num_queries << " queries\n";
    }

    // Build GPU search structures
    if (verbose) std::cout << "Building GPU search structures...\n";
    auto gpu_start = std::chrono::high_resolution_clock::now();
    index.buildCUDAKnnSearch(k, flann::SearchParams(checks));
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    if (verbose) std::cout << "  GPU build time: " << gpu_ms << " ms\n";

    // Convert to GPU-only format (discards CPU tree)
    if (verbose) std::cout << "Converting to GPU-only format...\n";
    index.convertToGPUFormat();

    // Save GPU format
    if (verbose) std::cout << "Saving GPU format to " << output << "...\n";
    auto save_start = std::chrono::high_resolution_clock::now();
    index.save(output);
    auto save_end = std::chrono::high_resolution_clock::now();
    double save_ms = std::chrono::duration<double, std::milli>(save_end - save_start).count();
    if (verbose) std::cout << "  Save time: " << save_ms << " ms\n";

    // Report sizes
    size_t input_size = getFileSize(input);
    size_t output_size = getFileSize(output);
    std::cout << "\nConversion complete!\n"
              << "  Input (CPU):  " << input_size << " bytes (" << (input_size / 1024.0 / 1024.0) << " MB)\n"
              << "  Output (GPU): " << output_size << " bytes (" << (output_size / 1024.0 / 1024.0) << " MB)\n"
              << "  Size ratio:   " << (float)output_size / input_size << "x\n";

    // Verify if requested
    if (verify && num_queries > 0) {
        if (verbose) std::cout << "\nVerifying conversion...\n";

        // Load GPU format
        flann::Matrix<ElementType> dataset2;
        flann::Index<Distance> loaded(dataset2, flann::SavedIndexParams(output));
        loaded.buildCUDAKnnSearch(k, flann::SearchParams(checks));

        // Run search after conversion
        flann::Matrix<size_t> indices_after(new size_t[num_queries * k], num_queries, k);
        flann::Matrix<DistanceType> dists_after(new DistanceType[num_queries * k], num_queries, k);

        flann::Matrix<ElementType> queries(const_cast<ElementType*>(loaded.getPoint(0)),
                                           num_queries, loaded.veclen());
        loaded.knnSearch(queries, indices_after, dists_after, k, flann::SearchParams(checks));

        // Compare results using precision (not exact match, since GPU may find better neighbors)
        // Count how many GPU results are in top-k CPU results (precision)
        int correct = 0;
        int total = num_queries * k;
        int better_count = 0;
        int worse_count = 0;

        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < (size_t)k; ++j) {
                bool found = false;
                for (size_t m = 0; m < (size_t)k; ++m) {
                    if (indices_after[i][j] == indices_before[i][m]) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    correct++;
                } else {
                    // Check if GPU found a better result (lower distance at same position)
                    if (dists_after[i][j] < dists_before[i][j]) {
                        better_count++;
                    } else if (dists_after[i][j] > dists_before[i][j]) {
                        worse_count++;
                        if (verbose && worse_count <= 3) {
                            std::cerr << "  Worse result at query " << i << " neighbor " << j
                                      << ": " << indices_before[i][j] << " (dist " << dists_before[i][j]
                                      << ") vs " << indices_after[i][j] << " (dist " << dists_after[i][j] << ")\n";
                        }
                    }
                }
            }
        }

        delete[] indices_before.ptr();
        delete[] dists_before.ptr();
        delete[] indices_after.ptr();
        delete[] dists_after.ptr();

        float precision = (float)(correct + better_count) / total;
        std::cout << "Verification results:\n"
                  << "  Matching results:  " << correct << "/" << total << "\n"
                  << "  Better (GPU):      " << better_count << "\n"
                  << "  Worse (GPU):       " << worse_count << "\n"
                  << "  Effective precision: " << (precision * 100) << "%\n";

        // Pass if precision >= 90% (allow for search algorithm differences)
        if (precision >= 0.90) {
            std::cout << "Verification: PASSED\n";
        } else {
            std::cout << "Verification: FAILED (precision too low)\n";
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];

    // Parse options
    std::string index_type_str;
    int k = 10;
    int checks = -1;  // Auto-select based on index type
    bool verify = false;
    bool verbose = false;

    for (int i = 3; i < argc; ++i) {
        if (strncmp(argv[i], "--index-type=", 13) == 0) {
            index_type_str = argv[i] + 13;
        } else if (strncmp(argv[i], "--k=", 4) == 0) {
            k = atoi(argv[i] + 4);
        } else if (strncmp(argv[i], "--checks=", 9) == 0) {
            checks = atoi(argv[i] + 9);
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    try {
        // Read file header
        FileHeader header;
        if (!readFileHeader(input, header)) {
            std::cerr << "Error: Cannot read file: " << input << "\n";
            return 1;
        }

        // Check for GPU format (shouldn't convert GPU to GPU)
        if (strncmp(header.signature, "FLANN_GPU_INDEX", 15) == 0) {
            std::cerr << "Error: File is already in GPU format\n";
            return 1;
        }

        // Check for valid CPU format
        if (strncmp(header.signature, "FLANN_INDEX", 11) != 0) {
            std::cerr << "Error: Invalid FLANN index file (signature: " << std::string(header.signature, 20) << ")\n";
            return 1;
        }

        // Determine index type
        flann_algorithm_t index_type = header.index_type;
        if (!index_type_str.empty()) {
            if (index_type_str == "kmeans") {
                index_type = FLANN_INDEX_KMEANS;
            } else if (index_type_str == "hierarchical") {
                index_type = FLANN_INDEX_HIERARCHICAL;
            } else {
                std::cerr << "Unknown index type: " << index_type_str << "\n";
                return 1;
            }
        }

        // Auto-select checks if not specified
        if (checks < 0) {
            checks = (index_type == FLANN_INDEX_HIERARCHICAL ||
                      index_type == FLANN_INDEX_HIERARCHICAL_CUDA) ? 2000 : 128;
        }

        std::cout << "FLANN Index GPU Conversion\n"
                  << "  Input:  " << input << "\n"
                  << "  Output: " << output << "\n"
                  << "  Index type: ";
        switch (index_type) {
            case FLANN_INDEX_KMEANS: std::cout << "KMeans (L2)\n"; break;
            case FLANN_INDEX_KMEANS_CUDA: std::cout << "KMeans CUDA (L2)\n"; break;
            case FLANN_INDEX_HIERARCHICAL: std::cout << "Hierarchical (Hamming)\n"; break;
            case FLANN_INDEX_HIERARCHICAL_CUDA: std::cout << "Hierarchical CUDA (Hamming)\n"; break;
            default: std::cout << "Unknown (" << index_type << ")\n"; break;
        }
        std::cout << "  Dataset: " << header.rows << " points x " << header.cols << " dims\n"
                  << "  Parameters: k=" << k << ", checks=" << checks << "\n\n";

        bool success = false;
        switch (index_type) {
            case FLANN_INDEX_KMEANS:
            case FLANN_INDEX_KMEANS_CUDA: {
                success = convertIndex<flann::L2<float>>(
                    input, output, k, checks, verify, verbose, header.rows, header.cols);
                break;
            }
            case FLANN_INDEX_HIERARCHICAL:
            case FLANN_INDEX_HIERARCHICAL_CUDA: {
                success = convertIndex<flann::Hamming<unsigned char>>(
                    input, output, k, checks, verify, verbose, header.rows, header.cols);
                break;
            }
            default:
                std::cerr << "Unsupported index type for GPU conversion: " << index_type << "\n";
                std::cerr << "Supported types: FLANN_INDEX_KMEANS (2), FLANN_INDEX_KMEANS_CUDA (8), "
                          << "FLANN_INDEX_HIERARCHICAL (5), FLANN_INDEX_HIERARCHICAL_CUDA (9)\n";
                return 1;
        }

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
