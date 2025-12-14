/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2024. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_GPU_SAVING_H_
#define FLANN_GPU_SAVING_H_

#include <cstring>
#include <stdio.h>

#include "flann/general.h"
#include "flann/util/serialization.h"

#ifdef FLANN_GPU_SIGNATURE_
#undef FLANN_GPU_SIGNATURE_
#endif
#define FLANN_GPU_SIGNATURE_ "FLANN_GPU_INDEX_v1.0"
#define FLANN_GPU_SIGNATURE_V2_ "FLANN_GPU_INDEX_v2.0"

namespace flann
{

/**
 * Structure representing the GPU index header.
 * Contains all metadata needed to restore GPU arrays directly.
 */
struct GPUIndexHeaderStruct {
    // Standard header fields (compatibility with IndexHeaderStruct layout)
    char signature[24];           // "FLANN_GPU_INDEX_v1.0"
    char version[16];             // FLANN version
    flann_datatype_t data_type;   // Element type (float, unsigned char, etc.)
    flann_algorithm_t index_type; // FLANN_INDEX_*_GPU_SAVED
    size_t rows;                  // Dataset size (number of points)
    size_t cols;                  // Original dimension (veclen_)
    /**
     * Compression flag (0 = uncompressed, 1 = LZ4).
     * IMPORTANT: Initialize to 0 in header constructors.
     * The SaveArchive sets this to 1 during serialization.
     */
    size_t compression;
    size_t first_block_size;      // For LZ4 streaming compatibility

    // GPU-specific fields
    size_t padded_veclen;         // Padded dimension for GPU alignment (multiple of 4)
    size_t num_nodes;             // Number of tree nodes
    size_t node_index_size;       // Size of unified nodeIndex array
    size_t leaf_count;            // Total leaf dataset points (KMeans) or num_trees (Hierarchical)

    // Algorithm parameters
    int branching;                // Branching factor
    int iterations;               // K-means iterations (KMeans only)
    float cb_index;               // Cluster boundary index (KMeans only)
    int trees;                    // Number of trees (Hierarchical only)
    int leaf_max_size;            // Max leaf size (Hierarchical only)
    flann_centers_init_t centers_init; // Center initialization method

    // Reserved for future use
    size_t reserved[8];
};

/**
 * RAII wrapper for GPU index header with serialization support.
 */
struct GPUIndexHeader
{
    GPUIndexHeaderStruct h;

    GPUIndexHeader()
    {
        memset(&h, 0, sizeof(h));
        strncpy(h.signature, FLANN_GPU_SIGNATURE_, sizeof(h.signature) - 1);
        strncpy(h.version, FLANN_VERSION_, sizeof(h.version) - 1);
        h.compression = 0;  // Will be set to 1 by SaveArchive during compression
        h.first_block_size = 0;
    }

    /**
     * Check if this header indicates GPU format
     */
    bool isGPUFormat() const
    {
        return strncmp(h.signature, "FLANN_GPU_INDEX", strlen("FLANN_GPU_INDEX")) == 0;
    }

    /**
     * Detect GPU format by peeking at file signature without consuming stream.
     *
     * @param stream Input file stream
     * @return true if file appears to be GPU format
     */
    static bool detectGPUFormat(FILE* stream)
    {
        long pos = ftell(stream);
        char sig[24];
        memset(sig, 0, sizeof(sig));

        if (fread(sig, sizeof(sig), 1, stream) != 1) {
            fseek(stream, pos, SEEK_SET);
            return false;
        }

        fseek(stream, pos, SEEK_SET);
        return strncmp(sig, "FLANN_GPU_INDEX", strlen("FLANN_GPU_INDEX")) == 0;
    }

private:
    template<typename Archive>
    void serialize(Archive& ar)
    {
        // Standard header fields
        ar & h.signature;
        ar & h.version;
        ar & h.data_type;
        ar & h.index_type;
        ar & h.rows;
        ar & h.cols;
        ar & h.compression;
        ar & h.first_block_size;

        // GPU-specific fields
        ar & h.padded_veclen;
        ar & h.num_nodes;
        ar & h.node_index_size;
        ar & h.leaf_count;
        ar & h.branching;
        ar & h.iterations;
        ar & h.cb_index;
        ar & h.trees;
        ar & h.leaf_max_size;
        ar & h.centers_init;

        // Reserved fields for future compatibility
        ar & serialization::make_binary_object(h.reserved, sizeof(h.reserved));
    }
    friend struct serialization::access;
};

/**
 * Structure representing the GPU index v2.0 header.
 * Stores pre-computed workspace layout for zero-computation loading.
 * Data is stored as a single contiguous blob with alignment padding included.
 */
struct GPUIndexHeaderV2Struct {
    // Standard header fields (compatibility with IndexHeaderStruct layout)
    char signature[24];           // "FLANN_GPU_INDEX_v2.0"
    char version[16];             // FLANN version
    flann_datatype_t data_type;   // Element type (float, unsigned char, etc.)
    flann_algorithm_t index_type; // FLANN_INDEX_*_GPU_SAVED
    size_t rows;                  // Dataset size (number of points)
    size_t cols;                  // Original dimension (veclen_)
    size_t compression;           // 0 = uncompressed, 1 = LZ4
    size_t first_block_size;      // For LZ4 streaming compatibility

    // GPU-specific fields (same as v1.0)
    size_t padded_veclen;         // Padded dimension for GPU alignment
    size_t num_nodes;             // Number of tree nodes
    size_t node_index_size;       // Size of unified nodeIndex array (in ints)
    size_t leaf_count;            // Leaf dataset points (KMeans) or num_trees (Hierarchical)

    // Algorithm parameters (same as v1.0)
    int branching;
    int iterations;
    float cb_index;
    int trees;
    int leaf_max_size;
    flann_centers_init_t centers_init;

    // =====================================================
    // NEW v2.0 fields: Pre-computed workspace layout
    // Eliminates all alignment computation at load time
    // =====================================================
    size_t workspace_total_size;  // Total bytes for GPU workspace blob
    size_t offset_node_index;     // Offset to node_index array (always 0 for KMeans)
    size_t offset_variance;       // Offset to variance array (KMeans only, 0 for Hierarchical)
    size_t offset_pivots;         // Offset to pivots array (KMeans only, 0 for Hierarchical)
    size_t offset_dataset;        // Offset to dataset array

    // Reserved for future use
    size_t reserved[4];
};

/**
 * RAII wrapper for GPU index v2.0 header with serialization support.
 */
struct GPUIndexHeaderV2
{
    GPUIndexHeaderV2Struct h;

    GPUIndexHeaderV2()
    {
        memset(&h, 0, sizeof(h));
        strncpy(h.signature, FLANN_GPU_SIGNATURE_V2_, sizeof(h.signature) - 1);
        strncpy(h.version, FLANN_VERSION_, sizeof(h.version) - 1);
        h.compression = 0;  // Will be set to 1 by SaveArchive during compression
        h.first_block_size = 0;
    }

    /**
     * Check if this header indicates v2.0 GPU format
     */
    bool isV2Format() const
    {
        return strncmp(h.signature, FLANN_GPU_SIGNATURE_V2_, strlen(FLANN_GPU_SIGNATURE_V2_)) == 0;
    }

    /**
     * Detect v2.0 format by peeking at file signature without consuming stream.
     */
    static bool detectV2Format(FILE* stream)
    {
        long pos = ftell(stream);
        char sig[24];
        memset(sig, 0, sizeof(sig));

        if (fread(sig, sizeof(sig), 1, stream) != 1) {
            fseek(stream, pos, SEEK_SET);
            return false;
        }

        fseek(stream, pos, SEEK_SET);
        return strncmp(sig, FLANN_GPU_SIGNATURE_V2_, strlen(FLANN_GPU_SIGNATURE_V2_)) == 0;
    }

private:
    template<typename Archive>
    void serialize(Archive& ar)
    {
        // Standard header fields
        ar & h.signature;
        ar & h.version;
        ar & h.data_type;
        ar & h.index_type;
        ar & h.rows;
        ar & h.cols;
        ar & h.compression;
        ar & h.first_block_size;

        // GPU-specific fields
        ar & h.padded_veclen;
        ar & h.num_nodes;
        ar & h.node_index_size;
        ar & h.leaf_count;
        ar & h.branching;
        ar & h.iterations;
        ar & h.cb_index;
        ar & h.trees;
        ar & h.leaf_max_size;
        ar & h.centers_init;

        // v2.0 workspace layout fields
        ar & h.workspace_total_size;
        ar & h.offset_node_index;
        ar & h.offset_variance;
        ar & h.offset_pivots;
        ar & h.offset_dataset;

        // Reserved fields
        ar & serialization::make_binary_object(h.reserved, sizeof(h.reserved));
    }
    friend struct serialization::access;
};

/**
 * Detect GPU format version from file stream.
 * @return 0 = not GPU format, 1 = v1.0, 2 = v2.0
 */
inline int detectGPUFormatVersion(FILE* stream)
{
    long pos = ftell(stream);
    char sig[24];
    memset(sig, 0, sizeof(sig));

    if (fread(sig, sizeof(sig), 1, stream) != 1) {
        fseek(stream, pos, SEEK_SET);
        return 0;
    }

    fseek(stream, pos, SEEK_SET);

    if (strncmp(sig, FLANN_GPU_SIGNATURE_V2_, strlen(FLANN_GPU_SIGNATURE_V2_)) == 0) {
        return 2;
    }
    if (strncmp(sig, "FLANN_GPU_INDEX", strlen("FLANN_GPU_INDEX")) == 0) {
        return 1;
    }
    return 0;
}

/**
 * Save GPU index header to stream (for debugging/verification only).
 * Normal save should use SaveArchive serialization.
 */
inline void save_gpu_header(FILE* stream, const GPUIndexHeader& header)
{
    fwrite(&header.h, sizeof(header.h), 1, stream);
}

/**
 * Load GPU index header from stream (raw read, for format detection).
 */
inline GPUIndexHeader load_gpu_header(FILE* stream)
{
    GPUIndexHeader header;
    int read_size = fread(&header.h, sizeof(header.h), 1, stream);

    if (read_size != 1) {
        throw FLANNException("Invalid GPU index file, cannot read header");
    }

    if (!header.isGPUFormat()) {
        throw FLANNException("Invalid GPU index file, wrong signature");
    }

    return header;
}

} // namespace flann

#endif /* FLANN_GPU_SAVING_H_ */
