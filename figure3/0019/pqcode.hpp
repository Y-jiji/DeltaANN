#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

/** @brief Dimensions per band of the fnct codebook `pqtrain` writes */
constexpr long PQ_BAND = 8;
/** @brief Centroids per band, 8 code bits */
constexpr long PQ_KS = 256;

/**
 * @brief A fnct product quantization codebook read from `pqtrain`'s raw little-endian f32 file,
 * centroid `i` of band `c` at `books[(c * PQ_KS + i) * PQ_BAND]`, dims past `ndims` padded with 0,
 * codes taken against the zero point; `pairs` holds every band's centroid-pair squared distance
 */
class PQCode {
  public:
    long ndims;
    long nband;
    std::vector<float> books;
    std::vector<float> pairs;

    /**
     * @brief Read the codebook and build the centroid-pair table
     * @param path raw f32 codebook written by `pqtrain` with BAND 8 and BITS 8
     * @param dims dimension of the points it codes
     */
    PQCode(const char* path, long dims);

    /**
     * @brief Code of one point, the nearest centroid of every band
     * @tparam T element type of the point
     * @param point `ndims` elements
     * @param code `nband` bytes written
     */
    template <typename T> void encode(const T* point, uint8_t* code) const;

    /**
     * @brief Symmetric distance, squared L2 between the centroids two codes name
     * @param a code of `nband` bytes
     * @param b code of `nband` bytes
     * @return the sum over bands of the centroid-pair squared distance
     */
    float sdc(const uint8_t* a, const uint8_t* b) const;

    /**
     * @brief Asymmetric lookup table of a query, squared L2 from each band of it to every centroid
     * @param query `ndims` floats
     * @param lut `nband * PQ_KS` floats written
     */
    void lookup(const float* query, float* lut) const;

    /**
     * @brief Asymmetric distance from a query's lookup table to a code
     * @param lut table from `lookup`
     * @param code code of `nband` bytes
     * @return the sum over bands of the table entry the code names
     */
    float adc(const float* lut, const uint8_t* code) const;
};

inline PQCode::PQCode(const char* path, long dims)
    : ndims(dims), nband((dims + PQ_BAND - 1) / PQ_BAND) {
    // [1]
    books.resize(nband * PQ_KS * PQ_BAND);
    FILE* file = std::fopen(path, "rb");
    size_t count = 0;
    if (file != nullptr) count = std::fread(books.data(), sizeof(float), books.size(), file);
    if (count != books.size()) {
        std::fprintf(stderr, "codebook %s: cannot read %zu floats\n", path, books.size());
        std::abort();
    }
    std::fclose(file);
    // [2]
    pairs.resize(nband * PQ_KS * PQ_KS);
    for (long c = 0; c < nband; c++) {
        for (long i = 0; i < PQ_KS; i++) {
            for (long j = 0; j < PQ_KS; j++) {
                float sum = 0;
                for (long d = 0; d < PQ_BAND; d++) {
                    float diff = books[(c * PQ_KS + i) * PQ_BAND + d]
                               - books[(c * PQ_KS + j) * PQ_BAND + d];
                    sum += diff * diff;
                }
                pairs[(c * PQ_KS + i) * PQ_KS + j] = sum;
            }
        }
    }
}

template <typename T> void PQCode::encode(const T* point, uint8_t* code) const {
    for (long c = 0; c < nband; c++) {
        // [1]
        float cast[PQ_BAND] = {0};
        for (long d = 0; d < PQ_BAND && c * PQ_BAND + d < ndims; d++) {
            cast[d] = (float) point[c * PQ_BAND + d];
        }
        // [2]
        float best = std::numeric_limits<float>::infinity();
        for (long i = 0; i < PQ_KS; i++) {
            float sum = 0;
            for (long d = 0; d < PQ_BAND; d++) {
                float diff = cast[d] - books[(c * PQ_KS + i) * PQ_BAND + d];
                sum += diff * diff;
            }
            if (sum < best) {
                best = sum;
                code[c] = (uint8_t) i;
            }
        }
    }
}

inline float PQCode::sdc(const uint8_t* a, const uint8_t* b) const {
    float sum = 0;
    for (long c = 0; c < nband; c++) {
        sum += pairs[(c * PQ_KS + a[c]) * PQ_KS + b[c]];
    }
    return sum;
}

inline void PQCode::lookup(const float* query, float* lut) const {
    for (long c = 0; c < nband; c++) {
        // [1]
        float cast[PQ_BAND] = {0};
        for (long d = 0; d < PQ_BAND && c * PQ_BAND + d < ndims; d++) {
            cast[d] = query[c * PQ_BAND + d];
        }
        // [2]
        for (long i = 0; i < PQ_KS; i++) {
            float sum = 0;
            for (long d = 0; d < PQ_BAND; d++) {
                float diff = cast[d] - books[(c * PQ_KS + i) * PQ_BAND + d];
                sum += diff * diff;
            }
            lut[c * PQ_KS + i] = sum;
        }
    }
}

inline float PQCode::adc(const float* lut, const uint8_t* code) const {
    float sum = 0;
    for (long c = 0; c < nband; c++) {
        sum += lut[c * PQ_KS + code[c]];
    }
    return sum;
}
