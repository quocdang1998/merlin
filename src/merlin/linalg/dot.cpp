// Copyright 2023 quocdang1998
#include "merlin/linalg/dot.hpp"

#include <array>  // std::array
#include <cmath>  // std::fma

#include "merlin/avx.hpp"     // merlin::PackedDouble, merlin::use_avx
#include "merlin/logger.hpp"  // WARNING

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Norm
// ---------------------------------------------------------------------------------------------------------------------

// Norm of contiguous vector
void linalg::norm(const double * vector, std::uint64_t size, double & result) noexcept {
    result = 0.0;
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    // calculate norm using avx on 4-double chunks
    PackedDouble<use_avx> chunk_result;
    PackedDouble<use_avx> clone_data;
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        clone_data = PackedDouble<use_avx>(vector);
        chunk_result.fma(clone_data, clone_data);
        vector += 4;
    }
    for (std::uint64_t i = 0; i < 4; i++) {
        result += chunk_result[i];
    }
    // add remainder
    for (std::uint64_t i = 0; i < remainder; i++) {
        result = std::fma(vector[i], vector[i], result);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Dot Product
// ---------------------------------------------------------------------------------------------------------------------

// Dot product of contiguous vectors
void linalg::dot(const double * vector1, const double * vector2, std::uint64_t size, double & result) noexcept {
    result = 0.0;
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    // calculate dot product using avx on 4-double chunks
    PackedDouble<use_avx> chunk_result;
    PackedDouble<use_avx> chunk_vec1, chunk_vec2;
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        chunk_vec1 = PackedDouble<use_avx>(vector1);
        chunk_vec2 = PackedDouble<use_avx>(vector2);
        chunk_result.fma(chunk_vec1, chunk_vec2);
        vector1 += 4;
        vector2 += 4;
    }
    for (std::uint64_t i = 0; i < 4; i++) {
        result += chunk_result[i];
    }
    // add remainder
    for (std::uint64_t i = 0; i < remainder; i++) {
        result = std::fma(vector1[i], vector2[i], result);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Householder Reflection
// ---------------------------------------------------------------------------------------------------------------------

// Apply Householder reflection on a vector
void linalg::householder(const double * reflector, double * target, std::uint64_t size, std::uint64_t range) noexcept {
    // calculate dot product between reflector and target
    double inner_prod;
    linalg::dot(reflector, target, size, inner_prod);
    inner_prod *= -2.0;
    // apply reflection by chunks
    std::uint64_t num_chunks = range / 4, remainder = range % 4;
    PackedDouble<use_avx> factor(inner_prod);
    PackedDouble<use_avx> chunk_reflector, chunk_target;
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        chunk_reflector = PackedDouble<use_avx>(reflector);
        chunk_target = PackedDouble<use_avx>(chunk_target);
        chunk_target.fma(factor, chunk_reflector);
        chunk_target.store(target);
        target += 4;
        reflector += 4;
    }
    // apply reflection on remainder
    for (std::uint64_t i = 0; i < remainder; i++) {
        target[i] = std::fma(inner_prod, reflector[i], target[i]);
    }
}

}  // namespace merlin
