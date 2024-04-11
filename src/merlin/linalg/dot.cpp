// Copyright 2023 quocdang1998
#include "merlin/linalg/dot.hpp"

#include <array>  // std::array
#include <cmath>  // std::fma, std::sqrt

#include "merlin/avx.hpp"     // merlin::AvxDouble, merlin::use_avx
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
    AvxDouble<use_avx> chunk_result;
    AvxDouble<use_avx> clone_data;
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        clone_data = AvxDouble<use_avx>(vector);
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

// Normalize a contiguous vector
void linalg::normalize(const double * src_vector, double * dest_vector, std::uint64_t size) noexcept {
    // calculate square root of the norm
    double norm;
    linalg::norm(src_vector, size, norm);
    norm = std::sqrt(norm);
    // divide by the norm on 4-double chunks
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    AvxDouble<use_avx> chunk_data;
    AvxDouble<use_avx> chunk_norm(norm);
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        chunk_data = AvxDouble<use_avx>(src_vector);
        chunk_data.divide(chunk_norm);
        chunk_data.store(dest_vector);
        src_vector += 4;
        dest_vector += 4;
    }
    // divide remainder
    for (std::uint64_t i = 0; i < remainder; i++) {
        dest_vector[i] = src_vector[i] / norm;
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
    AvxDouble<use_avx> chunk_result;
    AvxDouble<use_avx> chunk_vec1, chunk_vec2;
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        chunk_vec1 = AvxDouble<use_avx>(vector1);
        chunk_vec2 = AvxDouble<use_avx>(vector2);
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
// Vector operation
// ---------------------------------------------------------------------------------------------------------------------

// Vector operation with another vector
void linalg::saxpy(double a, const double * x, double * y, std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    // performing linear update using avx on 4-double chunks
    AvxDouble<use_avx> chunk_x, chunk_y;
    AvxDouble<use_avx> chunk_a(a);
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        chunk_x = AvxDouble<use_avx>(x);
        chunk_y = AvxDouble<use_avx>(y);
        chunk_y.fma(chunk_a, chunk_x);
        chunk_y.store(y);
        x += 4;
        y += 4;
    }
    // add remainder
    for (std::uint64_t i = 0; i < remainder; i++) {
        y[i] = std::fma(a, x[i], y[i]);
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
    AvxDouble<use_avx> factor(inner_prod);
    AvxDouble<use_avx> chunk_reflector, chunk_target;
    for (std::uint64_t i_chunk = 0; i_chunk < num_chunks; i_chunk++) {
        chunk_reflector = AvxDouble<use_avx>(reflector);
        chunk_target = AvxDouble<use_avx>(chunk_target);
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
