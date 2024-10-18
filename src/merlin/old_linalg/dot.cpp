// Copyright 2023 quocdang1998
#include "merlin/old_linalg/dot.hpp"

#include <array>    // std::array
#include <cmath>    // std::fma
#include <utility>  // std::swap

#include "merlin/avx.hpp"  // merlin::AvxDouble, merlin::use_avx

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Vector operation
// ---------------------------------------------------------------------------------------------------------------------

// Multiply a vector by a scalar and store it into another vector
void linalg::avx_multiply(double a, const double * x, double * y, std::uint64_t nchunks,
                          std::uint64_t remain) noexcept {
    // performing linear multiplication using avx on 4-double chunks
    AvxDouble<use_avx> chunk_x;
    AvxDouble<use_avx> chunk_a(a);
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
        chunk_x = AvxDouble<use_avx>(x);
        chunk_x.mult(chunk_a);
        chunk_x.store(y);
        x += 4;
        y += 4;
    }
    // add remainder
    for (std::uint64_t i = 0; i < remain; i++) {
        y[i] = a * x[i];
    }
}

// Vector operation with another vector
void linalg::avx_saxpy(double a, const double * x, double * y, std::uint64_t nchunks, std::uint64_t remain) noexcept {
    // performing linear update using avx on 4-double chunks
    AvxDouble<use_avx> chunk_x, chunk_y;
    AvxDouble<use_avx> chunk_a(a);
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
        chunk_x = AvxDouble<use_avx>(x);
        chunk_y = AvxDouble<use_avx>(y);
        chunk_y.fma(chunk_a, chunk_x);
        chunk_y.store(y);
        x += 4;
        y += 4;
    }
    // add remainder
    for (std::uint64_t i = 0; i < remain; i++) {
        y[i] = std::fma(a, x[i], y[i]);
    }
}

// Swap contents of 2 vectors
void linalg::avx_swap(double * x, double * y, std::uint64_t nchunks, std::uint64_t remain) noexcept {
    // performing swap using avx on 4-double chunks
    AvxDouble<use_avx> chunk_x, chunk_y;
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
        chunk_x = AvxDouble<use_avx>(x);
        chunk_y = AvxDouble<use_avx>(y);
        chunk_x.store(y);
        chunk_y.store(x);
        x += 4;
        y += 4;
    }
    // swap remainder
    for (std::uint64_t i = 0; i < remain; i++) {
        std::swap(x[i], y[i]);
    }
}

// Divide contents of 2 vectors.
void linalg::avx_vecdiv(const double * x, const double * y, double * z, std::uint64_t nchunks,
                        std::uint64_t remain) noexcept {
    // performing division
    AvxDouble<use_avx> chunk_dividend, chunk_divisor;
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
        chunk_dividend = AvxDouble<use_avx>(x);
        chunk_divisor = AvxDouble<use_avx>(y);
        chunk_dividend.divide(chunk_divisor);
        chunk_dividend.store(z);
        x += 4;
        y += 4;
        z += 4;
    }
    // divide remainder
    for (std::uint64_t i = 0; i < remain; i++) {
        z[i] = x[i] / y[i];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Dot Product and Norm
// ---------------------------------------------------------------------------------------------------------------------

// Dot product of contiguous vectors
void linalg::avx_dot(const double * vector1, const double * vector2, std::uint64_t nchunks, std::uint64_t remain,
                     double & result) noexcept {
    result = 0.0;
    // calculate dot product using avx on 4-double chunks
    AvxDouble<use_avx> chunk_result;
    AvxDouble<use_avx> chunk_vec1, chunk_vec2;
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
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
    for (std::uint64_t i = 0; i < remain; i++) {
        result = std::fma(vector1[i], vector2[i], result);
    }
}

// Norm of contiguous vector with chunks
void linalg::avx_norm(const double * vector, std::uint64_t nchunks, std::uint64_t remain, double & result) noexcept {
    result = 0.0;
    // calculate norm using avx on 4-double chunks
    AvxDouble<use_avx> chunk_result;
    AvxDouble<use_avx> clone_data;
    for (std::uint64_t i_chunk = 0; i_chunk < nchunks; i_chunk++) {
        clone_data = AvxDouble<use_avx>(vector);
        chunk_result.fma(clone_data, clone_data);
        vector += 4;
    }
    for (std::uint64_t i = 0; i < 4; i++) {
        result += chunk_result[i];
    }
    // add remainder
    for (std::uint64_t i = 0; i < remain; i++) {
        result = std::fma(vector[i], vector[i], result);
    }
}

}  // namespace merlin
