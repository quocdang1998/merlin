// Copyright 2023 quocdang1998
#include "merlin/linalg/dot.hpp"

#include <array>  // std::array
#include <cmath>  // std::fma

#ifdef __AVX__
#include <immintrin.h>
#endif  // __AVX__

#include "merlin/logger.hpp"  // WARNING

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Norm
// ---------------------------------------------------------------------------------------------------------------------

// Norm of contiguous vector calculated without using AVX
void linalg::__norm_no_avx(double * vector, std::uint64_t size, double & result) noexcept {
    result = 0.0;
    std::uint64_t num_iteration = size / 4, remainder = size % 4;
    std::array<double, 4> reg;
    // vectorize dot product on 4-double chunks
    for (std::uint64_t iter = 0; iter < num_iteration; iter++) {
        for (std::uint64_t j = 0; j < 4; j++) {
            reg[j] = std::fma(vector[4 * iter + j], vector[4 * iter + j], reg[j]);
        }
    }
    for (std::uint64_t i = 0; i < 4; i++) {
        result += reg[i];
    }
    // add remainder
    for (std::uint64_t i = 4 * num_iteration; i < size; i++) {
        result = std::fma(vector[i], vector[i], result);
    }
}

#ifdef __AVX__

// Norm of contiguous vector (with 256-bits register AVX optimization)
void linalg::__norm_256_avx(double * vector, std::uint64_t size, double & result) noexcept {
    result = 0.0;
    std::uint64_t num_avx_iteration = size / 4, remainder = size % 4;
    // calculate norm using avx on 4-double chunks
    ::__m256d avx_result = ::_mm256_set1_pd(0.0);
    ::__m256d avx_clone;
    for (std::uint64_t avx_iteration = 0; avx_iteration < num_avx_iteration; avx_iteration++) {
        avx_clone = ::_mm256_loadu_pd(vector + 4 * avx_iteration);
        avx_result = ::_mm256_fmadd_pd(avx_clone, avx_clone, avx_result);
    }
    for (std::uint64_t i = 0; i < 4; i++) {
        result += *(reinterpret_cast<double *>(&avx_result) + i);
    }
    // add remainder
    for (std::uint64_t i = 4 * num_avx_iteration; i < size; i++) {
        result = std::fma(vector[i], vector[i], result);
    }
}

#endif  // __AVX__

// ---------------------------------------------------------------------------------------------------------------------
// Dot Product
// ---------------------------------------------------------------------------------------------------------------------

// Dot product of contiguous vector calculated without using AVX
void linalg::__dot_no_avx(double * vector1, double * vector2, std::uint64_t size, double & result) noexcept {
    result = 0.0;
    std::uint64_t num_iteration = size / 4, remainder = size % 4;
    std::array<double, 4> reg;
    // vectorize dot product on 4-double chunks
    for (std::uint64_t iter = 0; iter < num_iteration; iter++) {
        for (std::uint64_t j = 0; j < 4; j++) {
            reg[j] = std::fma(vector1[4 * iter + j], vector2[4 * iter + j], reg[j]);
        }
    }
    for (std::uint64_t i = 0; i < 4; i++) {
        result += reg[i];
    }
    // add remainder
    for (std::uint64_t i = 4 * num_iteration; i < size; i++) {
        result = std::fma(vector1[i], vector2[i], result);
    }
}

#ifdef __AVX__

// Dot product of contiguous vectors (with 256-bits register AVX optimization)
void linalg::__dot_256_avx(double * vector1, double * vector2, std::uint64_t size, double & result) noexcept {
    result = 0.0;
    std::uint64_t num_avx_iteration = size / 4, remainder = size % 4;
    // calculate dot product using avx on 4-double chunks
    ::__m256d avx_result = ::_mm256_set1_pd(0.0);
    ::__m256d avx_clone1, avx_clone2;
    for (std::uint64_t avx_iteration = 0; avx_iteration < num_avx_iteration; avx_iteration++) {
        avx_clone1 = ::_mm256_loadu_pd(vector1 + 4 * avx_iteration);
        avx_clone2 = ::_mm256_loadu_pd(vector2 + 4 * avx_iteration);
        avx_result = ::_mm256_fmadd_pd(avx_clone1, avx_clone2, avx_result);
    }
    for (std::uint64_t i = 0; i < 4; i++) {
        result += *(reinterpret_cast<double *>(&avx_result) + i);
    }
    // add remainder
    for (std::uint64_t i = 4 * num_avx_iteration; i < size; i++) {
        result = std::fma(vector1[i], vector2[i], result);
    }
}

#endif  // __AVX__

}  // namespace merlin
