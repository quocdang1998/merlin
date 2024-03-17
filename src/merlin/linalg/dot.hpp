// Copyright 2023 quocdang1998
#ifndef MERLIN_LINALG_SEQUENCE_HPP_
#define MERLIN_LINALG_SEQUENCE_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::linalg {

// Norm
// ----

/** @brief Norm of contiguous vector calculated without using AVX.*/
MERLIN_EXPORTS void __norm_no_avx(double * vector, std::uint64_t size, double & result) noexcept;

/** @brief Norm of contiguous vector (with 256-bits register AVX optimization).*/
MERLIN_EXPORTS void __norm_256_avx(double * vector, std::uint64_t size, double & result) noexcept;

/** @brief Norm of contiguous vector.
 *  @details Calculate norm of a vector using 1 CPU thread.
 *  @param vector Pointer to the first element of the vector.
 *  @param size Number of elements in the vector.
 *  @param result Returned norm.
 */
void norm(double * vector, std::uint64_t size, double & result) noexcept {
#ifdef __AVX__
    linalg::__norm_256_avx(vector, size, result);
#else
    linalg::__norm_no_avx(vector, size, result);
#endif  // __AVX__
}

// Dot product
// -----------

/** @brief Dot product of contiguous vectors calculated without using AVX.*/
MERLIN_EXPORTS void __dot_no_avx(double * vector1, double * vector2, std::uint64_t size, double & result) noexcept;

/** @brief Dot product of contiguous vectors (with 256-bits register AVX optimization).*/
MERLIN_EXPORTS void __dot_256_avx(double * vector1, double * vector2, std::uint64_t size, double & result) noexcept;

/** @brief Dot product of 2 contiguous vector.
 *  @details Calculate dot product of 2 vectors using 1 CPU thread.
 *  @param vector1 Pointer to the first element of the first vector.
 *  @param vector2 Pointer to the first element of the second vector.
 *  @param size Number of elements in the vector.
 *  @param result Returned dot-product.
 */
void dot(double * vector1, double * vector2, std::uint64_t size, double & result) noexcept {
#ifdef __AVX__
        __dot_256_avx(vector1, vector2, size, result);
#else
        __dot_no_avx(vector1, vector2, size, result);
#endif  // __AVX__
}

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_SEQUENCE_HPP_
