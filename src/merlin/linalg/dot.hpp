// Copyright 2023 quocdang1998
#ifndef MERLIN_LINALG_SEQUENCE_HPP_
#define MERLIN_LINALG_SEQUENCE_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::linalg {

// Norm
// ----

/** @brief Norm of contiguous vector calculated without using AVX.*/
MERLIN_EXPORTS void __norm_no_avx(double * vector, std::uint64_t size, double & result);

/** @brief Norm of contiguous vector (with 256-bits register AVX optimization).*/
MERLIN_EXPORTS void __norm_256_avx(double * vector, std::uint64_t size, double & result);

/** @brief Norm of contiguous vector.
 *  @details Calculate norm of a vector using 1 CPU thread, with/without acceleration from AVX.
 *  @tparam use_256avx Use 256-bits register AVX to accelerate calculation.
 *  @note Result obtained with AVX optimization may not be the same as calculated without AVX.
 *  @param vector Pointer to the first element of the vector.
 *  @param size Number of elements in the vector.
 *  @param result Returned norm.
 */
template <bool use_256avx = true>
void norm(double * vector, std::uint64_t size, double & result) {
    if constexpr (use_256avx) {
        __norm_256_avx(vector, size, result);
    } else {
        __norm_no_avx(vector, size, result);
    }
}

// Dot product
// -----------

/** @brief Dot product of contiguous vectors calculated without using AVX.*/
MERLIN_EXPORTS void __dot_no_avx(double * vector1, double * vector2, std::uint64_t size, double & result);

/** @brief Dot product of contiguous vectors (with 256-bits register AVX optimization).*/
MERLIN_EXPORTS void __dot_256_avx(double * vector1, double * vector2, std::uint64_t size, double & result);

/** @brief Dot product of 2 contiguous vector.
 *  @details Calculate dot product of 2 vectors using 1 CPU thread, with/without acceleration from AVX.
 *  @tparam use_256avx Use 256-bits register AVX to accelerate calculation.
 *  @note Result obtained with AVX optimization may not be the same as calculated without AVX.
 *  @param vector1 Pointer to the first element of the first vector.
 *  @param vector2 Pointer to the first element of the second vector.
 *  @param size Number of elements in the vector.
 *  @param result Returned dot-product.
 */
template <bool use_256avx = true>
void dot(double * vector1, double * vector2, std::uint64_t size, double & result) {
    if constexpr (use_256avx) {
        __dot_256_avx(vector1, vector2, size, result);
    } else {
        __dot_no_avx(vector1, vector2, size, result);
    }
}

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_SEQUENCE_HPP_
