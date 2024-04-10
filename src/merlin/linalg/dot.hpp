// Copyright 2023 quocdang1998
#ifndef MERLIN_LINALG_SEQUENCE_HPP_
#define MERLIN_LINALG_SEQUENCE_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::linalg {

// Norm
// ----

/** @brief Norm of contiguous vector.
 *  @details Calculate norm of a vector using 1 CPU thread.
 *  @param vector Pointer to the first element of the vector.
 *  @param size Number of elements in the vector.
 *  @param result Returned norm.
 */
MERLIN_EXPORTS void norm(const double * vector, std::uint64_t size, double & result) noexcept;

/** @brief Normalize a contiguous vector.
 *  @param src_vector Pointer to the vector to normalize.
 *  @param dest_vector Pointer to the location to write normalized value. Can be the same as source vector.
 *  @param size Size of vector.
 */
MERLIN_EXPORTS void normalize(const double * src_vector, double * dest_vector, std::uint64_t size) noexcept;

// Dot product
// -----------

/** @brief Dot product of 2 contiguous vector.
 *  @details Calculate dot product of 2 vectors using 1 CPU thread.
 *  @param vector1 Pointer to the first element of the first vector.
 *  @param vector2 Pointer to the first element of the second vector.
 *  @param size Number of elements in each vector.
 *  @param result Returned dot-product.
 */
MERLIN_EXPORTS void dot(const double * vector1, const double * vector2, std::uint64_t size, double & result) noexcept;

// Vector operation
// ----------------

/** @brief %Vector operation with another vector.
 *  @details Perform the operation @f$ \boldsymbol{y} = a \boldsymbol{x} + \boldsymbol{y} @f$.
 */
MERLIN_EXPORTS void saxpy(double a, const double * x, double * y, std::uint64_t size) noexcept;

// Householder reflection
// ----------------------

/** @brief Apply Householder reflection on a vector.
 *  @details Calculate Householder reflection on a target vector @f$ \boldsymbol{x} @f$ through a reflector vector
 *  @f$ \boldsymbol{v} @f$:
 *  @f[ \boldsymbol{H}(\boldsymbol{v}) \boldsymbol{x} = \boldsymbol{x} - 2 (\boldsymbol{x} \cdot \boldsymbol{v})
 *  \boldsymbol{v} @f]
 *  @note The reflector vector must be normalized.
 *  @param reflector Pointer to the first element of the reflector vector.
 *  @param target Pointer to the first element of the target vector.
 *  @param size Number of elements in each vector.
 *  @param range Number of elements in the target vector to apply the transformation on.
 */
MERLIN_EXPORTS void householder(const double * reflector, double * target, std::uint64_t size,
                                std::uint64_t range) noexcept;

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_SEQUENCE_HPP_
