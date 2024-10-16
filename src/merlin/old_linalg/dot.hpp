// Copyright 2023 quocdang1998
#ifndef MERLIN_LINALG_SEQUENCE_HPP_
#define MERLIN_LINALG_SEQUENCE_HPP_

#include <cmath>    // std::sqrt
#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::linalg {

// Vector operation
// ----------------

// Multiply a vector by a scalar and store it into another vector
MERLIN_EXPORTS void avx_multiply(double a, const double * x, double * y, std::uint64_t nchunks,
                                 std::uint64_t remain) noexcept;

/** @brief Multiply a vector by a scalar and store it into another vector.
 *  @details Perform the operation @f$ \boldsymbol{y} = a \boldsymbol{x} @f$.
 */
inline void multiply(double a, const double * x, double * y, std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_multiply(a, x, y, num_chunks, remainder);
}

// Vector operation with another vector
MERLIN_EXPORTS void avx_saxpy(double a, const double * x, double * y, std::uint64_t nchunks,
                              std::uint64_t remain) noexcept;

/** @brief Vector operation with another vector.
 *  @details Perform the operation @f$ \boldsymbol{y} = a \boldsymbol{x} + \boldsymbol{y} @f$.
 *  @param a Scalar.
 *  @param x Multiplier vector.
 *  @param y Result vector, can be the same as ``x``.
 *  @param size Number of elements.
 */
inline void saxpy(double a, const double * x, double * y, std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_saxpy(a, x, y, num_chunks, remainder);
}

// Swap contents of 2 vectors
MERLIN_EXPORTS void avx_swap(double * x, double * y, std::uint64_t nchunks, std::uint64_t remain) noexcept;

/** @brief Swap contents of 2 vectors.
 *  @details Swap content of 2 vectors @f$ \boldsymbol{x} @f$ and  @f$ \boldsymbol{y} @f$.
 */
inline void swap(double * x, double * y, std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_swap(x, y, num_chunks, remainder);
}

// Divide contents of 2 vectors.
MERLIN_EXPORTS void avx_vecdiv(const double * x, const double * y, double * z, std::uint64_t nchunks,
                               std::uint64_t remain) noexcept;

/** @brief Divide contents of 2 vectors.
 *  @details Element-wise division of 2 vectors @f$ \boldsymbol{z} = \boldsymbol{x} / \boldsymbol{y} @f$.
 */
inline void vecdiv(const double * x, const double * y, double * z, std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_vecdiv(x, y, z, num_chunks, remainder);
}

// Dot product and norm
// --------------------

// Dot product of 2 vectors
MERLIN_EXPORTS void avx_dot(const double * vector1, const double * vector2, std::uint64_t nchunks, std::uint64_t remain,
                            double & result) noexcept;

/** @brief Dot product of 2 vectors.
 *  @details Calculate dot product of 2 vectors using 1 CPU thread.
 *  @param vector1 Pointer to the first element of the first vector.
 *  @param vector2 Pointer to the first element of the second vector.
 *  @param size Number of elements in each vector.
 *  @param result Returned dot-product.
 */
inline void dot(const double * vector1, const double * vector2, std::uint64_t size, double & result) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_dot(vector1, vector2, num_chunks, remainder, result);
}

// Norm of contiguous vector with chunks
MERLIN_EXPORTS void avx_norm(const double * vector, std::uint64_t nchunks, std::uint64_t remain,
                             double & result) noexcept;

/** @brief Norm of contiguous vector.
 *  @details Calculate norm of a vector using 1 CPU thread.
 *  @param vector Pointer to the first element of the vector.
 *  @param size Number of elements in the vector.
 *  @param result Returned norm.
 */
inline void norm(const double * vector, std::uint64_t size, double & result) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_norm(vector, num_chunks, remainder, result);
}

// Normalize a contiguous vector
inline void avx_normalize(const double * src_vector, double * dest_vector, std::uint64_t nchunks,
                          std::uint64_t remain) noexcept {
    double norm;
    linalg::avx_norm(src_vector, nchunks, remain, norm);
    norm = 1.0 / std::sqrt(norm);
    linalg::avx_multiply(norm, src_vector, dest_vector, nchunks, remain);
}

/** @brief Normalize a contiguous vector.
 *  @param src_vector Pointer to the vector to normalize.
 *  @param dest_vector Pointer to the location to write normalized value. Can be the same as source vector.
 *  @param size Size of vector.
 */
inline void normalize(const double * src_vector, double * dest_vector, std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_normalize(src_vector, dest_vector, num_chunks, remainder);
}

// Householder reflection
// ----------------------

// Apply Householder reflection on a vector
inline void avx_householder(const double * reflector, double * target, std::uint64_t nchunks,
                            std::uint64_t remain) noexcept {
    double coeff;
    linalg::avx_dot(reflector, target, nchunks, remain, coeff);
    coeff *= -2.0;
    linalg::avx_saxpy(coeff, reflector, target, nchunks, remain);
}

/** @brief Apply Householder reflection on a vector.
 *  @details Calculate Householder reflection on a target vector @f$ \boldsymbol{x} @f$ through a reflector vector
 *  @f$ \boldsymbol{v} @f$:
 *  @f[ \boldsymbol{H}(\boldsymbol{v}) \boldsymbol{x} = \boldsymbol{x} - 2 (\boldsymbol{x} \cdot \boldsymbol{v})
 *  \boldsymbol{v} @f]
 *  @note The reflector vector must be normalized.
 *  @param reflector Pointer to the first element of the reflector vector.
 *  @param target Pointer to the first element of the target vector.
 *  @param size Number of elements in each vector.
 */
inline void householder(const double * reflector, double * target, std::uint64_t size) noexcept {
    std::uint64_t num_chunks = size / 4, remainder = size % 4;
    linalg::avx_householder(reflector, target, num_chunks, remainder);
}

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_SEQUENCE_HPP_
