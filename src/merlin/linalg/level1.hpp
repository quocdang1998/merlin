// Copyright 2024 quocdang1998
#ifndef MERLIN_LINALG_LEVEL1_HPP_
#define MERLIN_LINALG_LEVEL1_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin::linalg {

// Vector-scalar arithmetic operations
// -----------------------------------

/** @brief Add a scalar to a vector and store the result.
 *  @details Add a scalar to elements of a vector with SIMD instructions:
 *  @f[ c_k \leftarrow a + b_k @f]
 *  The addition operations leverage vector instructions. Padding elements at the end of the vector must be set to
 *  zeros.
 *  @param a Summand scalar.
 *  @param b Summand vector. The memory must be aligned to the requirement of the largest possible SIMD type.
 *  @param c Result vector. This can be the same the summand vector. It memory must also be aligned.
 *  @param offset Number of parameters to ignore, counted from the beginning of the aligned memory (for input/output
 *  vectors). The offset is assumed to be less than the size of an SIMD pack.
 *  @param nsimd Number of SIMD packs to calculate.
 */
MERLIN_EXPORTS void add_vectors(double a, const double * b, double * c, std::uint64_t nsimd,
                                std::uint64_t offset = 0) noexcept;

/** @brief Subtract a vector from a scalar and store the result.
 *  @details Subtract elements of a vector from a scalar with SIMD instructions:
 *  @f[ c_k \leftarrow a - b_k @f]
 *  The subtraction operations leverage vector instructions. Padding elements at the end of the vector must be set to
 *  zeros.
 *  @param a Minuend scalar.
 *  @param b Subtrahend vector. The memory must be aligned to the requirement of the largest possible SIMD type.
 *  @param c Result vector. This can be the same the subtrahend vector. It memory must also be aligned.
 *  @param offset Number of parameters to ignore, counted from the beginning of the aligned memory (for input/output
 *  vectors). The offset is assumed to be less than the size of an SIMD pack.
 *  @param nsimd Number of SIMD packs to calculate.
 */
MERLIN_EXPORTS void subtract_vectors(double a, const double * b, double * c, std::uint64_t nsimd,
                                     std::uint64_t offset = 0) noexcept;

/** @brief Subtract a scalar from a vector and store the result.
 *  @details Subtract a scalar from elements of a vector with SIMD instructions:
 *  @f[ c_k \leftarrow a_k - b @f]
 *  The subtraction operations leverage vector instructions. Padding elements at the end of the vector must be set to
 *  zeros.
 *  @param a Minuend vector. The memory must be aligned to the requirement of the largest possible SIMD type.
 *  @param b Subtrahend scalar.
 *  @param c Result vector. This can be the same the minuend vector. It memory must also be aligned.
 *  @param offset Number of parameters to ignore, counted from the beginning of the aligned memory (for input/output
 *  vectors). The offset is assumed to be less than the size of an SIMD pack.
 *  @param nsimd Number of SIMD packs to calculate.
 */
MERLIN_EXPORTS void subtract_vectors(const double * a, double b, double * c, std::uint64_t nsimd,
                                     std::uint64_t offset = 0) noexcept;

/** @brief Multiply a scalar by a vector and store the result.
 *  @details Multiply a scalar to elements of a vector with SIMD instructions:
 *  @f[ c_k \leftarrow a b_k @f]
 *  The multiplication operations leverage vector instructions. Padding elements at the end of the vector must be set to
 *  zeros.
 *  @param a Scalar multiplier.
 *  @param b Multiplicand vector. The memory must be aligned to the requirement of the largest possible SIMD type.
 *  @param c Result vector. This can be the same the multiplicand vector. It memory must also be aligned.
 *  @param offset Number of parameters to ignore, counted from the beginning of the aligned memory (for input/output
 *  vectors). The offset is assumed to be less than the size of an SIMD pack.
 *  @param nsimd Number of SIMD packs to calculate.
 */
MERLIN_EXPORTS void multiply_vectors(double a, const double * b, double * c, std::uint64_t nsimd,
                                     std::uint64_t offset = 0) noexcept;

// Element wise operations
// -----------------------

/** @brief Add two vectors and store the result.
 *  @details Add two aligned vectors by packs:
 *  @f[ \boldsymbol{c} \leftarrow \boldsymbol{a} + \boldsymbol{b} @f]
 *  The addition operations leverage vector instructions.
 *  @param a First summand vector. The memory must be aligned to the requirement of the largest possible register vector.
 *  @param b Second summand vector. The memory must be aligned to the requirement of the largest possible register vector.
 *  @param c Result vector. This can be the same as one of the summand vectors.
 *  @param offset Number of parameters to ignore, counted from the beginning of the aligned memory (for both vectors).
 *  The offset is assumed to be less than the size of a register pack.
 *  @param npacks Number of register packs to calculate.
 */
MERLIN_EXPORTS void add_vectors(const double * a, const double * b, double * c, std::uint64_t npacks,
                                std::uint64_t offset = 0) noexcept;

/** @brief Subtract two vectors and store the result.
 *  @details Add two aligned vectors by packs:
 *  @f[ \boldsymbol{c} \leftarrow \boldsymbol{a} - \boldsymbol{b} @f]
 *  The addition operations leverage vector instructions.
 *  @param a Minuend vector. The memory must be aligned to the requirement of the largest possible register vector.
 *  @param b Subtrahend vector. The memory must be aligned to the requirement of the largest possible register vector.
 *  @param c Result vector. This can be the same as one of the input vectors.
 *  @param offset Number of parameters to ignore, counted from the beginning of the aligned memory (for both vectors).
 *  The offset is assumed to be less than the size of a register pack.
 *  @param npacks Number of register packs to calculate.
 */
MERLIN_EXPORTS void subtract_vectors(const double * a, const double * b, double * c, std::uint64_t npacks,
                                     std::uint64_t offset = 0) noexcept;


}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_LEVEL1_HPP_
