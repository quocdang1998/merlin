// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_QR_SOLVE_HPP_
#define MERLIN_LINALG_QR_SOLVE_HPP_

#include "merlin/cuda_interface.hpp"      // __cuhostdev__, __cudevice__
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/linalg/declaration.hpp"  // merlin::linalg::Matrix
#include "merlin/vector.hpp"              // merlin::Vector

namespace merlin::linalg {

/** @brief Householder reflection.
 *  @details Equivalent to multiplying the matrix @f$ \mathbf{M} @f$ and @f$ \mathbf{B} @f$ to the left by:
 *
 *  @f[ \mathbf{I} - 2 \boldsymbol{v} \boldsymbol{v}^\intercal @f]
 *  @param M Matrix to apply the transformation.
 *  @param v %Vector to reflect over.
 *  @param start_dimension Dimension before which the reflection is not applied.
 *  @param thread_idx Index of the current thread inside the thread group.
 *  @param nthreads Number of threads to perform the calculation over the matrix.
 */
__cuhostdev__ void householder_reflect(linalg::Matrix & M, const floatvec & v, std::uint64_t start_dimension,
                                       std::uint64_t thread_idx, std::uint64_t nthreads) noexcept;

/** @brief Solve upper right linear system.
 *  @details Transform the matrix @f$ \mathbf{B} \longleftarrow \mathbf{R}^{-1} \mathbf{B} @f$, in which @f$ \mathbf{R}
 *  @f$ is an upper right triangular matrix.
 *  @param R Upper right triangular matrix (expected to be a square invertible matrix).
 *  @param B Matrix to apply the transformation on (expected to have the same number of rows as R).
 *  @param thread_idx Index of the current thread inside the thread group.
 *  @param nthreads Number of parallel threads for solving.
 */
__cuhostdev__ void upright_solver(const linalg::Matrix & R, linalg::Matrix & B, std::uint64_t thread_idx,
                                  std::uint64_t nthreads) noexcept;

/** @brief QR decomposition by CPU parallelism.
 *  @details The square matrix @f$ \mathbf{M} @f$ is decomposed into @f$ \mathbf{M} = \mathbf{Q} \mathbf{R} @f$, in
 *  which @f$ \mathbf{Q} @f$ is an orthogonal matrix and @f$ \mathbf{R} @f$ is an upper right triangular matrix. After
 *  the transformation, @f$ \mathbf{M} \longleftarrow \mathbf{R} @f$ and @f$ \mathbf{B} \longleftarrow
 *  \mathbf{Q}^\intercal \mathbf{B} @f$.
 *  @param M Matrix to be reduced to upper right form (expected to be a square matrix).
 *  @param B Matrix to apply the transformation on (expected to have the same number of rows as M).
 *  @param buffer Pointer to a shared memory location with the size of ``double[M.nrow()]`` for calculation.
 *  @param norm Shared memory for calculation.
 *  @param thread_idx Index of the current thread inside the thread group.
 *  @param nthreads Number of threads to perform the calculation over the matrix.
 */
MERLIN_EXPORTS void qr_decomposition_cpu(linalg::Matrix & M, linalg::Matrix & B, double * buffer, double & norm,
                                         std::uint64_t thread_idx, std::uint64_t nthreads) noexcept;

/** @brief Solve linear system by QR decomposition by CPU parallelism.
 *  @details Calculate inversion of a matrix by QR decomposition with Householder reflection method. the matrix @f$
 *  \mathbf{M} @f$ is decomposed into @f$ \mathbf{M} = \mathbf{Q} \mathbf{R} @f$, in which @f$ \mathbf{Q} @f$ is an
 *  orthogonal matrix and @f$ \mathbf{R} @f$ is an upper right triangular matrix. After the transformation,
 *  @f$ \mathbf{M} \longleftarrow \mathbf{R} @f$ and @f$ \mathbf{B} \longleftarrow \mathbf{M}^{-1} \mathbf{B} @f$.
 *
 *  This function will not throw any exception. It expects the number of rows of the matrices @f$ \mathbf{M} @f$ and
 *  @f$ \mathbf{B} @f$ are equal.
 *  @param M Linear system to solve (expected to be a square matrix).
 *  @param B Matrix to apply the transformation on (expected to have the same number of rows as M).
 *  @param buffer Pointer to a shared memory location with the size of ``double[M.nrow()]`` for calculation.
 *  @param norm Shared memory for calculation.
 *  @param thread_idx Index of the current thread inside the thread group.
 *  @param nthreads Number of threads to perform the calculation.
 */
MERLIN_EXPORTS void qr_solve_cpu(linalg::Matrix & M, linalg::Matrix & B, double * buffer, double & norm,
                                 std::uint64_t thread_idx, std::uint64_t nthreads) noexcept;

#ifdef __NVCC__

/** @brief QR decomposition by GPU parallelism.
 *  @details The square matrix @f$ \mathbf{M} @f$ is decomposed into @f$ \mathbf{M} = \mathbf{Q} \mathbf{R} @f$, in
 *  which @f$ \mathbf{Q} @f$ is an orthogonal matrix and @f$ \mathbf{R} @f$ is an upper right triangular matrix. After
 *  the transformation, @f$ \mathbf{M} \longleftarrow \mathbf{R} @f$ and @f$ \mathbf{B} \longleftarrow
 *  \mathbf{Q}^\intercal \mathbf{B} @f$.
 *  @param M Matrix to be reduced to upper right form (expected to be a square matrix).
 *  @param B Matrix to apply the transformation on (expected to have the same number of rows as M).
 *  @param buffer Pointer to a shared memory location with the size of ``double[M.nrow()]`` for calculation.
 *  @param norm Shared memory for calculation.
 *  @param thread_idx Index of the current thread inside the thread group.
 *  @param nthreads Number of threads to perform the calculation over the matrix.
 */
__cudevice__ void qr_decomposition_gpu(linalg::Matrix & M, linalg::Matrix & B, double * buffer, double & norm,
                                       std::uint64_t thread_idx, std::uint64_t nthreads) noexcept;

/** @brief Solve linear system by QR decomposition by GPU parallelism.
 *  @details Calculate inversion of a matrix by QR decomposition with Householder reflection method. the matrix @f$
 *  \mathbf{M} @f$ is decomposed into @f$ \mathbf{M} = \mathbf{Q} \mathbf{R} @f$, in which @f$ \mathbf{Q} @f$ is an
 *  orthogonal matrix and @f$ \mathbf{R} @f$ is an upper right triangular matrix. After the transformation,
 *  @f$ \mathbf{M} \longleftarrow \mathbf{R} @f$ and @f$ \mathbf{B} \longleftarrow \mathbf{M}^{-1} \mathbf{B} @f$.
 *
 *  This function will not throw any exception. It expects the number of rows of the matrices @f$ \mathbf{M} @f$ and
 *  @f$ \mathbf{B} @f$ are equal.
 *  @param M Linear system to solve (expected to be a square matrix).
 *  @param B Matrix to apply the transformation on (expected to have the same number of rows as M).
 *  @param buffer Pointer to a shared memory location with the size of ``double[M.nrow()]`` for calculation.
 *  @param norm Shared memory for calculation.
 *  @param thread_idx Index of the current thread inside the thread group.
 *  @param nthreads Number of threads to perform the calculation.
 */
__cudevice__ void qr_solve_gpu(linalg::Matrix & M, linalg::Matrix & B, double * buffer, double & norm,
                               std::uint64_t thread_idx, std::uint64_t nthreads) noexcept;

#endif  // __NVCC__

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_QR_SOLVE_HPP_
