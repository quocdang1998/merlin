// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_QR_SOLVE_HPP_
#define MERLIN_LINALG_QR_SOLVE_HPP_

#include "merlin/cuda_decorator.hpp"  // merlin::linalg::Matrix
#include "merlin/linalg/declaration.hpp"  // __cuhostdev__
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::linalg {

/** @brief Solve linear system by QR decomposition by CPU.
 *  @details Solve the system @f$ \mathbf{M} \boldsymbol{x} = \boldsymbol{b} @f$ by the stable QR decomposition method
 *  of Householder. After the procedure, the matrix @f$ \mathbf{M} @f$ is reduced to identity matrix, while the
 *  solution is stored to the vector @f$ \boldsymbol{x} @f$.
 * 
 *  This function will not throw any exception. It expects the dimensions of the matrix and the size of the vector are
 *  equals.
 *  @param M Linear system to solve.
 *  @param x Input vector to solve.
 *  @param nthreads Number of threads to perform the calculation.
 */
void qr_solve_cpu(linalg::Matrix & M, floatvec & x, std::uint64_t nthreads = 1) noexcept;

/** @brief QR decomposition by CPU parallelism.
 *  @details After the transformation, the matrix is reduced to upper right form, while the transformation is applied to
 *  the vector.
 *  @param matrix Matrix to be reduced to upper right form.
 *  @param x %Vector to solve.
 *  @param nthreads Number of parallel threads for solving.
 */
void qr_decomposition_cpu(linalg::Matrix & matrix, floatvec & x, std::uint64_t nthreads = 1) noexcept;

/** @brief Householder reflection by CPU.
 *  @details Equivalent to multiplying the matrix @f$ \mathbf{M} @f$ and @f$ \boldsymbol{x} @f$ to the left by:
 *
 *  @f[ \mathbf{I} - 2 \boldsymbol{v} \boldsymbol{v}^\intercal @f]
 *  @param M Matrix to apply the transformation.
 *  @param x %Vector to apply the transformation.
 *  @param v %Vector to reflect over.
 *  @param start_dimension Dimension before which the reflection is not applied.
 *  @param nthreads Number of threads to perform the calculation over the matrix.
 */
void householder_cpu(linalg::Matrix & M, floatvec & x, const floatvec & v, std::uint64_t start_dimension = 0,
                     std::uint64_t nthreads = 1) noexcept;

/** @brief Solve upper right linear system by CPU.
 *  @details Solve the linear system @f$ \mathbf{R} \boldsymbol{a} = \boldsymbol{x} @f$, in which @f$ \mathbf{R} @f$ is
 *  an upper right triangular matrix.
 *  @param R Upper right triangular matrix.
 *  @param x %Vector to solve and store the solution.
 *  @param nthreads Number of parallel threads for solving.
 */
void upright_solver_cpu(linalg::Matrix & R, floatvec & x, std::uint64_t nthreads = 1) noexcept;

#ifdef __NVCC__

/** @brief Solve linear system by QR decomposition on GPU.
 *  @details Solve the system @f$ \mathbf{M} \boldsymbol{x} = \boldsymbol{b} @f$ by the stable QR decomposition method
 *  of Householder. After the procedure, the matrix @f$ \mathbf{M} @f$ is reduced to identity matrix, while the
 *  solution is stored to the vector @f$ \boldsymbol{x} @f$.
 *  @param M Matrix to apply the transformation.
 *  @param x %Vector to apply the transformation (should be copied to shared memory).
 *  @param buffer Pointer to a shared memory location with the size of ``double[x.size()+block_size]`` for calculation.
 *  @param thread_idx Index of the current thread in block.
 *  @param block_size Size of the current block.
 */
__cudevice__ void qr_solve_gpu(linalg::Matrix & M, floatvec & x, double * buffer, std::uint64_t thread_idx,
                               std::uint64_t block_size) noexcept;

/** @brief QR decomposition by GPU parallelism.
 *  @param matrix Matrix to apply the transformation.
 *  @param x %Vector to apply the transformation (should be copied to shared memory).
 *  @param buffer Pointer to a shared memory location with the size of ``double[x.size()+block_size]`` for calculation.
 *  @param thread_idx Index of the current thread in block.
 *  @param block_size Size of the current block.
 */
__cudevice__ void qr_decomposition_gpu(linalg::Matrix & matrix, floatvec & x, double * buffer, std::uint64_t thread_idx,
                                       std::uint64_t block_size) noexcept;

/** @brief Householder reflection by GPU.
 *  @param M Matrix to apply the transformation.
 *  @param x %Vector to apply the transformation (should be copied to shared memory).
 *  @param v %Vector to reflect over (should be copied to shared memory).
 *  @param start_dimension Dimension before which the reflection is not applied.
 *  @param buffer Pointer to a shared memory location, with the size at least ``double[x.size()]`` for calculation.
 *  @param thread_idx Index of the current thread in block.
 *  @param block_size Size of the current block.
 */
__cudevice__ void householder_gpu(linalg::Matrix & M, floatvec & x, const floatvec & v, std::uint64_t start_dimension,
                                  double * buffer, std::uint64_t thread_idx, std::uint64_t block_size) noexcept;

/** @brief Solve upper right linear system by GPU.
 *  @param R Upper right triangular matrix.
 *  @param x %Vector to solve and store the solution.
 *  @param thread_idx Index of the current thread in block.
 *  @param block_size Size of the current block.
 */
__cudevice__ void upright_solver_gpu(linalg::Matrix & R, floatvec & x, std::uint64_t thread_idx,
                                     std::uint64_t block_size) noexcept;

#endif  // __NVCC__

}  // namespace merlin::linalg

#endif  // MERLIN_LINALG_QR_SOLVE_HPP_
