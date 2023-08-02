// Copyright 2023 quocdang1998
#ifndef MERLIN_INTPL_LAGRANGE_HPP_
#define MERLIN_INTPL_LAGRANGE_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Slice
#include "merlin/cuda/stream.hpp"        // merlin::cuda::Stream
#include "merlin/cuda_interface.hpp"     // __cuhostdev__
#include "merlin/env.hpp"                // merlin::Environment
#include "merlin/intpl/grid.hpp"         // merlin::intpl::CartesianGrid
#include "merlin/vector.hpp"             // merlin::Vector

namespace merlin::intpl {

// GPU kernel wrapper
// ------------------

/** @brief Call the GPU kernel calculating the coefficient with Lagrange method.
 *  @param p_grid Pointer to Cartesian grid pre-allocated on GPU.
 *  @param p_value Pointer to value array pre-allocated on GPU.
 *  @param p_coeff Pointer to coefficient array pre-allocated on GPU.
 *  @param shared_mem_size Size (in bytes) of the block-wise shared memory.
 *  @param stream_ptr Pointer to the CUDA calculation stream in form of an unsigned integer pointer.
 *  @param n_thread Number of CUDA threads for parallel execution.
 *  @note This function is asynchronous. It simply push the CUDA kernel to the stream.
 */
void call_lagrange_coeff_kernel(const intpl::CartesianGrid * p_grid, const array::Parcel * p_value,
                                array::Parcel * p_coeff, std::uint64_t shared_mem_size, std::uintptr_t stream_ptr,
                                std::uint64_t n_thread);

/** @brief Evaluate Lagrange interpolation on a full Cartesian grid.
 *  @param grid Cartesian grid.
 *  @param coeff Calculated coefficients.
 *  @param x Evaluate point, must have the same dimension as grid and coeff.
 *  @param collapsed_index_data Pointer to pre-allocated memory for storing collapsed index array (minimum size is the
 *  number of dimension).
 *  @param whole_coeff_data Same purpose as the former argument.
 *  @note In case of default arguments (``nullptr`` is passed tot he function), the program will allocate the memory
 *  automatically at every call, which resulted in a poor performance, especially on GPU.
 */
__cuhostdev__ double eval_lagrange_single_core(const intpl::CartesianGrid & grid, const array::NdData & coeff,
                                               const Vector<double> & x, std::uint64_t * collapsed_index_data = nullptr,
                                               std::uint64_t * whole_coeff_data = nullptr);

/** @brief Call the GPU kernel evaluating Lagrange interpolation.
 *  @param p_grid Pointer to Cartesian grid pre-allocated on GPU.
 *  @param p_coeff Pointer to coefficient array pre-allocated on GPU.
 *  @param p_points Pointer to coordinates of points array pre-allocated on GPU.
 *  @param p_result Pointer to result pre-allocated on GPU.
 *  @param shared_mem_size Size (in bytes) of the block-wise shared memory.
 *  @param stream_ptr Pointer to the CUDA calculation stream in form of an unsigned integer pointer.
 *  @param n_thread Number of CUDA threads for parallel execution.
 *  @note This function is asynchronous. It simply push the CUDA kernel to the stream.
 */
void call_lagrange_eval_kernel(const intpl::CartesianGrid * p_grid, const array::Parcel * p_coeff,
                               const array::Parcel * p_points, Vector<double> * p_result, std::uint64_t shared_mem_size,
                               std::uintptr_t stream_ptr, std::uint64_t n_thread);

// Calculate coefficients
// ----------------------

/** @brief Calculate Lagrange interpolation coefficients on a full Cartesian grid using CPU.
 *  @param grid Cartesian grid.
 *  @param value Array of function values, must have the same shape as the grid.
 *  @param coeff Array storing interpolation coefficient after the calculation.
 *  @param nthreads Number of parallel threads.
 */
void calc_lagrange_coeffs_cpu(const intpl::CartesianGrid & grid, const array::Array & value, array::Array & coeff,
                              std::uint64_t nthreads = 1);

/** @brief Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU.
 *  @details The execution is performed by one single CUDA block. The number of threads in the block is ``96`` by
 *  default.
 *  @param grid Cartesian grid.
 *  @param value Array of function values, must have the same shape as the grid.
 *  @param coeff Array storing interpolation coefficient after the calculation.
 *  @param stream CUDA stream of execution of the CUDA kernel.
 *  @param n_thread Number of CUDA threads in the execution block.
 *  @note This is asynchronous calculation. User should call ``merlin::cuda::Stream::synchronize(void)`` to force the
 *  CPU to wait until the calculation has finished.
 */
void calc_lagrange_coeffs_gpu(const intpl::CartesianGrid & grid, const array::Parcel & value, array::Parcel & coeff,
                              const cuda::Stream & stream = cuda::Stream(),
                              std::uint64_t n_thread = Environment::default_block_size);

/** @brief Calculate Lagrange interpolation coefficients on a sparse grid using CPU.
 *  @param grid Sparse grid.
 *  @param value Array of function values, must have the same shape as the grid.
 *  @param coeff Array storing interpolation coefficient after the calculation.
 */
void calc_lagrange_coeffs_cpu(const intpl::SparseGrid & grid, const array::Array & value, array::Array & coeff);

// Evaluate interpolation
// ----------------------

/** @brief Evaluate Lagrange interpolation on a full Cartesian grid using CPU.
 *  @param grid Cartesian grid.
 *  @param coeff Calculated coefficients.
 *  @param x Evaluate point, must have the same dimension as grid and coeff.
 */
double eval_lagrange_cpu(const intpl::CartesianGrid & grid, const array::Array & coeff, const Vector<double> & x);

/** @brief Evaluate Lagrange interpolation on a full Cartesian grid using GPU.
 *  @param grid Cartesian grid.
 *  @param coeff Calculated coefficients.
 *  @param points 2D array of shape (``npoint``, ``ndim``), storing coordinates of points to interpolate.
 *  @param stream CUDA stream of execution of the CUDA kernel.
 *  @param n_thread Number of CUDA threads in the execution block.
 *  @note This is asynchronous calculation. User should call ``merlin::cuda::Stream::synchronize(void)`` to force the
 *  CPU to wait until the calculation has finished.
 */
Vector<double> eval_lagrange_gpu(const intpl::CartesianGrid & grid, const array::Parcel & coeff,
                                 const array::Parcel & points, const cuda::Stream & stream = cuda::Stream(),
                                 std::uint64_t n_thread = Environment::default_block_size);

/** @brief Calculate Lagrange interpolation coefficients on a sparse grid using CPU.
 *  @param grid Sparse grid.
 *  @param coeff Calculated coefficients.
 *  @param x Evaluate point, must have the same dimension as grid and coeff.
 */
double eval_lagrange_cpu(const intpl::SparseGrid & grid, const array::Array & coeff, const Vector<double> & x);

}  // namespace merlin::intpl

#endif  // MERLIN_INTPL_LAGRANGE_HPP_
