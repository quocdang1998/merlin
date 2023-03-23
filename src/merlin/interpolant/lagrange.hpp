// Copyright 2023 quocdang1998
#ifndef MERLIN_INTERPOLANT_LAGRANGE_HPP_
#define MERLIN_INTERPOLANT_LAGRANGE_HPP_

#include <cstdint>  // std::uint64_t, std::uintptr_t

#include "merlin/array/declaration.hpp"  // merlin::array::Array, merlin::array::Slice
#include "merlin/cuda/stream.hpp"  // merlin::cuda::Stream
#include "merlin/env.hpp"  // merlin::Environment
#include "merlin/interpolant/grid.hpp"  // merlin::interpolant::CartesianGrid
#include "merlin/vector.hpp"  // merlin::Vector

namespace merlin::interpolant {

// GPU kernel wrapper
// ------------------

/** @brief Call the GPU kernel calculating the coefficient with Lagrange method.
 *  @param p_grid Pointer to Cartesian grid pre-allocated on GPU.
 *  @param p_value Pointer to value array pre-allocated on GPU.
 *  @param p_coeff Pointer to coefficient array pre-allocated on GPU.
 *  @param shared_mem_size Size (in bytes) of the block-wise shared memory.
 *  @param stream_ptr Pointer to the CUDA calculation stream in form of an unsigned integer pointer.
 *  @param n_thread Number of CUDA threads for parallel execution.
 *  @note This function is asynchronious. It simply push the CUDA kernel to the stream.
 */
void call_lagrange_coeff_kernel(const interpolant::CartesianGrid * p_grid, const array::Parcel * p_value,
                                array::Parcel * p_coeff, std::uint64_t shared_mem_size, std::uintptr_t stream_ptr,
                                std::uint64_t n_thread);

// Calculate coefficients
// ----------------------

/** @brief Calculate Lagrange interpolation coefficients on a full Cartesian grid using CPU.
 *  @param grid Cartesian grid.
 *  @param value Array of function values, must have the same shape as the grid.
 *  @param coeff Array storing interpolation coefficient after the calculation.
 */
void calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::Array & value,
                              array::Array & coeff);

/** @brief Calculate Lagrange interpolation coefficients on a full Cartesian grid using GPU.
 *  @param grid Cartesian grid.
 *  @param value Array of function values, must have the same shape as the grid.
 *  @param coeff Array storing interpolation coefficient after the calculation.
 *  @param stream CUDA stream of execution of the CUDA kernel.
 *  @note This is asynchronous calculation. User should call ``merlin::cuda::Stream::synchronize(void)`` to force the
 *  CPU to wait until the calculation has finished.
 */
void calc_lagrange_coeffs_gpu(const interpolant::CartesianGrid & grid, const array::Parcel & value,
                              array::Parcel & coeff, const cuda::Stream & stream = cuda::Stream(),
                              std::uint64_t n_thread = Environment::default_block_size);

/** @brief Calculate Lagrange interpolation coefficients on a sparse grid using CPU.
 *  @param grid Sparse grid.
 *  @param value Array of function values, must have the same shape as the grid.
 *  @param coeff Array storing interpolation coefficient after the calculation.
 */
void calc_lagrange_coeffs_cpu(const interpolant::SparseGrid & grid, const array::Array & value,
                              array::Array & coeff);

// Evaluate interpolation
// ----------------------

/** @brief Evaluate Lagrange interpolation on a full Cartesian grid using CPU.
 *  @param grid Cartesian grid.
 *  @param coeff Calculated coefficients.
 *  @param x Evaluate point, must have the same dimension as grid and coeff.
 */
double eval_lagrange_cpu(const interpolant::CartesianGrid & grid, const array::Array & coeff,
                         const Vector<double> & x);

/** @brief Calculate Lagrange interpolation coefficients on a sparse grid using CPU.
 *  @param grid Sparse grid.
 *  @param coeff Calculated coefficients.
 *  @param x Evaluate point, must have the same dimension as grid and coeff.
 */
double eval_lagrange_cpu(const interpolant::SparseGrid & grid, const array::Array & coeff,
                         const Vector<double> & x);

#ifdef __comment
/** @brief Calculate Lagrage interpolation coefficients on a Cartesian grid using CPU.*/
void calc_lagrange_coeffs_cpu(const interpolant::CartesianGrid & grid, const array::NdData & value,
                              const Vector<array::Slice> & slices, array::NdData & coeff);

/** @brief Calculate Lagrage interpolation coefficients on a sparse grid using CPU.*/
void calc_lagrange_coeffs_cpu(const interpolant::SparseGrid & grid, array::NdData & coeff);

/** @brief Evaluate Lagrange interpolation on a Cartesian grid using CPU.*/
double eval_lagrange_cpu(const interpolant::CartesianGrid & grid, const array::NdData & coeff,
                         const Vector<array::Slice> & slices, const Vector<double> & x);

/** @brief Evaluate Lagrange interpolation on a Sparse grid using CPU.*/
double eval_lagrange_cpu(const interpolant::SparseGrid & grid, const array::NdData & coeff, const Vector<double> & x);
#endif

}  // namespace merlin::interpolant

#endif  // MERLIN_INTERPOLANT_LAGRANGE_HPP_
